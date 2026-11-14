//! GPU tests for S2's launches (docs/S2.md): each must be **byte-identical** to the launch it replaces.
//!
//! - `gated_rmsnorm` (GDN) with other tile shapes (rows x heads per tile block, `gated_rmsnorm_hb`) and occupancy
//!   hints vs the pre-S2 launch (16 rows of one head, compiler defaults), on the engine's layout (z a strided
//!   column slice of the `[T, 12352]` projection), at a partial row tile and a full bucket.
//! - The MLP down projection with the residual add in its epilogue (`gemm_nvfp4_raw_residual`) followed by
//!   `zc_rmsnorm_res` vs `gemm_nvfp4_raw` (bf16 `mo`) followed by `add_zc_rmsnorm` (the engine's pre-S2 pair):
//!   the updated residual and the normed output, on elephant-1's down shape (N = 4096, K = 12288). The plain
//!   `zc_rmsnorm` kernel on the same residual is reported too: it is *not* bit-identical to `add_zc_rmsnorm`'s norm
//!   (1-10 elements per million differ by one bf16 ulp; the compiler lowers its row reduction differently), which
//!   is why S2 adds `zc_rmsnorm_res`, `add_zc_rmsnorm` with the add removed.
//! - `zc_rmsnorm` with occupancy hints vs the compiler default (layer 0's input norm, and every input norm after a
//!   fused residual add).
//!
//! Run: `scripts/gpu.sh cargo test --release --test kernels_s2 -- --nocapture --test-threads 1`

use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use sysone::contract::DecisionBatch;
use sysone::engine::kernels::KernelTuning;
use sysone::kernels::gdn::{self, DView, GatedNormTiles, GdnPlan};
use sysone::kernels::norm;
use sysone::kernels::nvfp4::{self, CheckpointProj, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles, RawAlpha};

const H: usize = 4096;
const I: usize = 12288;
const PROJ: usize = gdn::CONV_DIM + gdn::VALUE_DIM + 2 * gdn::NUM_V_HEADS;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    /// `t` rows of `k` values: row scales 2^-8 .. 2^8, 128-column block scales 2^-4 .. 2^4, every 7th row zero.
    fn rows(&mut self, t: usize, k: usize) -> Vec<bf16> {
        let mut v = Vec::with_capacity(t * k);
        for r in 0..t {
            let row = 2f64.powi((r % 17) as i32 - 8);
            for i in 0..k {
                let x = if r % 7 == 6 { 0.0 } else { self.normal() * row * 2f64.powi(((i / 128) % 9) as i32 - 4) };
                v.push(bf16::from_f64(x));
            }
        }
        v
    }
}

fn stream() -> Arc<cuda_core::Stream> {
    static STREAM: std::sync::OnceLock<Arc<cuda_core::Stream>> = std::sync::OnceLock::new();
    STREAM.get_or_init(|| Context::new(0).unwrap().new_stream().unwrap()).clone()
}

fn sms() -> usize {
    Context::new(0).unwrap().multiprocessor_count().unwrap() as usize
}

fn upload<T: DType>(v: Vec<T>, shape: &[usize], s: &Arc<cuda_core::Stream>) -> Tensor<T> {
    api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(s).unwrap()
}

fn bits(t: impl ToHostVec<bf16>, s: &Arc<cuda_core::Stream>) -> Vec<u16> {
    t.to_host_vec().sync_on(s).unwrap().iter().map(|b| b.to_bits()).collect()
}

fn mismatches<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

fn tiles(rows: usize, heads: usize, occupancy: Option<i32>) -> GatedNormTiles {
    GatedNormTiles { rows, heads, occupancy, warps: None }
}

#[test]
fn gated_rmsnorm_tile_shapes_are_bit_identical() {
    let s = stream();
    let mut rng = Rng(0x5252_1111_aaaa_0007);
    // The engine's S2 launch first, then every other shape the sweep (bench_s2) looked at closely.
    let cfgs = [KernelTuning::DEFAULT.gated_norm, tiles(16, 1, Some(4)), tiles(8, 1, Some(8)), tiles(32, 1, None),
                tiles(16, 4, None), tiles(8, 4, None), tiles(8, 16, None), tiles(4, 32, None), tiles(2, 32, None),
                tiles(1, 32, Some(2)), tiles(1, 16, Some(2)), tiles(2, 16, Some(4)),
                GatedNormTiles { rows: 1, heads: 32, occupancy: None, warps: Some(8) }];
    let mut bad = Vec::new();
    for t in [300usize, 1024] {
        let mut b = DecisionBatch::default();
        b.push_request(&vec![1u32; t - 1], &[(&[2u32][..], 2)]).unwrap();
        let plan = GdnPlan::new(&b.segments, t, &s).unwrap();
        // z: the strided z columns of a [T, 12352] projection (as in the engine); x: the [T, 4096] scan output.
        let mut proj = rng.rows(t, PROJ);
        for (i, v) in proj.iter_mut().enumerate() {
            // Gates over the full sigmoid range, including saturated ones.
            if i % 61 == 0 {
                *v = bf16::from_f64(if i % 2 == 0 { 30.0 } else { -30.0 });
            }
        }
        let proj = upload(proj, &[t, PROJ], &s);
        let core = upload(rng.rows(t, gdn::VALUE_DIM), &[t, gdn::VALUE_DIM], &s);
        let w = upload((0..gdn::HEAD_DIM).map(|_| bf16::from_f64(1.0 + rng.normal() * 0.3)).collect(), &[gdn::HEAD_DIM], &s);
        let z = DView::of(&proj).unwrap().columns(gdn::CONV_DIM, gdn::VALUE_DIM).unwrap();
        let x = DView::of(&core).unwrap();
        let run = |cfg: Option<GatedNormTiles>| {
            let out = api::zeros::<bf16>(&[t, gdn::VALUE_DIM]).sync_on(&s).unwrap();
            let o = DView::of(&out).unwrap();
            // SAFETY: every buffer outlives the synchronous launch.
            unsafe {
                match cfg {
                    None => gdn::gated_rmsnorm(&plan, x.clone(), z.clone(), &w, norm::RMS_EPS, o).unwrap(),
                    Some(c) => gdn::gated_rmsnorm_with(&plan, x.clone(), z.clone(), &w, norm::RMS_EPS, o, c).unwrap(),
                }
                .sync_on(&s)
                .unwrap()
            };
            bits(out, &s)
        };
        let reference = run(None);
        let nonzero = reference.iter().filter(|&&b| b & 0x7fff != 0).count();
        assert!(nonzero > reference.len() / 2, "reference output mostly zero ({nonzero}/{})", reference.len());
        for cfg in cfgs {
            let m = mismatches(&run(Some(cfg)), &reference);
            println!("gated_rmsnorm T={t} {cfg:?}: mismatches {m}/{}", reference.len());
            if m > 0 {
                bad.push((t, cfg));
            }
        }
    }
    assert!(bad.is_empty(), "gated_rmsnorm launch shapes changed the output: {bad:?}");
}

/// The down projection `[4096, 12288]` with a realistic global scale.
fn down_weight(rng: &mut Rng, s: &Arc<cuda_core::Stream>) -> Nvfp4Linear {
    let p: Vec<u8> = (0..H * I / 2).map(|_| rng.next() as u8).collect();
    let sc: Vec<u8> = (0..H * I / 16).map(|_| ((4 + rng.next() % 7) << 3 | (rng.next() & 7)) as u8).collect();
    let proj = CheckpointProj { n: H, k: I, weight_packed: &p, weight_scale: &sc, weight_global_scale: 6144.0,
                                input_global_scale: 233.0 };
    Nvfp4Linear::upload(&nvfp4::prepare_weights(&[proj]).unwrap(), s).unwrap()
}

#[test]
fn down_gemm_residual_then_norm_is_bit_identical_to_gemm_then_add_norm() {
    let s = stream();
    let mut rng = Rng(0x0d0d_5252_e1e1_9999);
    let w = down_weight(&mut rng, &s);
    let alpha = RawAlpha { lo: w.alpha_lo, hi: w.alpha_hi, split_n: w.split_n };
    let nw = upload((0..H).map(|_| bf16::from_f64(rng.normal() * 0.5)).collect(), &[H], &s);
    let tl = GemmTiles::DEFAULT;
    let mut bad = Vec::new();
    for t in [300usize, 1024] {
        // Activations: the product quantized under the down projection's input scale, rows of very different size.
        let x = Arc::new(upload(rng.rows(t, I), &[t, I], &s));
        let a = nvfp4::quantize(&x, 233.0, Nvfp4Act::zeros(t, I, &s).unwrap(), QuantTiles::DEFAULT, &s).unwrap();
        let (xp, xs): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (Arc::new(a.packed), Arc::new(a.scales));
        // Residual rows on several scales relative to mo (so the add rounds in both directions and cancels).
        let mut res = rng.rows(t, H);
        for (i, v) in res.iter_mut().enumerate() {
            if (i / H) % 5 == 0 {
                *v = bf16::from_f64(v.to_f64() * 64.0);
            }
        }
        // Pre-S2: mo = gemm_nvfp4_raw; residual += mo, out = norm (add_zc_rmsnorm at the engine's F1 hint).
        let mo = Arc::new(api::zeros::<bf16>(&[t, H]).sync_on(&s).unwrap());
        let mo = nvfp4::gemm_raw_op(&xp, &xs, &w.packed, &w.scales, alpha, mo, tl, sms()).unwrap().sync_on(&s).unwrap();
        let mo_bits = bits(&mo, &s);
        assert!(mo_bits.iter().filter(|&&b| b & 0x7fff != 0).count() > mo_bits.len() / 2, "mo mostly zero");
        let mut r0 = upload(res.clone(), &[t, H], &s);
        let mut o0 = api::zeros::<bf16>(&[t, H]).sync_on(&s).unwrap();
        norm::add_zc_rmsnorm_with(&mut r0, &mut o0, &mo, &nw, norm::RMS_EPS, KernelTuning::F1.norm_occupancy)
            .unwrap()
            .sync_on(&s)
            .unwrap();
        let (r0, o0) = (bits(r0, &s), bits(o0, &s));
        // S2: residual += mo in the GEMM epilogue, then the plain norm (with and without the occupancy hint).
        for gemm_occ in [None, Some(1)] {
            let r1 = Arc::new(upload(res.clone(), &[t, H], &s));
            let r1 = nvfp4::gemm_raw_residual_op_with(&xp, &xs, &w.packed, &w.scales, alpha, r1, tl, sms(),
                                                      CompileOptions { occupancy: gemm_occ, ..Default::default() })
                .unwrap()
                .sync_on(&s)
                .unwrap();
            let rb = bits(&r1, &s);
            let mr = mismatches(&rb, &r0);
            for norm_occ in [None, Some(2)] {
                let mut o1 = api::zeros::<bf16>(&[t, H]).sync_on(&s).unwrap();
                norm::zc_rmsnorm_with(&mut o1, &r1, &nw, norm::RMS_EPS, norm_occ).unwrap().sync_on(&s).unwrap();
                let ob = bits(o1, &s);
                let mo_ = mismatches(&ob, &o0);
                println!("down residual T={t} gemm occupancy {gemm_occ:?}, plain zc_rmsnorm (reported) occupancy {norm_occ:?}: residual \
                          mismatches {mr}/{}, normed {mo_}/{}", r0.len(), o0.len());
                for (j, (a, b)) in ob.iter().zip(&o0).enumerate().filter(|(_, (a, b))| a != b).take(4) {
                    println!("    row {} col {}: {:#06x} vs {:#06x}", j / H, j % H, a, b);
                }
                // Reported only (module docs): the engine uses zc_rmsnorm_res after a fused residual add.
                if mr > 0 {
                    bad.push((t, gemm_occ, norm_occ, "residual"));
                }
            }
            let mut r1 = Arc::try_unwrap(r1).unwrap();
            for norm_occ in [None, Some(3), KernelTuning::DEFAULT.norm_occupancy] {
                let mut o1 = api::zeros::<bf16>(&[t, H]).sync_on(&s).unwrap();
                norm::zc_rmsnorm_res_with(&mut r1, &mut o1, &nw, norm::RMS_EPS, norm_occ).unwrap().sync_on(&s).unwrap();
                let mo_ = mismatches(&bits(o1, &s), &o0);
                println!("down residual T={t} gemm occupancy {gemm_occ:?}, zc_rmsnorm_res occupancy {norm_occ:?}: \
                          residual mismatches {mr}/{}, normed {mo_}/{}", r0.len(), o0.len());
                if mr + mo_ > 0 {
                    bad.push((t, gemm_occ, norm_occ, "zc_rmsnorm_res"));
                }
            }
        }
    }
    assert!(bad.is_empty(), "the residual add in the down GEMM epilogue changed the residual or the norm: {bad:?}");
}

#[test]
fn zc_rmsnorm_occupancy_hints_are_bit_identical() {
    let s = stream();
    let mut rng = Rng(0x3131_7777_0000_5555);
    let t = 257usize;
    let x = upload(rng.rows(t, H), &[t, H], &s);
    let w = upload((0..H).map(|_| bf16::from_f64(rng.normal() * 0.5)).collect(), &[H], &s);
    let mut o0 = api::zeros::<bf16>(&[t, H]).sync_on(&s).unwrap();
    norm::zc_rmsnorm(&mut o0, &x, &w, norm::RMS_EPS).unwrap().sync_on(&s).unwrap();
    let o0 = bits(o0, &s);
    let mut bad = Vec::new();
    for occ in [None, Some(2), Some(3), Some(4)] {
        let mut o1 = api::zeros::<bf16>(&[t, H]).sync_on(&s).unwrap();
        norm::zc_rmsnorm_with(&mut o1, &x, &w, norm::RMS_EPS, occ).unwrap().sync_on(&s).unwrap();
        let m = mismatches(&bits(o1, &s), &o0);
        println!("zc_rmsnorm occupancy {occ:?}: mismatches {m}/{}", o0.len());
        if m > 0 {
            bad.push(occ);
        }
    }
    assert!(bad.is_empty(), "occupancy hints changed zc_rmsnorm: {bad:?}");
}
