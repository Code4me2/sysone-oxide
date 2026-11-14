//! GPU tests for F1's NVFP4 kernels (docs/F1.md): each must be **byte-identical** to the N1 path it replaces.
//!
//! - `gemm_nvfp4_swiglu_il` (gate/up GEMM on interleaved gate/up rows with SiLU·mul + quantize in the epilogue)
//!   vs `gemm_nvfp4_raw` (bf16 `[T, 2I]` gate/up, checkpoint row order) followed by N1's `silu_mul_nvfp4`:
//!   packed values and scales, on elephant-1's MLP shape (K = 4096, I = 12288), at several T including a partial
//!   row tile, for the engine's tiles and three other shapes.
//! - `silu_mul_nvfp4_rb` (tunable tile shape / occupancy) vs N1's `silu_mul_nvfp4`.
//! - `add_zc_rmsnorm` / `add_zc_rmsnorm_nvfp4` with occupancy hints vs the compiler default.
//! - The raw GEMM itself across tile shapes with the same BK (reported; asserted for the shapes F1 relies on).
//!
//! Weights are random nibbles with e4m3 scales in the checkpoint's range; activations are real quantizations of
//! random bf16 rows with a wide magnitude spread (so the epilogue sees large, tiny and zero groups).
//!
//! Run: `scripts/gpu.sh cargo test --release --test kernels_nvfp4_f1 -- --nocapture --test-threads 1`

use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use sysone::kernels::norm;
use sysone::kernels::nvfp4::{
    self, CheckpointProj, GemmKernel, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles, RawAlpha, SiluTiles,
};

const K: usize = 4096;
const I: usize = 12288;

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
    /// `t` rows of `k` values: row scales 2^-8 .. 2^8, group scales 2^-4 .. 2^4, every 7th row zero.
    fn rows(&mut self, t: usize, k: usize) -> Vec<bf16> {
        let mut v = Vec::with_capacity(t * k);
        for r in 0..t {
            let row = 2f64.powi((r % 17) as i32 - 8);
            for i in 0..k {
                let x = if r % 7 == 6 { 0.0 } else { self.normal() * row * 2f64.powi(((i / 16) % 9) as i32 - 4) };
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

fn act_bytes(packed: &Arc<Tensor<f4e2m1fnx2>>, scales: &Arc<Tensor<f8e4m3fn>>, s: &Arc<cuda_core::Stream>) -> (Vec<u8>, Vec<u8>) {
    let p = packed.to_host_vec().sync_on(s).unwrap().iter().map(|b| b.to_bits()).collect();
    let sc = scales.to_host_vec().sync_on(s).unwrap().iter().map(|b| b.0).collect();
    (p, sc)
}

fn mismatches<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Gate/up weights `[2I, K]` (two parts with different global scales), and their host bytes (packed, scales;
/// gate rows then up rows) for building interleaved copies.
fn gate_up(rng: &mut Rng, s: &Arc<cuda_core::Stream>) -> (Nvfp4Linear, Vec<u8>, Vec<u8>) {
    let parts: Vec<(Vec<u8>, Vec<u8>)> = (0..2)
        .map(|_| {
            let p = (0..I * K / 2).map(|_| rng.next() as u8).collect();
            // e4m3 exponent fields 4..10 (scales ~2^-4 .. 2^3), random mantissa.
            let sc = (0..I * K / 16).map(|_| ((4 + rng.next() % 7) << 3 | (rng.next() & 7)) as u8).collect();
            (p, sc)
        })
        .collect();
    let projs: Vec<CheckpointProj> = parts
        .iter()
        .enumerate()
        .map(|(j, (p, sc))| CheckpointProj {
            n: I,
            k: K,
            weight_packed: p,
            weight_scale: sc,
            weight_global_scale: [9408.0, 5120.0][j],
            input_global_scale: 628.0,
        })
        .collect();
    let lin = Nvfp4Linear::upload(&nvfp4::prepare_weights(&projs).unwrap(), s).unwrap();
    let packed = [parts[0].0.clone(), parts[1].0.clone()].concat();
    let scales = [parts[0].1.clone(), parts[1].1.clone()].concat();
    (lin, packed, scales)
}

/// Device copies of the gate/up weight with rows interleaved in blocks of `block` (the engine's load-time layout).
fn interleaved(packed: &[u8], scales: &[u8], block: usize, s: &Arc<cuda_core::Stream>)
    -> (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) {
    let p = nvfp4::interleave_gate_up(packed, 2 * I, K / 2, block).unwrap();
    let sc = nvfp4::interleave_gate_up(scales, 2 * I, K / 16, block).unwrap();
    let p: Vec<f4e2m1fnx2> = p.into_iter().map(f4e2m1fnx2::from_bits).collect();
    let sc: Vec<f8e4m3fn> = sc.into_iter().map(f8e4m3fn).collect();
    (Arc::new(upload(p, &[2 * I, K / 2], s)), Arc::new(upload(sc, &[2 * I, K / 16], s)))
}

/// NVFP4 activations of `t` random rows (quantized by the standalone kernel under the gate/up input scale).
fn activations(rng: &mut Rng, t: usize, s: &Arc<cuda_core::Stream>) -> (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) {
    let x = Arc::new(upload(rng.rows(t, K), &[t, K], s));
    let out = Nvfp4Act::zeros(t, K, s).unwrap();
    let a = nvfp4::quantize(&x, 628.0, out, QuantTiles::DEFAULT, s).unwrap();
    (Arc::new(a.packed), Arc::new(a.scales))
}

fn tiles(bm: usize, bn: usize, group_m: usize) -> GemmTiles {
    GemmTiles { bm, bn, bk: 128, group_m, blocks_per_sm: 0, kernel: GemmKernel::Raw }
}

/// The N1 path: bf16 gate/up through the raw GEMM at its default tiles, then `silu_mul_nvfp4`.
fn reference(x: &Arc<Tensor<f4e2m1fnx2>>, xs: &Arc<Tensor<f8e4m3fn>>, w: &Nvfp4Linear, igs: f32, t: usize,
             s: &Arc<cuda_core::Stream>) -> (Vec<u8>, Vec<u8>, Tensor<bf16>) {
    let z = api::zeros::<bf16>(&[t, 2 * I]).sync_on(s).unwrap();
    let gu = nvfp4::gemm(x, xs, w, z, GemmTiles::DEFAULT, sms(), s).unwrap();
    let a = Nvfp4Act::zeros(t, I, s).unwrap();
    let (mut q, mut sc) = (a.packed, a.scales);
    nvfp4::silu_mul_nvfp4(&mut q, &mut sc, &gu, igs).unwrap().sync_on(s).unwrap();
    let (p, b) = act_bytes(&Arc::new(q), &Arc::new(sc), s);
    (p, b, gu)
}

/// Down-projection input scales: the checkpoint's range and one that saturates most e4m3 scales at 448.
const IGS: [f32; 2] = [233.0, 2.0e5];

#[test]
fn gemm_swiglu_is_bit_identical_to_gemm_then_silu_mul_nvfp4() {
    let s = stream();
    let mut rng = Rng(0x1f1f_2e2e_3d3d_4c4c);
    let (w, packed, scales) = gate_up(&mut rng, &s);
    let alpha = RawAlpha { lo: w.alpha_lo, hi: w.alpha_hi, split_n: w.split_n };
    assert_eq!(w.split_n, I);
    // The engine's tiles (128x128, block 64) without and with its occupancy hint, then other shapes.
    let cfgs = [(GemmTiles::DEFAULT, None), (GemmTiles::DEFAULT, Some(2)), (tiles(64, 128, 8), None),
                (tiles(128, 64, 8), None), (tiles(128, 128, 1), None)];
    let weights: Vec<_> = cfgs.iter().map(|(c, _)| interleaved(&packed, &scales, c.bn / 2, &s)).collect();
    let mut bad = Vec::new();
    for t in [300usize, 1024] {
        let (x, xs) = activations(&mut rng, t, &s);
        for igs in IGS {
            let (rp, rs, _) = reference(&x, &xs, &w, igs, t, &s);
            for ((cfg, occ), (wp, ws)) in cfgs.iter().zip(&weights) {
                let a = Nvfp4Act::zeros(t, I, &s).unwrap();
                let opts = CompileOptions { occupancy: *occ, ..Default::default() };
                let (q, sc) = nvfp4::gemm_swiglu_il_op_with(&x, &xs, wp, ws, alpha, igs, Arc::new(a.packed),
                                                            Arc::new(a.scales), *cfg, sms(), opts)
                    .unwrap()
                    .sync_on(&s)
                    .unwrap();
                let (gp, gs) = act_bytes(&q, &sc, &s);
                let (mp, ms) = (mismatches(&gp, &rp), mismatches(&gs, &rs));
                println!("gemm_swiglu T={t} igs={igs} tiles {}x{} g{} (block {}) occupancy {occ:?}: packed mismatches \
                          {mp}/{}, scales {ms}/{}", cfg.bm, cfg.bn, cfg.group_m, cfg.bn / 2, gp.len(), gs.len());
                if mp + ms > 0 {
                    bad.push((t, igs, cfg.bm, cfg.bn, cfg.group_m, *occ));
                }
            }
        }
    }
    assert!(bad.is_empty(), "fused gate/up + SiLU·mul + quantize differs from the N1 path: {bad:?}");
}

#[test]
fn interleave_gate_up_is_a_row_permutation() {
    // 2I = 8 rows of 3 bytes, block 2: [g0 g1 u0 u1 g2 g3 u2 u3].
    let rows: Vec<u8> = (0..8u8).flat_map(|r| [r, r, r]).collect();
    let out = nvfp4::interleave_gate_up(&rows, 8, 3, 2).unwrap();
    let order: Vec<u8> = out.chunks(3).map(|c| c[0]).collect();
    assert_eq!(order, [0, 1, 4, 5, 2, 3, 6, 7]);
    assert!(nvfp4::interleave_gate_up(&rows, 8, 3, 3).is_err());
}

/// The residual-add norms with the engine's occupancy hint against the compiler default (N1's launch).
#[test]
fn norm_occupancy_hints_are_bit_identical() {
    let s = stream();
    let mut rng = Rng(0x7777_0000_1111_3333);
    let t = 257usize;
    let (res, x) = (rng.rows(t, K), rng.rows(t, K));
    let w: Vec<bf16> = (0..K).map(|_| bf16::from_f64(rng.normal() * 0.5)).collect();
    let wd = upload(w, &[K], &s);
    let xd = upload(x, &[t, K], &s);
    let bits = |v: Vec<bf16>| v.iter().map(|b| b.to_bits()).collect::<Vec<u16>>();
    let run_bf16 = |occ: Option<i32>| {
        let mut r = upload(res.clone(), &[t, K], &s);
        let mut o = api::zeros::<bf16>(&[t, K]).sync_on(&s).unwrap();
        norm::add_zc_rmsnorm_with(&mut r, &mut o, &xd, &wd, norm::RMS_EPS, occ).unwrap().sync_on(&s).unwrap();
        (bits(r.to_host_vec().sync_on(&s).unwrap()), bits(o.to_host_vec().sync_on(&s).unwrap()))
    };
    let run_q = |occ: Option<i32>| {
        let mut r = upload(res.clone(), &[t, K], &s);
        let a = Nvfp4Act::zeros(t, K, &s).unwrap();
        let (mut q, mut sc) = (a.packed, a.scales);
        nvfp4::add_zc_rmsnorm_nvfp4_with(&mut r, &mut q, &mut sc, &xd, &wd, norm::RMS_EPS, 628.0, occ)
            .unwrap()
            .sync_on(&s)
            .unwrap();
        let (p, b) = act_bytes(&Arc::new(q), &Arc::new(sc), &s);
        (bits(r.to_host_vec().sync_on(&s).unwrap()), p, b)
    };
    let (r0, o0) = run_bf16(None);
    let (qr0, qp0, qs0) = run_q(None);
    let mut bad = Vec::new();
    for occ in [Some(2), Some(3)] {
        let (r1, o1) = run_bf16(occ);
        let (qr1, qp1, qs1) = run_q(occ);
        let m = [mismatches(&r1, &r0), mismatches(&o1, &o0), mismatches(&qr1, &qr0), mismatches(&qp1, &qp0),
                 mismatches(&qs1, &qs0)];
        println!("occupancy {occ:?}: mismatches: add_zc_rmsnorm residual {} out {}; add_zc_rmsnorm_nvfp4 residual {} packed {} scales \
                  {}", m[0], m[1], m[2], m[3], m[4]);
        if m.iter().sum::<usize>() > 0 {
            bad.push(occ);
        }
    }
    assert!(bad.is_empty(), "occupancy hints changed the norms: {bad:?}");
}

#[test]
fn silu_mul_nvfp4_tile_shapes_are_bit_identical() {
    let s = stream();
    let t = 256usize;
    let mut rng = Rng(0x5151_aaaa_cccc_0002);
    let gu = upload(rng.rows(t, 2 * I), &[t, 2 * I], &s);
    let mut bad = Vec::new();
    for igs in IGS {
        let a = Nvfp4Act::zeros(t, I, &s).unwrap();
        let (mut q0, mut s0) = (a.packed, a.scales);
        nvfp4::silu_mul_nvfp4(&mut q0, &mut s0, &gu, igs).unwrap().sync_on(&s).unwrap();
        let (rp, rs) = act_bytes(&Arc::new(q0), &Arc::new(s0), &s);
        for st in [
            SiluTiles { rows: 1, cols: 1024, occupancy: None },
            SiluTiles { rows: 1, cols: 256, occupancy: None },
            SiluTiles { rows: 1, cols: 512, occupancy: Some(4) },
            SiluTiles { rows: 2, cols: 512, occupancy: None },
            SiluTiles { rows: 4, cols: 256, occupancy: Some(2) },
            SiluTiles { rows: 8, cols: 128, occupancy: None },
            SiluTiles { rows: 1, cols: 1024, occupancy: Some(3) },
        ] {
            let a = Nvfp4Act::zeros(t, I, &s).unwrap();
            let (mut q, mut sc) = (a.packed, a.scales);
            nvfp4::silu_mul_nvfp4_with(&mut q, &mut sc, &gu, igs, st).unwrap().sync_on(&s).unwrap();
            let (gp, gs) = act_bytes(&Arc::new(q), &Arc::new(sc), &s);
            let (mp, ms) = (mismatches(&gp, &rp), mismatches(&gs, &rs));
            println!("silu_mul_nvfp4 {st:?} igs={igs}: packed mismatches {mp}/{}, scales {ms}/{}", gp.len(), gs.len());
            if mp + ms > 0 {
                bad.push((igs, st));
            }
        }
    }
    assert!(bad.is_empty(), "silu_mul_nvfp4 tile shapes differ: {bad:?}");
}

/// The raw GEMM's bf16 output across tile shapes with the same BK: every output element is the same
/// `mmaf_scaled` chain over K, so BM / BN / grouping must not change a bit.
#[test]
fn gemm_raw_tile_shapes_with_same_bk_are_bit_identical() {
    let s = stream();
    let mut rng = Rng(0x0a0b_0c0d_0e0f_1011);
    let (w, ..) = gate_up(&mut rng, &s);
    let t = 384usize;
    let (x, xs) = activations(&mut rng, t, &s);
    let bits = |v: Vec<bf16>| v.iter().map(|b| b.to_bits()).collect::<Vec<u16>>();
    let run = |cfg: GemmTiles| {
        let z = api::zeros::<bf16>(&[t, 2 * I]).sync_on(&s).unwrap();
        bits(nvfp4::gemm(&x, &xs, &w, z, cfg, sms(), &s).unwrap().to_host_vec().sync_on(&s).unwrap())
    };
    let r = run(GemmTiles::DEFAULT);
    let mut bad = Vec::new();
    for cfg in [tiles(128, 64, 8), tiles(64, 128, 8), tiles(128, 128, 1), tiles(128, 256, 8), tiles(256, 128, 8)] {
        let m = mismatches(&run(cfg), &r);
        println!("gemm_nvfp4_raw tiles {}x{}x128 g{}: mismatches {m}/{}", cfg.bm, cfg.bn, cfg.group_m, r.len());
        if m > 0 {
            bad.push((cfg.bm, cfg.bn, cfg.group_m));
        }
    }
    assert!(bad.is_empty(), "raw GEMM results depend on the tile shape: {bad:?}");
}
