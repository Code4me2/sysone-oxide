//! S2 microbenchmark (docs/S2.md): the two launches S2 changes, each against the launch it replaces.
//!
//! - `--gated`: the GDN `gated_rmsnorm` (`[T, 4096]` core output and the strided z columns of the `[T, 12352]`
//!   projection, as in the engine) for a sweep of rows per tile block x occupancy hint x warps per CTA
//!   (`--gated-cfgs R[oO][wW],...`); `16` is the pre-S2 launch.
//! - `--down`: the MLP down projection and the next layer's input norm: pre-S2 `gemm_nvfp4_raw` (bf16 `mo`) +
//!   `add_zc_rmsnorm` (occupancy 2) against S2's `gemm_nvfp4_raw_residual` (residual += mo in the epilogue) +
//!   `zc_rmsnorm_res` (with occupancy hints), each pair and each launch alone.
//!
//! Every variant is byte-identical (tests/kernels_s2.rs); this only times them. Random data at elephant-1's shapes.
//!
//!     scripts/gpu.sh ./target/release/bench_s2 --tokens 1024,4096,16384,32768 --gated --down
//!
//! Reports the best of `--reps` batches of `--iters` back-to-back launches (ms per launch).

use anyhow::Result;
use clap::Parser;
use cutile::cuda_async::device_operation::ExecutionContext;
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tile_kernel::CompileOptions;
use std::time::Instant;
use sysone::contract::DecisionBatch;
use sysone::kernels::gdn::{self, DView, GatedNormTiles, GdnPlan};
use sysone::kernels::norm;
use sysone::kernels::nvfp4::{self, CheckpointProj, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles, RawAlpha};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    allow_shared_gpu: bool,
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384,32768")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 20)]
    iters: usize,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    #[arg(long)]
    gated: bool,
    /// gated_rmsnorm launch shapes, `R[hH][oO][wW]`: R rows x H heads per tile block, occupancy hint O, W warps per CTA.
    #[arg(long, value_delimiter = ',',
          default_value = "16,16o4,16o8,8o8,16w8,16h2,16h4,16h8,8h2,8h4,8h8,8h16,8h32,4h4,4h8,4h16,4h32,2h16,2h32,1h32,16h2o2,16h4o2,8h4o2,8h8o2,4h8o2,4h16o2,8h4o4,4h8o4,2h32o2,16h4w8,8h8w8")]
    gated_cfgs: Vec<String>,
    #[arg(long)]
    down: bool,
    /// zc_rmsnorm_res occupancy hints to time after the residual GEMM (0 = compiler default).
    #[arg(long, value_delimiter = ',', default_value = "0,2,3")]
    zc_occ: Vec<i32>,
    /// Occupancy hints to also time on the residual down GEMM (reported; the default launch has none).
    #[arg(long, value_delimiter = ',')]
    down_occ: Vec<i32>,
}

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
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16777216.0 - 0.5
    }
}

fn upload<T: DType>(v: Vec<T>, shape: &[usize], s: &Arc<cutile::cuda_core::Stream>) -> Result<Tensor<T>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(s)?)
}

fn time_batch(stream: &Arc<cutile::cuda_core::Stream>, iters: usize, reps: usize,
              mut launch: impl FnMut(&ExecutionContext) -> Result<()>) -> Result<f64> {
    let mut best = f64::MAX;
    for rep in 0..=reps {
        let ctx = ExecutionContext::new(stream.clone());
        let n = if rep == 0 { 1 } else { iters };
        let t0 = Instant::now();
        for _ in 0..n {
            launch(&ctx)?;
        }
        stream.synchronize()?;
        if rep > 0 {
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
        }
    }
    Ok(best)
}

fn parse_gated(s: &str) -> GatedNormTiles {
    let mut t = GatedNormTiles { rows: 0, heads: 1, occupancy: None, warps: None };
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    t.rows = s[..end].parse().expect("R[oO][wW]");
    let mut rest = &s[end..];
    while !rest.is_empty() {
        let (tag, tail) = rest.split_at(1);
        let e = tail.find(|c: char| !c.is_ascii_digit()).unwrap_or(tail.len());
        let v: i32 = tail[..e].parse().unwrap();
        match tag {
            "o" => t.occupancy = Some(v),
            "w" => t.warps = Some(v),
            "h" => t.heads = v as usize,
            _ => panic!("bad gated config {s}"),
        }
        rest = &tail[e..];
    }
    t
}

fn down_weight(stream: &Arc<cutile::cuda_core::Stream>, rng: &mut Rng) -> Result<Nvfp4Linear> {
    let p: Vec<u8> = (0..H * I / 2).map(|_| rng.next() as u8).collect();
    let sc: Vec<u8> = (0..H * I / 16).map(|_| ((4 + rng.next() % 7) << 3 | (rng.next() & 7)) as u8).collect();
    let proj = CheckpointProj { n: H, k: I, weight_packed: &p, weight_scale: &sc, weight_global_scale: 6144.0,
                                input_global_scale: 233.0 };
    Nvfp4Linear::upload(&nvfp4::prepare_weights(&[proj])?, stream)
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("# {}", sysone::engine::platform::require_idle_gpu(args.allow_shared_gpu)?);
    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let sms = device.multiprocessor_count()? as usize;
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let gb = |bytes: usize, ms: f64| bytes as f64 / (ms * 1e-3) / 1e9;
    let wd = if args.down { Some(down_weight(&stream, &mut rng)?) } else { None };
    for &t in &args.tokens {
        println!("\nT={t}");
        if args.gated {
            let mut b = DecisionBatch::default();
            b.push_request(&vec![1u32; t - 1], &[(&[2u32][..], 2)])?;
            let plan = GdnPlan::new(&b.segments, t, &stream)?;
            let proj = upload((0..t * PROJ).map(|_| bf16::from_f32(rng.f() * 4.0)).collect(), &[t, PROJ], &stream)?;
            let core = upload((0..t * gdn::VALUE_DIM).map(|_| bf16::from_f32(rng.f())).collect(), &[t, gdn::VALUE_DIM],
                              &stream)?;
            let w = upload((0..gdn::HEAD_DIM).map(|_| bf16::from_f32(0.9 + rng.f() * 0.2)).collect(), &[gdn::HEAD_DIM],
                           &stream)?;
            let out = api::zeros::<bf16>(&[t, gdn::VALUE_DIM]).sync_on(&stream)?;
            let z = DView::of(&proj)?.columns(gdn::CONV_DIM, gdn::VALUE_DIM)?;
            let (x, o) = (DView::of(&core)?, DView::of(&out)?);
            let bytes = 3 * t * gdn::VALUE_DIM * 2;
            let mut base = 0.0;
            for spec in &args.gated_cfgs {
                let cfg = parse_gated(spec);
                let r = time_batch(&stream, args.iters, args.reps, |ctx| {
                    // SAFETY: every buffer outlives the synchronous timing loop.
                    unsafe {
                        gdn::gated_rmsnorm_with(&plan, x.clone(), z.clone(), &w, 1e-6, o.clone(), cfg)?.execute(ctx)?
                    };
                    Ok(())
                });
                match r {
                    Ok(ms) => {
                        if cfg == GatedNormTiles::F1 {
                            base = ms;
                        }
                        let rel = if base > 0.0 { format!("{:.2}x", base / ms) } else { String::new() };
                        println!("  gated_rmsnorm {spec:<6} {ms:>7.3} ms {:>6.1} GB/s  x24 {:>6.1} ms {rel}{}",
                                 gb(bytes, ms), 24.0 * ms, if cfg == GatedNormTiles::F1 { "  (pre-S2)" } else { "" });
                    }
                    Err(e) => println!("  gated_rmsnorm {spec:<6} failed: {e:#}"),
                }
            }
        }
        if let Some(w) = &wd {
            let xh: Vec<bf16> = (0..t * I).map(|_| bf16::from_f32(rng.f() * 4.0)).collect();
            let xb: Arc<Tensor<bf16>> = upload(xh, &[t, I], &stream)?.into();
            let a = nvfp4::quantize(&xb, 233.0, Nvfp4Act::zeros(t, I, &stream)?, QuantTiles::DEFAULT, &stream)?;
            let (xp, xs): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (a.packed.into(), a.scales.into());
            let alpha = RawAlpha { lo: w.alpha_lo, hi: w.alpha_hi, split_n: w.split_n };
            let mo: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[t, H]).sync_on(&stream)?.into();
            let res: Arc<Tensor<bf16>> =
                upload((0..t * H).map(|_| bf16::from_f32(rng.f())).collect(), &[t, H], &stream)?.into();
            let mut hn = api::zeros::<bf16>(&[t, H]).sync_on(&stream)?;
            let nw = api::zeros::<bf16>(&[H]).sync_on(&stream)?;
            let tl = GemmTiles::DEFAULT;
            let tflop = |ms: f64| 2.0 * (t * H * I) as f64 / (ms * 1e-3) / 1e12;
            let g = time_batch(&stream, args.iters, args.reps, |ctx| {
                unsafe { nvfp4::gemm_raw_op(&xp, &xs, &w.packed, &w.scales, alpha, mo.clone(), tl, sms)?.execute(ctx)? };
                Ok(())
            })?;
            let gres = |occ: Option<i32>| {
                time_batch(&stream, args.iters, args.reps, |ctx| {
                    unsafe {
                        nvfp4::gemm_raw_residual_op_with(&xp, &xs, &w.packed, &w.scales, alpha, res.clone(), tl, sms,
                                                         CompileOptions { occupancy: occ, ..Default::default() })?
                            .execute(ctx)?
                    };
                    Ok(())
                })
            };
            let gr = gres(None)?;
            for occ in &args.down_occ {
                match gres(Some(*occ)) {
                    Ok(ms) => println!("  gemm_nvfp4_raw_residual occupancy {occ}: {ms:.3} ms ({:.1} TFLOP/s)", tflop(ms)),
                    Err(e) => println!("  gemm_nvfp4_raw_residual occupancy {occ}: failed: {e:#}"),
                }
            }
            // The norms run on separate copies of the residual (in place; values do not matter for timing).
            let mut r2 = upload((0..t * H).map(|_| bf16::from_f32(rng.f())).collect(), &[t, H], &stream)?;
            let n_add = time_batch(&stream, args.iters, args.reps, |ctx| {
                unsafe { norm::add_zc_rmsnorm_with(&mut r2, &mut hn, &mo, &nw, 1e-6, Some(2))?.execute(ctx)? };
                Ok(())
            })?;
            println!("  down gemm_nvfp4_raw {g:.3} ms ({:.1} TFLOP/s) | gemm_nvfp4_raw_residual {gr:.3} ms ({:.1} TFLOP/s) \
                      | add_zc_rmsnorm o2 {n_add:.3} ms ({:.0} GB/s)", tflop(g), tflop(gr), gb(t * H * 8, n_add));
            for &occ in &args.zc_occ {
                let o = if occ == 0 { None } else { Some(occ) };
                let n_zc = time_batch(&stream, args.iters, args.reps, |ctx| {
                    unsafe { norm::zc_rmsnorm_res_with(&mut r2, &mut hn, &nw, 1e-6, o)?.execute(ctx)? };
                    Ok(())
                })?;
                println!("    zc_rmsnorm_res occ {o:?} {n_zc:.3} ms ({:.0} GB/s): pre-S2 pair {:.3} ms, S2 pair {:.3} ms, \
                          saves {:.3} ms (x32 {:.1} ms)", gb(t * H * 4, n_zc), g + n_add, gr + n_zc,
                         g + n_add - gr - n_zc, 32.0 * (g + n_add - gr - n_zc));
            }
        }
    }
    Ok(())
}
