//! F1 microbenchmark of the NVFP4 MLP middle: the gate/up GEMM followed by SiLU·mul + quantize, as N1 runs it
//! (`gemm_nvfp4_raw` writing bf16 `[T, 24576]`, then `silu_mul_nvfp4`) against F1's variants: `silu_mul_nvfp4`
//! with other tile shapes / occupancy hints, the fused `gemm_nvfp4_swiglu_il` (SiLU·mul + quantize in the GEMM
//! epilogue, interleaved gate/up rows) at several tile shapes, compile hints on the raw GEMM (`--gemm-opts`) and the
//! residual-add norms with occupancy hints (`--norms`). Every variant is byte-identical (tests/kernels_nvfp4_f1.rs);
//! this only times them. Random weights and activations at elephant-1's shapes (K = 4096, I = 12288).
//!
//!     scripts/gpu.sh ./target/release/bench_nvfp4_mlp --tokens 1024,4096,16384,32768
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
use sysone::kernels::nvfp4::{
    self, CheckpointProj, GemmKernel, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles, RawAlpha, SiluTiles,
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    allow_shared_gpu: bool,
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    iters: usize,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    /// Fused gate/up + SiLU·mul GEMM tiles to time, `BMxBNg<G>[/<opts>]` (BN = gate + up rows per weight tile;
    /// opts as `--gemm-opts`).
    #[arg(long, value_delimiter = ',', default_value = "128x128g8,64x128g8,128x64g8")]
    swiglu: Vec<String>,
    /// SiLU·mul tile shapes, `RxB` or `RxBoO` (occupancy hint O).
    #[arg(long, value_delimiter = ',', default_value = "1x1024,1x512,1x256,2x512,4x256,8x128,1x1024o2,1x512o4,2x256o4,4x128o8")]
    silu: Vec<String>,
    /// Also time the residual-add norms (bf16 and NVFP4-quantizing) with occupancy hints.
    #[arg(long)]
    norms: bool,
    /// Raw gate/up GEMM compile-option variants, `o<occupancy>w<warps>` (either part optional), e.g. `w8,w16,o2`.
    #[arg(long, value_delimiter = ',')]
    gemm_opts: Vec<String>,
}

fn parse_opts(s: &str) -> CompileOptions {
    let mut o = CompileOptions::default();
    let mut rest = s;
    while !rest.is_empty() {
        let (tag, tail) = rest.split_at(1);
        let end = tail.find(|c: char| !c.is_ascii_digit()).unwrap_or(tail.len());
        let v: i32 = tail[..end].parse().unwrap();
        match tag {
            "o" => o.occupancy = Some(v),
            "w" => o.num_worker_warps_per_cta = Some(v),
            _ => panic!("bad option {s}"),
        }
        rest = &tail[end..];
    }
    o
}

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
}

fn gate_up(stream: &Arc<cutile::cuda_core::Stream>, rng: &mut Rng) -> Result<Nvfp4Linear> {
    let data: Vec<(Vec<u8>, Vec<u8>)> = (0..2)
        .map(|_| {
            let p = (0..I * K / 2).map(|_| rng.next() as u8).collect();
            let s = (0..I * K / 16).map(|_| ((4 + rng.next() % 7) << 3 | (rng.next() & 7)) as u8).collect();
            (p, s)
        })
        .collect();
    let projs: Vec<CheckpointProj> = data
        .iter()
        .enumerate()
        .map(|(i, (p, s))| CheckpointProj {
            n: I,
            k: K,
            weight_packed: p,
            weight_scale: s,
            weight_global_scale: 9408.0 / (i + 1) as f32,
            input_global_scale: 628.0,
        })
        .collect();
    Nvfp4Linear::upload(&nvfp4::prepare_weights(&projs)?, stream)
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

fn parse_swiglu(s: &str) -> GemmTiles {
    let (dims, g) = s.split_once('g').expect("BMxBNg<G>");
    let d: Vec<usize> = dims.split('x').map(|v| v.parse().unwrap()).collect();
    GemmTiles { bm: d[0], bn: d[1], bk: 128, group_m: g.parse().unwrap(), blocks_per_sm: 0, kernel: GemmKernel::Raw }
}

fn parse_silu(s: &str) -> SiluTiles {
    let (dims, occ) = match s.split_once('o') {
        Some((d, o)) => (d, Some(o.parse().unwrap())),
        None => (s, None),
    };
    let (r, b) = dims.split_once('x').expect("RxB");
    SiluTiles { rows: r.parse().unwrap(), cols: b.parse().unwrap(), occupancy: occ }
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("# {}", sysone::engine::platform::require_idle_gpu(args.allow_shared_gpu)?);
    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let sms = device.multiprocessor_count()? as usize;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let w = gate_up(&stream, &mut rng)?;
    let alpha = RawAlpha { lo: w.alpha_lo, hi: w.alpha_hi, split_n: w.split_n };
    let igs = 233.0f32;
    let gemm_tflop = |t: usize, ms: f64| 2.0 * (t * 2 * I * K) as f64 / (ms * 1e-3) / 1e12;
    for &t in &args.tokens {
        let xh: Vec<bf16> =
            (0..t * K).map(|_| bf16::from_f32(((rng.next() >> 40) as f32 / 16777216.0 - 0.5) * 4.0)).collect();
        let x: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(xh)).reshape(&[t, K]).sync_on(&stream)?.into();
        let a = nvfp4::quantize(&x, 628.0, Nvfp4Act::zeros(t, K, &stream)?, QuantTiles::DEFAULT, &stream)?;
        let (xp, xs): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (a.packed.into(), a.scales.into());
        let gu: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[t, 2 * I]).sync_on(&stream)?.into();
        let out = Nvfp4Act::zeros(t, I, &stream)?;
        let (mut q, mut s) = (out.packed, out.scales);

        let g = time_batch(&stream, args.iters, args.reps, |ctx| {
            unsafe { nvfp4::gemm_op(&xp, &xs, &w, gu.clone(), GemmTiles::DEFAULT, sms)?.execute(ctx)? };
            Ok(())
        })?;
        println!("\nT={t}: gate/up gemm_nvfp4_raw (N1) {g:.3} ms ({:.1} TFLOP/s)", gemm_tflop(t, g));
        for spec in &args.gemm_opts {
            let o = parse_opts(spec);
            let r = time_batch(&stream, args.iters, args.reps, |ctx| {
                let al = RawAlpha { lo: w.alpha_lo, hi: w.alpha_hi, split_n: w.split_n };
                unsafe {
                    nvfp4::gemm_raw_op_with(&xp, &xs, &w.packed, &w.scales, al, gu.clone(), GemmTiles::DEFAULT, sms,
                                            o.clone())?
                        .execute(ctx)?
                };
                Ok(())
            });
            match r {
                Ok(ms) => println!("  gate/up raw {spec:<8} {ms:>8.3} ms ({:.1} TFLOP/s)", gemm_tflop(t, ms)),
                Err(e) => println!("  gate/up raw {spec:<8} failed: {e:#}"),
            }
        }
        let mut n1_silu = 0.0;
        for spec in &args.silu {
            let st = parse_silu(spec);
            if t % st.rows != 0 {
                continue;
            }
            let ms = if st == SiluTiles::N1 && spec == "1x1024" {
                time_batch(&stream, args.iters, args.reps, |ctx| {
                    unsafe { nvfp4::silu_mul_nvfp4(&mut q, &mut s, &gu, igs)?.execute(ctx)? };
                    Ok(())
                })?
            } else {
                time_batch(&stream, args.iters, args.reps, |ctx| {
                    unsafe { nvfp4::silu_mul_nvfp4_with(&mut q, &mut s, &gu, igs, st)?.execute(ctx)? };
                    Ok(())
                })?
            };
            if spec == "1x1024" {
                n1_silu = ms;
            }
            let gbs = (t * 2 * I * 2 + t * I / 2 + t * I / 16) as f64 / (ms * 1e-3) / 1e9;
            println!("  silu_mul_nvfp4 {spec:<10} {ms:>8.3} ms  {gbs:>6.1} GB/s{}", if spec == "1x1024" { "  (N1)" } else { "" });
        }
        let (qa, sa): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (Arc::new(q), Arc::new(s));
        println!("  N1 gate/up + silu_mul_nvfp4: {:.3} ms", g + n1_silu);
        for spec in &args.swiglu {
            // Timing only: the weights are random, so their row order (interleaved or not) does not matter here.
            // An optional `/<opts>` suffix: compile hints as in --gemm-opts (e.g. `128x128g8/w8`).
            let (tspec, ospec) = spec.split_once('/').unwrap_or((spec, ""));
            let tl = parse_swiglu(tspec);
            let o = parse_opts(ospec);
            let r = time_batch(&stream, args.iters, args.reps, |ctx| {
                unsafe {
                    nvfp4::gemm_swiglu_il_op_with(&xp, &xs, &w.packed, &w.scales, alpha, igs, qa.clone(), sa.clone(), tl,
                                                  sms, o.clone())?
                        .execute(ctx)?
                };
                Ok(())
            });
            match r {
                Ok(ms) => println!("  gemm_nvfp4_swiglu {spec:<10} {ms:>8.3} ms ({:.1} TFLOP/s)  vs N1 pair {:.3} ms: {:.2}x",
                                   gemm_tflop(t, ms), g + n1_silu, (g + n1_silu) / ms),
                Err(e) => println!("  gemm_nvfp4_swiglu {spec:<10} failed: {e:#}"),
            }
        }
        // Keep the borrowed outputs alive until here.
        drop((qa, sa));
        if args.norms {
            // Residual add + norm (bf16 out) and residual add + norm + quantize, with occupancy hints.
            let mk = |rng: &mut Rng| -> Result<Tensor<bf16>> {
                let v: Vec<bf16> = (0..t * K).map(|_| bf16::from_f32((rng.next() >> 40) as f32 / 16777216.0 - 0.5)).collect();
                Ok(api::copy_host_vec_to_device(&Arc::new(v)).reshape(&[t, K]).sync_on(&stream)?)
            };
            let (mut res, xin) = (mk(&mut rng)?, mk(&mut rng)?);
            let mut out = api::zeros::<bf16>(&[t, K]).sync_on(&stream)?;
            let wn: Tensor<bf16> = api::zeros::<bf16>(&[K]).sync_on(&stream)?;
            let a = Nvfp4Act::zeros(t, K, &stream)?;
            let (mut nq, mut ns) = (a.packed, a.scales);
            for occ in [None, Some(2), Some(3), Some(4)] {
                let ms = time_batch(&stream, args.iters, args.reps, |ctx| {
                    unsafe {
                        sysone::kernels::norm::add_zc_rmsnorm_with(&mut res, &mut out, &xin, &wn, 1e-6, occ)?.execute(ctx)?
                    };
                    Ok(())
                })?;
                let msq = time_batch(&stream, args.iters, args.reps, |ctx| {
                    unsafe {
                        nvfp4::add_zc_rmsnorm_nvfp4_with(&mut res, &mut nq, &mut ns, &xin, &wn, 1e-6, 628.0, occ)?
                            .execute(ctx)?
                    };
                    Ok(())
                })?;
                let gb = |bytes: usize, ms: f64| bytes as f64 / (ms * 1e-3) / 1e9;
                println!("  add_zc_rmsnorm occ {occ:?}: {ms:.3} ms ({:.0} GB/s) | add_zc_rmsnorm_nvfp4 {msq:.3} ms ({:.0} GB/s)",
                         gb(t * K * 8, ms), gb(t * K * 6 + t * K / 2 + t * K / 16, msq));
            }
        }
    }
    Ok(())
}
