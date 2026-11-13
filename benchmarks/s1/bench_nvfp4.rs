//! W3c microbenchmark: NVFP4 activation quantization and block-scaled FP4 GEMMs at elephant-1's MLP shapes
//! (DESIGN.md §3.1, §6.6, §13 Q8), against cuBLAS bf16 at the same shapes.
//!
//! Reports TFLOP/s per shape and tile configuration, then the implied per-forward GEMM speedup: the MLP on NVFP4
//! (quantize + fused gate/up + quantize + down) with every other projection on cuBLAS bf16, vs all-bf16.
//!
//! Inputs are random (non-trivial bit patterns, so clocks are not flattered). Run under the shared GPU lock:
//!
//!     flock /tmp/sysone-gpu.lock cargo run --release --bin bench_nvfp4 -- --tokens 1024,4096,16384

use anyhow::Result;
use clap::Parser;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t as Op};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, sys::CUevent_flags};
use cutile::cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::cuda_async::device_operation::ExecutionContext;
use std::time::Instant;
use sysone::kernels::nvfp4::{self, CheckpointProj, GemmKernel, GemmTiles, Nvfp4Act, Nvfp4Linear, QuantTiles};

#[derive(Parser)]
struct Args {
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    iters: usize,
    /// Sweep tile configurations instead of only the default.
    #[arg(long)]
    sweep: bool,
    /// Explicit GEMM configs, e.g. `128x128x128g8p0,128x256x128g1p1m` (BMxBNxBK, group_m, blocks per SM; `m` = Mapped kernel).
    #[arg(long, value_delimiter = ',')]
    configs: Vec<String>,
    /// Skip the quantize kernel.
    #[arg(long)]
    no_quant: bool,
    /// Only these MLP GEMMs (`gate_up`, `down`).
    #[arg(long, value_delimiter = ',', default_value = "gate_up,down")]
    only: Vec<String>,
}

fn parse_cfg(s: &str) -> GemmTiles {
    let (dims, rest) = s.split_once('g').expect("BMxBNxBKg<G>p<P>");
    let (g, p) = rest.split_once('p').expect("g<G>p<P>");
    let (p, kernel) = match p.strip_suffix('m') { Some(p) => (p, GemmKernel::Mapped), None => (p, GemmKernel::Raw) };
    let d: Vec<usize> = dims.split('x').map(|v| v.parse().unwrap()).collect();
    GemmTiles { bm: d[0], bn: d[1], bk: d[2], group_m: g.parse().unwrap(), blocks_per_sm: p.parse().unwrap(), kernel }
}

/// GEMMs per forward (DESIGN.md §6.3), as in hwprobe: (name, n, k, count).
const PROJS: &[(&str, usize, usize, usize)] = &[
    ("gdn.in_proj_qkv", 8192, 4096, 24),
    ("gdn.in_proj_z", 4096, 4096, 24),
    ("gdn.in_proj_ab", 64, 4096, 24),
    ("gdn.out_proj", 4096, 4096, 24),
    ("attn.q_proj", 8192, 4096, 8),
    ("attn.kv_proj", 2048, 4096, 8),
    ("attn.o_proj", 4096, 4096, 8),
    ("mlp.gate_up", 24576, 4096, 32),
    ("mlp.down", 4096, 12288, 32),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn tflops(t: usize, n: usize, k: usize, ms: f64) -> f64 {
    2.0 * (t * n * k) as f64 / (ms * 1e-3) / 1e12
}

// ---- cuBLAS bf16 baseline (same call as hwprobe) ----

fn cublas_ms(stream: &Arc<CudaStream>, blas: &CudaBlas, t: usize, n: usize, k: usize, iters: usize) -> Result<f64> {
    let mut rng = Rng(0x1234_5678_9abc_def1 ^ (n * k) as u64);
    let host: Vec<half::bf16> =
        (0..(1 << 20)).map(|_| half::bf16::from_f32(((rng.next() >> 40) as f32 / 16777216.0 - 0.5) * 0.1)).collect();
    let fill = |len: usize| -> Result<CudaSlice<half::bf16>> {
        let src = stream.clone_htod(&host)?;
        let mut out = stream.alloc_zeros::<half::bf16>(len)?;
        let mut off = 0;
        while off < len {
            let m = host.len().min(len - off);
            stream.memcpy_dtod(&src.slice(..m), &mut out.slice_mut(off..off + m))?;
            off += m;
        }
        Ok(out)
    };
    let (w, x) = (fill(n * k)?, fill(t * k)?);
    let mut y = stream.alloc_zeros::<half::bf16>(t * n)?;
    let cfg = GemmConfig {
        transa: Op::CUBLAS_OP_T,
        transb: Op::CUBLAS_OP_N,
        m: n as i32,
        n: t as i32,
        k: k as i32,
        alpha: half::bf16::ONE,
        lda: k as i32,
        ldb: k as i32,
        beta: half::bf16::ZERO,
        ldc: n as i32,
    };
    unsafe { blas.gemm(cfg, &w, &x, &mut y)? };
    let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..iters {
        unsafe { blas.gemm(cfg, &w, &x, &mut y)? };
    }
    let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    end.synchronize()?;
    Ok(start.elapsed_ms(&end)? as f64 / iters as f64)
}

// ---- NVFP4 ----

fn random_linear(n_halves: &[usize], k: usize, stream: &Arc<cutile::cuda_core::Stream>, rng: &mut Rng) -> Result<Nvfp4Linear> {
    let data: Vec<(Vec<u8>, Vec<u8>)> = n_halves
        .iter()
        .map(|&n| {
            let p = (0..n * k / 2).map(|_| rng.next() as u8).collect();
            let s = (0..n * k / 16).map(|_| ((4 + rng.next() % 7) << 3 | (rng.next() & 7)) as u8).collect();
            (p, s)
        })
        .collect();
    let projs: Vec<CheckpointProj> = n_halves
        .iter()
        .zip(&data)
        .enumerate()
        .map(|(i, (&n, (p, s)))| CheckpointProj {
            n,
            k,
            weight_packed: p,
            weight_scale: s,
            weight_global_scale: 9408.0 / (i + 1) as f32,
            input_global_scale: 628.0,
        })
        .collect();
    Nvfp4Linear::upload(&nvfp4::prepare_weights(&projs)?, stream)
}

/// Mean ms per launch of `iters` back-to-back asynchronous launches (one warm-up batch first, JIT included), best of
/// `reps` batches. Batching amortizes host synchronization; GPU time-slicing with other processes still shows.
fn time_batch<S>(stream: &Arc<cutile::cuda_core::Stream>, iters: usize, reps: usize, state: &mut Option<S>,
                 mut launch: impl FnMut(S, &ExecutionContext) -> Result<S>) -> Result<f64> {
    let mut best = f64::MAX;
    for rep in 0..=reps {
        let ctx = ExecutionContext::new(stream.clone());
        let n = if rep == 0 { 1 } else { iters };
        let t0 = Instant::now();
        for _ in 0..n {
            let st = state.take().unwrap();
            *state = Some(launch(st, &ctx)?);
        }
        stream.synchronize()?;
        if rep > 0 {
            best = best.min(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
        }
    }
    Ok(best)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // cuBLAS bf16 for every projection at every T (cudarc context), then free it.
    // The MLP shapes are re-measured right next to their NVFP4 runs (the GPU is shared; see the module docs).
    let mut bf16_ms = std::collections::BTreeMap::new();
    let cctx = CudaContext::new(0)?;
    let cstream = cctx.default_stream();
    let blas = CudaBlas::new(cstream.clone())?;
    for &t in &args.tokens {
        for &(name, n, k, _) in PROJS {
            if !name.starts_with("mlp.") {
                bf16_ms.insert((t, name), cublas_ms(&cstream, &blas, t, n, k, args.iters)?);
            }
        }
    }

    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let sms = device.multiprocessor_count()? as usize;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let gu = random_linear(&[12288, 12288], 4096, &stream, &mut rng)?;
    let dn = random_linear(&[4096], 12288, &stream, &mut rng)?;
    println!("device SMs: {sms}");

    let gemm_cfgs: Vec<GemmTiles> = if !args.configs.is_empty() {
        args.configs.iter().map(|c| parse_cfg(c)).collect()
    } else if args.sweep {
        let mut v = Vec::new();
        for (bm, bn, bk) in [(128, 128, 128), (128, 256, 128), (256, 128, 128), (128, 128, 256), (64, 256, 128)] {
            for group_m in [1, 8, 16] {
                for blocks_per_sm in [0, 1, 2] {
                    v.push(GemmTiles { bm, bn, bk, group_m, blocks_per_sm, kernel: GemmKernel::Raw });
                }
            }
        }
        v
    } else {
        vec![GemmTiles::DEFAULT]
    };
    let quant_cfgs: Vec<QuantTiles> = if args.no_quant {
        vec![]
    } else if args.sweep {
        vec![QuantTiles::DEFAULT, QuantTiles { br: 1, bc: 1024 }, QuantTiles { br: 4, bc: 512 }, QuantTiles { br: 8, bc: 256 }, QuantTiles { br: 32, bc: 128 }, QuantTiles { br: 64, bc: 64 }, QuantTiles { br: 16, bc: 512 }]
    } else {
        vec![QuantTiles::DEFAULT]
    };

    println!("\n{:<10} {:>6} {:>6} {:>6} {:<24} {:>9} {:>9} {:>10} {:>8}",
             "op", "T", "N", "K", "config", "ms", "TFLOP/s", "cuBLAS bf16", "x bf16");
    let mut summary = Vec::new();
    for &t in &args.tokens {
        let mut best_ms = std::collections::BTreeMap::new();
        for (label, lin) in [("gate_up", &gu), ("down", &dn)] {
            if !args.only.iter().any(|o| o == label) {
                continue;
            }
            let (n, k) = (lin.n, lin.k);
            let bf = cublas_ms(&cstream, &blas, t, n, k, args.iters)?;
            // Quantize kernel (bf16 [T, K] -> NVFP4).
            let xh: Vec<bf16> = (0..t * k).map(|_| bf16::from_f32((rng.next() >> 40) as f32 / 16777216.0 - 0.5)).collect();
            let x: Arc<Tensor<bf16>> = api::copy_host_vec_to_device(&Arc::new(xh)).reshape(&[t, k]).sync_on(&stream)?.into();
            let mut act = Some(Nvfp4Act::zeros(t, k, &stream)?);
            let mut qbest = (if args.no_quant { 0.0 } else { f64::MAX }, QuantTiles::DEFAULT);
            for &qc in &quant_cfgs {
                let igs = lin.input_global_scale;
                let ms = time_batch(&stream, args.iters, 3, &mut act, |a, ctx| {
                    Ok(unsafe { nvfp4::quantize_op(&x, igs, a, qc)?.execute(ctx)? })
                })?;
                let gbs = (t * k * 2 + t * k / 2 + t * k / 16) as f64 / (ms * 1e-3) / 1e9;
                println!("{:<10} {t:>6} {:>6} {k:>6} {:<24} {ms:>9.3} {:>9} {:>10} {:>8}   {gbs:.0} GB/s",
                         "quantize", "", format!("{}x{}", qc.br, qc.bc), "", "", "");
                if ms < qbest.0 {
                    qbest = (ms, qc);
                }
            }
            let act = act.unwrap();
            let (xp, xs): (Arc<Tensor<f4e2m1fnx2>>, Arc<Tensor<f8e4m3fn>>) = (act.packed.into(), act.scales.into());
            let z: Arc<Tensor<bf16>> = api::zeros::<bf16>(&[t, n]).sync_on(&stream)?.into();
            let mut zs = Some(api::zeros::<bf16>(&[t, n]).sync_on(&stream)?);
            let mut gbest = (f64::MAX, GemmTiles::DEFAULT);
            for &gc in &gemm_cfgs {
                if n % gc.bn != 0 || k % gc.bk != 0 {
                    continue;
                }
                let r = if gc.kernel == GemmKernel::Raw {
                    let mut st = Some(());
                    time_batch(&stream, args.iters, 3, &mut st, |(), ctx| {
                        unsafe { nvfp4::gemm_op(&xp, &xs, lin, z.clone(), gc, sms)?.execute(ctx)? };
                        Ok(())
                    })
                } else {
                    // The mapped kernel consumes its output partition; time it synchronously.
                    time_batch(&stream, args.iters, 3, &mut zs, |zz, _| nvfp4::gemm(&xp, &xs, lin, zz, gc, sms, &stream))
                };
                let cfg = format!("{}x{}x{} g{} p{}{}", gc.bm, gc.bn, gc.bk, gc.group_m, gc.blocks_per_sm,
                                  if gc.kernel == GemmKernel::Mapped { " mapped" } else { "" });
                match r {
                    Ok(ms) => {
                        println!("{label:<10} {t:>6} {n:>6} {k:>6} {cfg:<24} {ms:>9.3} {:>9.1} {:>10.1} {:>8.2}",
                                 tflops(t, n, k, ms), tflops(t, n, k, bf), bf / ms);
                        if ms < gbest.0 {
                            gbest = (ms, gc);
                        }
                    }
                    Err(e) => {
                        println!("{label:<10} {t:>6} {n:>6} {k:>6} {cfg:<24} failed: {e}");
                        zs = Some(api::zeros::<bf16>(&[t, n]).sync_on(&stream)?);
                    }
                }
            }
            best_ms.insert(label, (qbest.0, gbest.0, gbest.1, bf));
            bf16_ms.insert((t, if label == "gate_up" { "mlp.gate_up" } else { "mlp.down" }), bf);
        }
        if best_ms.len() < 2 {
            continue;
        }
        let (qg, gg, gcfg, bg) = best_ms["gate_up"];
        let (qd, gd, dcfg, bd) = best_ms["down"];
        let other_bf16: f64 = PROJS
            .iter()
            .filter(|p| !p.0.starts_with("mlp."))
            .map(|&(name, _, _, count)| bf16_ms[&(t, name)] * count as f64)
            .sum();
        let mlp_bf16 = 32.0 * (bg + bd);
        let mlp_fp4_gemm = 32.0 * (gg + gd);
        let mlp_fp4_all = 32.0 * (qg + gg + qd + gd);
        summary.push(format!(
            "T={t:>6}: MLP bf16 {mlp_bf16:8.1} ms | NVFP4 GEMMs {mlp_fp4_gemm:8.1} ms ({:.2}x) | +quantize {mlp_fp4_all:8.1} ms ({:.2}x) \
             | other GEMMs (bf16) {other_bf16:7.1} ms | forward GEMM time {:.1} -> {:.1} ms = {:.2}x (unfused quantize) / {:.2}x (fused) \
             | best gate_up {}x{}x{} g{} p{}, down {}x{}x{} g{} p{}",
            mlp_bf16 / mlp_fp4_gemm,
            mlp_bf16 / mlp_fp4_all,
            other_bf16 + mlp_bf16,
            other_bf16 + mlp_fp4_all,
            (other_bf16 + mlp_bf16) / (other_bf16 + mlp_fp4_all),
            (other_bf16 + mlp_bf16) / (other_bf16 + mlp_fp4_gemm),
            gcfg.bm, gcfg.bn, gcfg.bk, gcfg.group_m, gcfg.blocks_per_sm,
            dcfg.bm, dcfg.bn, dcfg.bk, dcfg.group_m, dcfg.blocks_per_sm,
        ));
    }
    println!("\nper-forward (32 layers, GEMM-only):");
    for s in summary {
        println!("{s}");
    }
    Ok(())
}
