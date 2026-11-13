//! Phase 0 hardware probe (DESIGN.md §5, §11 W0): device memory bandwidth, bf16 GEMM throughput at the
//! model's projection shapes, and a GEMM-only estimate of one elephant-1 forward vs packed tokens T.
//!
//! cuBLAS bf16 with fp32 accumulation is the baseline every cutile GEMM must beat. Inputs are random so
//! clocks are not flattered by all-zero data.
//!
//! With `--profile-out PATH` it also writes the machine's platform profile (`sysone::engine::platform`), which
//! the engine and scheduler read to size token budgets and buckets (PLATFORM.md P1, P4).

use anyhow::Result;
use clap::Parser;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t as Op};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, sys::CUevent_flags};
use half::bf16;
use std::path::PathBuf;
use std::sync::Arc;
use sysone::engine::platform::{ForwardPoint, PlatformProfile};

#[derive(Parser)]
struct Args {
    /// Packed token counts to sweep.
    #[arg(long, value_delimiter = ',', default_value = "256,1024,4096,8192,16384")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 20)]
    iters: usize,
    /// Copy size for the bandwidth test, in MiB.
    #[arg(long, default_value_t = 1024)]
    bw_mib: usize,
    /// Write the platform profile (JSON) here.
    #[arg(long)]
    profile_out: Option<PathBuf>,
}

/// One projection of the model: `[T, k] x [n, k]^T`, appearing `count` times per forward.
struct Proj {
    name: &'static str,
    n: usize,
    k: usize,
    count: usize,
}

/// elephant-1 GEMMs per forward (DESIGN.md §6.3): 24 GatedDeltaNet layers, 8 full-attention layers, 32 MLPs.
const PROJS: &[Proj] = &[
    Proj { name: "gdn.in_proj_qkv", n: 8192, k: 4096, count: 24 },
    Proj { name: "gdn.in_proj_z", n: 4096, k: 4096, count: 24 },
    Proj { name: "gdn.in_proj_ab", n: 64, k: 4096, count: 24 },
    Proj { name: "gdn.out_proj", n: 4096, k: 4096, count: 24 },
    Proj { name: "attn.q_proj", n: 8192, k: 4096, count: 8 },
    Proj { name: "attn.kv_proj", n: 2048, k: 4096, count: 8 },
    Proj { name: "attn.o_proj", n: 4096, k: 4096, count: 8 },
    Proj { name: "mlp.gate_up", n: 24576, k: 4096, count: 32 },
    Proj { name: "mlp.down", n: 4096, k: 12288, count: 32 },
];

fn random_bf16(stream: &Arc<CudaStream>, len: usize, seed: u64) -> Result<CudaSlice<bf16>> {
    let block = 1 << 22;
    let mut s = seed | 1;
    let host: Vec<bf16> = (0..block.min(len))
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            bf16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1)
        })
        .collect();
    let src = stream.clone_htod(&host)?;
    let mut out = stream.alloc_zeros::<bf16>(len)?;
    let mut off = 0;
    while off < len {
        let n = block.min(len - off);
        stream.memcpy_dtod(&src.slice(..n), &mut out.slice_mut(off..off + n))?;
        off += n;
    }
    Ok(out)
}

fn time_ms(stream: &Arc<CudaStream>, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?; // warm-up (cuBLAS heuristics, clocks)
    let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    for _ in 0..iters {
        f()?;
    }
    let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    end.synchronize()?;
    Ok(start.elapsed_ms(&end)? as f64 / iters as f64)
}

/// Row-major `y[T, n] = x[T, k] · w[n, k]^T`, expressed for column-major cuBLAS as `y^T = w · x^T`.
fn gemm(blas: &CudaBlas, w: &CudaSlice<bf16>, x: &CudaSlice<bf16>, y: &mut CudaSlice<bf16>, t: usize, n: usize, k: usize)
    -> Result<()> {
    let cfg = GemmConfig {
        transa: Op::CUBLAS_OP_T,
        transb: Op::CUBLAS_OP_N,
        m: n as i32,
        n: t as i32,
        k: k as i32,
        alpha: bf16::ONE,
        lda: k as i32,
        ldb: k as i32,
        beta: bf16::ZERO,
        ldc: n as i32,
    };
    unsafe { blas.gemm(cfg, w, x, y)? };
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("device: {} (sm_{major}{minor})", ctx.name()?);
    use cudarc::driver::sys::CUdevice_attribute as Attr;
    let integrated = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0;
    let sm_count = ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
    let (_, total_mem) = cudarc::driver::result::mem_get_info()?;
    println!("integrated (unified memory): {integrated}; {sm_count} SMs; {:.1} GiB visible", total_mem as f64 / (1u64 << 30) as f64);

    // Memory bandwidth: device-to-device copy moves 2 bytes of traffic per byte copied.
    let bytes = args.bw_mib << 20;
    let a = random_bf16(&stream, bytes / 2, 1)?;
    let mut b = stream.alloc_zeros::<bf16>(bytes / 2)?;
    let ms = time_ms(&stream, args.iters, || Ok(stream.memcpy_dtod(&a, &mut b)?))?;
    let bw = 2.0 * bytes as f64 / (ms * 1e-3) / 1e9;
    println!("memory: d2d copy {} MiB: {ms:.3} ms -> {bw:.1} GB/s", args.bw_mib);
    drop((a, b));

    // Real-size, distinct weights per layer so the forward estimate streams all ~13.8 GB like the model will.
    let blas = CudaBlas::new(stream.clone())?;
    let weights: Vec<Vec<CudaSlice<bf16>>> = PROJS
        .iter()
        .enumerate()
        .map(|(i, p)| (0..p.count).map(|j| random_bf16(&stream, p.n * p.k, (i * 100 + j) as u64 + 7)).collect())
        .collect::<Result<_>>()?;
    let weight_gb: f64 = PROJS.iter().map(|p| (p.n * p.k * p.count * 2) as f64).sum::<f64>() / 1e9;
    println!("weights: {weight_gb:.2} GB in GEMMs; streaming floor per forward at measured BW: {:.1} ms",
             weight_gb / bw * 1e3);

    let t_max = *args.tokens.iter().max().unwrap();
    let x = random_bf16(&stream, t_max * 12288, 3)?;
    let mut y = stream.alloc_zeros::<bf16>(t_max * 24576)?;

    println!("\nper-projection (one instance), bf16 in/out, fp32 accumulate:");
    println!("{:>6} {:<16} {:>9} {:>9}", "T", "proj", "ms", "TFLOP/s");
    let mut forward_rows = Vec::new();
    let mut best_tflops = 0.0f64;
    for &t in &args.tokens {
        let mut fwd_ms = 0.0;
        let mut fwd_flop = 0.0;
        for (p, ws) in PROJS.iter().zip(&weights) {
            let ms = time_ms(&stream, args.iters, || gemm(&blas, &ws[0], &x, &mut y, t, p.n, p.k))?;
            let flop = 2.0 * (t * p.n * p.k) as f64;
            let tflops = flop / (ms * 1e-3) / 1e12;
            best_tflops = best_tflops.max(tflops);
            println!("{t:>6} {:<16} {ms:>9.3} {tflops:>9.1}", p.name);
            fwd_flop += flop * p.count as f64;
        }
        // Whole-model GEMM sweep with every layer's own weights, in model order.
        let ms = time_ms(&stream, (args.iters / 4).max(2), || {
            for layer in 0..32 {
                let gdn = layer % 4 != 3;
                let idx = if gdn { layer - layer / 4 } else { layer / 4 };
                let ids: &[usize] = if gdn { &[0, 1, 2, 3] } else { &[4, 5, 6] };
                for &i in ids.iter().chain(&[7, 8]) {
                    let p = &PROJS[i];
                    let w = &weights[i][if i >= 7 { layer } else { idx }];
                    gemm(&blas, w, &x, &mut y, t, p.n, p.k)?;
                }
            }
            Ok(())
        })?;
        fwd_ms += ms;
        forward_rows.push((t, fwd_ms, fwd_flop));
    }

    println!("\nGEMM-only forward estimate (all 32 layers, distinct weights):");
    println!("{:>6} {:>10} {:>9} {:>12}", "T", "ms", "TFLOP/s", "tokens/s");
    let mut forward = Vec::new();
    for (t, ms, flop) in forward_rows {
        let (tflops, tps) = (flop / (ms * 1e-3) / 1e12, t as f64 / (ms * 1e-3));
        println!("{t:>6} {ms:>10.2} {tflops:>9.1} {tps:>12.0}");
        forward.push(ForwardPoint { tokens: t, ms, tflops, tokens_per_s: tps });
    }
    if let Some(path) = args.profile_out {
        forward.sort_by_key(|p| p.tokens);
        let profile = PlatformProfile {
            version: PlatformProfile::VERSION,
            device_name: ctx.name()?,
            compute_capability: [major, minor],
            integrated,
            sm_count,
            total_mem_bytes: total_mem as u64,
            mem_bw_gbs: bw,
            gemm_bf16_tflops: best_tflops,
            gemm_weight_bytes: (weight_gb * 1e9) as u64,
            forward,
            note: format!("hwprobe, iters {}, bw copy {} MiB", args.iters, args.bw_mib),
        };
        profile.save(&path)?;
        println!("\nplatform profile written to {} (weight floor {:.1} ms, token budget at 90% of peak tok/s: {:?})",
                 path.display(), profile.weight_floor_ms(), profile.token_budget(0.9));
    }
    Ok(())
}
