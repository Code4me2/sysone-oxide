//! Microbenchmark of the GatedDeltaNet mixer kernels (`sysone::kernels::gdn`) on realistic segment trees.
//!
//! For each packed size T it builds requests of one prefix plus 8 suffixes of 100 tokens, times every kernel of
//! one GDN layer (conv, prep, WY, prefix scan, suffix scan, gated norm) and the whole mixer, and relates the
//! per-layer time to a forward: elephant-1 has 24 GDN layers, and Phase 0 measured the GEMM-only forward
//! (cuBLAS bf16) at 154.5 / 625.4 / 2320.1 ms for T = 1k / 4k / 16k (docs/PHASE0.md).
//!
//! Run under the shared GPU lock: `flock /tmp/sysone-gpu.lock cargo run --release --bin bench_gdn`.

use anyhow::Result;
use clap::Parser;
use cutile::cuda_core::Stream;
use cutile::half::bf16;
use cutile::prelude::*;
use std::time::Instant;
use sysone::contract::DecisionBatch;
use sysone::kernels::gdn::*;

#[cutile::module]
mod probes {
    use cutile::core::*;

    /// `iters` dependent [M, K] x [K, N] MMAs in element type E (accumulating in f32); writes the sum.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn mma_chain<E: ElementType, const M: i32, const N: i32, const K: i32>(
        out: &mut Tensor<f32, { [M, N] }>,
        a: &Tensor<f32, { [M, K] }>,
        b: &Tensor<f32, { [K, N] }>,
        iters: i32,
    ) {
        let at: Tile<f32, { [M, K] }> = a.partition(const_shape![M, K]).load([0i32, 0i32]);
        let bt: Tile<f32, { [K, N] }> = b.partition(const_shape![K, N]).load([0i32, 0i32]);
        let ae: Tile<E, { [M, K] }> = convert_tile(at);
        let be: Tile<E, { [K, N] }> = convert_tile(bt);
        let mut acc: Tile<f32, { [M, N] }> = constant(0.0f32, const_shape![M, N]);
        for _i in 0i32..iters {
            acc = mma(ae, be, acc);
        }
        out.store(acc);
    }

    /// `iters` rounds of the 64-step forward substitution of the WY kernel on one [64, 64] tile.
    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn fwd_subst(out: &mut Tensor<f32, { [64, 64] }>, a: &Tensor<f32, { [64, 64] }>, iters: i32) {
        let at: Tile<f32, { [64, 64] }> = a.partition(const_shape![64, 64]).load([0i32, 0i32]);
        let idx: Tile<i32, { [64] }> = iota(const_shape![64]);
        let rows: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![64, 1]).broadcast(const_shape![64, 64]);
        let cols: Tile<i32, { [64, 64] }> = idx.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
        let zero: Tile<f32, { [64, 64] }> = constant(0.0f32, const_shape![64, 64]);
        let one: Tile<f32, { [64, 64] }> = constant(1.0f32, const_shape![64, 64]);
        let mut acc: Tile<f32, { [64, 64] }> = zero;
        for _r in 0i32..iters {
            let mut tinv: Tile<f32, { [64, 64] }> = select(eq_tile(rows, cols), one, zero);
            for i in 1i32..64i32 {
                let sel: Tile<bool, { [64, 64] }> = eq_tile(rows, broadcast_scalar(i, const_shape![64, 64]));
                let a_row: Tile<f32, { [64] }> = reduce_sum(select(sel, at, zero), 0i32);
                let contrib: Tile<f32, { [64] }> =
                    reduce_sum(a_row.reshape(const_shape![64, 1]).broadcast(const_shape![64, 64]) * tinv, 0i32);
                let new_row: Tile<f32, { [64, 64] }> =
                    tinv - contrib.reshape(const_shape![1, 64]).broadcast(const_shape![64, 64]);
                tinv = select(sel, new_row, tinv);
            }
            acc = acc + tinv;
        }
        out.store(acc);
    }
}

fn probe(s: &Arc<Stream>) -> Result<()> {
    let programs = 384usize;
    for (ty, m, n, k) in [("f32", 64, 64, 128), ("tf32", 64, 64, 128), ("bf16", 64, 64, 128), ("f32", 64, 128, 64), ("f32", 128, 64, 64)] {
        let a = api::ones::<f32>(&[m, k]).sync_on(s)?;
        let b = api::ones::<f32>(&[k, n]).sync_on(s)?;
        let iters = 64;
        let gens = vec![ty.to_string(), m.to_string(), n.to_string(), k.to_string()];
        let run = |gens: Vec<String>| -> Result<f64> {
            let out = api::zeros::<f32>(&[m * programs, n]).sync_on(s)?.partition([m, n]);
            let t0 = Instant::now();
            let _ = unsafe { probes::mma_chain(out, &a, &b, iters as i32) }.generics(gens).sync_on(s)?;
            Ok(t0.elapsed().as_secs_f64())
        };
        run(gens.clone())?;
        let secs = run(gens)?;
        let flops = 2.0 * (m * n * k * iters * programs) as f64;
        println!("mma {ty:>5} [{m}x{k}]x[{k}x{n}]: {:.3} ms, {:.2} TFLOP/s ({programs} programs x {iters} dependent MMAs)", secs * 1e3, flops / secs / 1e12);
    }
    let a = api::ones::<f32>(&[64, 64]).sync_on(s)?;
    for iters in [1, 4] {
        let out = api::zeros::<f32>(&[64 * programs, 64]).sync_on(s)?.partition([64, 64]);
        let _ = unsafe { probes::fwd_subst(out, &a, iters) }.sync_on(s)?;
        let out = api::zeros::<f32>(&[64 * programs, 64]).sync_on(s)?.partition([64, 64]);
        let t0 = Instant::now();
        let _ = unsafe { probes::fwd_subst(out, &a, iters) }.sync_on(s)?;
        println!("forward substitution 64x64 x{iters}: {:.3} ms for {programs} programs", t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(())
}

#[derive(Parser)]
struct Args {
    /// Measure even if other processes use the GPU (the output is then marked not authoritative).
    #[arg(long)]
    allow_shared_gpu: bool,
    /// Run the primitive probes (mma rates, forward substitution) instead of the mixer benchmark.
    #[arg(long)]
    probe: bool,
    /// Comma-separated list of tree shapes: `T` for the default 2k-prefix trees, or `T:P` for prefixes of P tokens.
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384,16384:15584")]
    shapes: Vec<String>,
    #[arg(long, default_value_t = 10)]
    iters: usize,
    /// Scan precision: fp32 (3xTF32, default), tf32, bf16.
    #[arg(long, default_value = "fp32")]
    precision: String,
    /// Override the prefix / suffix scan value-column blocks, the warps per CTA and the forward-substitution
    /// block of the chunk inverse (`GdnOptions`). Each takes a comma-separated list; every combination is run.
    #[arg(long, value_delimiter = ',')]
    prefix_bv: Vec<usize>,
    #[arg(long, value_delimiter = ',')]
    suffix_bv: Vec<usize>,
    #[arg(long, value_delimiter = ',')]
    scan_warps: Vec<i32>,
    #[arg(long, value_delimiter = ',')]
    tp_warps: Vec<i32>,
    #[arg(long, value_delimiter = ',')]
    tp_fsb: Vec<i32>,
    /// Scan variants with the stacked x/o product: comma-separated 0/1.
    #[arg(long, value_delimiter = ',')]
    merge_xo: Vec<i32>,
    /// Time the Phase 1 (W3a) four-launch scan instead (inverse, WY, prefix scan, suffix scan).
    #[arg(long)]
    w3a: bool,
}

const SUFFIXES: usize = 8;
const SUFFIX_LEN: usize = 100;
const PROJ_COLS: usize = CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS;

/// Requests of `prefix` + 8 × 100 tokens until `t` is filled; the last request's prefix takes the remainder.
fn tree(t: usize, prefix: usize) -> DecisionBatch {
    let per = prefix + SUFFIXES * SUFFIX_LEN;
    let mut b = DecisionBatch::default();
    let mut left = t;
    while left > 0 {
        let p = if left >= per { prefix } else { left.saturating_sub(SUFFIXES * SUFFIX_LEN).max(1) };
        let n_suf = if left >= per { SUFFIXES } else { ((left - p) / SUFFIX_LEN).max(1) };
        let suffix = vec![1u32; SUFFIX_LEN.min(left - p)];
        let sufs: Vec<(&[u32], u16)> = (0..n_suf).map(|_| (&suffix[..], 2u16)).collect();
        b.push_request(&vec![1u32; p], &sufs).unwrap();
        left = t.saturating_sub(b.num_tokens());
    }
    b
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn upload(v: Vec<bf16>, shape: &[usize], s: &Arc<Stream>) -> Result<Tensor<bf16>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(s)?)
}

/// Mean seconds per run of `make()` over `iters` back-to-back submissions with one sync.
fn time(iters: usize, s: &Arc<Stream>, make: &dyn Fn() -> DeviceOpVec<()>) -> Result<f64> {
    make().sync_on(s)?; // JIT + warm-up
    let mut all = DeviceOpVec::with_capacity(iters);
    for _ in 0..iters {
        all.push(make());
    }
    s.synchronize()?;
    let t0 = Instant::now();
    all.sync_on(s)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("# {}", sysone::engine::platform::require_idle_gpu(args.allow_shared_gpu)?);
    let ctx = Context::new(0)?;
    let s = ctx.new_stream()?;
    if let Ok(dir) = std::env::var("GDN_JIT_STORE") {
        // Persist cubins (e.g. to inspect register use with cuobjdump).
        cutile::jit_cache::enable(Arc::new(cutile::jit_cache::FileSystemJitStore::new(dir)?));
    }
    if args.probe {
        return probe(&s);
    }
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let conv_w = upload((0..CONV_DIM * 4).map(|_| bf16::from_f32(rng.next() * 0.5)).collect(), &[CONV_DIM, 1, 4], &s)?;
    let a_log = upload((0..32).map(|i| bf16::from_f32(-4.0 + 0.25 * i as f32)).collect(), &[32], &s)?;
    let dt_bias = upload((0..32).map(|i| bf16::from_f32(-6.0 + 0.5 * i as f32)).collect(), &[32], &s)?;
    let norm_w = upload((0..128).map(|_| bf16::from_f32(0.9 + rng.next() * 0.2)).collect(), &[128], &s)?;
    let w = GdnWeights { conv_w: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_w: &norm_w };
    let gemm_forward_ms = |t: usize| match t {
        ..=1024 => Some(154.5),
        4096 => Some(625.4),
        16384 => Some(2320.1),
        _ => None,
    };

    let mut base = GdnOptions::default();
    base.precision = match args.precision.as_str() {
        "fp32" => GdnPrecision::Fp32Split3,
        "tf32" => GdnPrecision::Tf32,
        "bf16" => GdnPrecision::Bf16Operands,
        p => anyhow::bail!("unknown precision {p}"),
    };
    let or = |v: &Vec<usize>, d: usize| if v.is_empty() { vec![d] } else { v.clone() };
    let ori = |v: &Vec<i32>, d: i32| if v.is_empty() { vec![d] } else { v.clone() };
    let mut configs = Vec::new();
    for prefix_bv in or(&args.prefix_bv, base.prefix_bv) {
        for suffix_bv in or(&args.suffix_bv, base.suffix_bv) {
            for scan_warps in ori(&args.scan_warps, base.scan_warps) {
                for tp_warps in ori(&args.tp_warps, base.tp_warps) {
                    for tp_fsb in ori(&args.tp_fsb, base.tp_fsb) {
                        for xo in ori(&args.merge_xo, base.scan_merge_xo as i32) {
                            let scan_merge_xo = xo != 0;
                            configs.push(GdnOptions { prefix_bv, suffix_bv, scan_warps, tp_warps, tp_fsb, scan_merge_xo, ..base });
                        }
                    }
                }
            }
        }
    }
    let w3a = args.w3a;
    if w3a {
        configs.truncate(1);
    }
    for opts in configs {
        println!("# {}", if w3a { "W3a scan (tinv, wy, scan_p, scan_s)".to_string() } else { format!("{opts:?} (tinv = Tinv + p, wy unused)") });
        let scan = |plan: &GdnPlan, v: DView<bf16>, scratch: &GdnScratch, core: DView<bf16>| -> DeviceOpVec<()> {
            // SAFETY: as for the other launches below.
            unsafe {
                if w3a { gdn_chunk_scan_w3a(plan, v, scratch, core) } else { gdn_chunk_scan_with(plan, v, scratch, core, &opts) }
            }
            .unwrap()
        };
        println!(
            "{:>6} {:>6} {:>4} {:>4} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>8} {:>8} | {:>8} {:>6}",
            "T", "prefix", "reqs", "segs", "conv", "prep", "tinv", "wy", "scan_p", "scan_s", "norm", "sum", "mixer", "x24 ms", "fwd %"
        );
        for shape in &args.shapes {
            let (t, prefix) = match shape.split_once(':') {
                Some((t, p)) => (t.parse::<usize>()?, p.parse::<usize>()?),
                None => (shape.parse::<usize>()?, 2048usize.min(shape.parse::<usize>()?.saturating_sub(SUFFIXES * SUFFIX_LEN).max(1))),
            };
            let batch = tree(t, prefix);
            batch.validate()?;
            let t = batch.num_tokens();
            let plan = GdnPlan::new(&batch.segments, t, &s)?;
            let proj = upload((0..t * PROJ_COLS).map(|_| bf16::from_f32(rng.next() * 2.0)).collect(), &[t, PROJ_COLS], &s)?;
            let scratch = GdnScratch::new(t, plan.num_prefixes, &s)?;
            let out = api::zeros::<bf16>(&[t, VALUE_DIM]).sync_on(&s)?;
            let pv = DView::of(&proj)?;
            let conv = scratch.conv_view()?;
            let core = DView::of(&scratch.core)?;
            let ov = DView::of(&out)?;
            let one = |op: BoxedDeviceOp<()>| {
                let mut v = DeviceOpVec::with_capacity(1);
                v.push(op);
                v
            };
            // SAFETY (all launches below): every buffer outlives the synchronous timing loops.
            let a = pv.columns(CONV_DIM + VALUE_DIM, 32)?;
            let b = pv.columns(CONV_DIM + VALUE_DIM + 32, 32)?;
            let v = conv.columns(2 * KEY_DIM, VALUE_DIM)?;
            let z = pv.columns(CONV_DIM, VALUE_DIM)?;
            let mixer = time(args.iters, &s, &|| {
                let mut ops = DeviceOpVec::with_capacity(4);
                unsafe {
                    ops.push(tree_conv1d_silu(&plan, pv.columns(0, CONV_DIM).unwrap(), &conv_w, conv).unwrap());
                    ops.push(gdn_prep(&plan, conv, a, b, &w, &scratch).unwrap());
                    ops.push(scan(&plan, v, &scratch, core).map(|_| ()));
                    ops.push(gated_rmsnorm(&plan, core, z, &norm_w, 1e-6, ov).unwrap());
                }
                ops
            })?;
            let t_conv = time(args.iters, &s, &|| one(unsafe { tree_conv1d_silu(&plan, pv.columns(0, CONV_DIM).unwrap(), &conv_w, conv).unwrap() }))?;
            let t_prep = time(args.iters, &s, &|| one(unsafe { gdn_prep(&plan, conv, a, b, &w, &scratch).unwrap() }))?;
            // The scan is several launches (W3a: inverse, WY, prefix scan, suffix scan; P1: inverse + p, prefix scan,
            // suffix scan); time each alone by replaying the full sequence and dropping the others (inputs stay valid
            // from the mixer run above).
            let n_ops = if w3a { 4 } else { 3 };
            let mut parts = [0f64; 4];
            for k in 0..n_ops {
                let slot = if n_ops == 3 && k > 0 { k + 1 } else { k };
                parts[slot] = time(args.iters, &s, &|| {
                    let mut ops = scan(&plan, v, &scratch, core);
                    let keep = ops.remove(k);
                    let mut v = DeviceOpVec::with_capacity(1);
                    v.push(keep);
                    v
                })?;
            }
            let t_norm = time(args.iters, &s, &|| one(unsafe { gated_rmsnorm(&plan, core, z, &norm_w, 1e-6, ov).unwrap() }))?;
            let sum = t_conv + t_prep + parts.iter().sum::<f64>() + t_norm;
            let fwd = gemm_forward_ms(t).map(|g| format!("{:>5.1}%", 100.0 * 24.0 * mixer * 1e3 / (g + 24.0 * mixer * 1e3)));
            let reqs = batch.segments.iter().filter(|s| s.is_prefix()).count();
            println!(
                "{t:>6} {prefix:>6} {reqs:>4} {:>4} | {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3} | {:>8.3} {:>8.3} | {:>8.1} {:>6}",
                batch.segments.len(),
                t_conv * 1e3,
                t_prep * 1e3,
                parts[0] * 1e3,
                parts[1] * 1e3,
                parts[2] * 1e3,
                parts[3] * 1e3,
                t_norm * 1e3,
                sum * 1e3,
                mixer * 1e3,
                24.0 * mixer * 1e3,
                fwd.unwrap_or_else(|| "n/a".into()),
            );
        }
    }
    println!("ms per layer (one GDN mixer, excluding the in/out projections); fwd % = 24 layers / (Phase 0 GEMM forward + 24 layers)");
    Ok(())
}
