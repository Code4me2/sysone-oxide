//! W3b microbenchmark: segment attention, norms, RoPE, pointwise and readout kernels at packed batch sizes
//! T in {1k, 4k, 16k} with realistic decision trees (DESIGN.md §6.1: requests = one prefix + N suffixes).
//!
//! Each kernel is captured `--iters` times into one CUDA graph and the graph is replayed, so the numbers
//! are device time per launch without host launch overhead. Run under the shared GPU lock:
//!
//! ```text
//! flock /tmp/sysone-gpu.lock cargo run --release --bin bench_attn
//! ```

use anyhow::Result;
use clap::Parser;
use cutile::half::bf16;
use cutile::prelude::*;
use cuda_async::cuda_graph::CudaGraph;
use std::time::Instant;
use sysone::contract::DecisionBatch;
use sysone::kernels::attention::{self as attn, AttnPlan, AttnTiles, SegAttnArgs};
use sysone::kernels::{norm, pointwise, readout};

#[derive(Parser)]
struct Args {
    /// Measure even if other processes use the GPU (the output is then marked not authoritative).
    #[arg(long)]
    allow_shared_gpu: bool,
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384")]
    tokens: Vec<usize>,
    #[arg(long, default_value_t = 10)]
    iters: usize,
    /// Interleaved timing rounds per seg_attn config (the minimum and the median are reported).
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Also sweep seg_attn tile shapes and P split.
    #[arg(long)]
    sweep: bool,
    /// Also time cutile-kernels' `fmha_prefill_gqa` (f16, one dense non-causal sequence of T tokens) at
    /// D = 256 and D = 128, as a ceiling reference for this GPU and toolchain.
    #[arg(long)]
    baseline: bool,
    /// Persist compiled cubins in this directory (for inspecting register and shared-memory use with
    /// `cuobjdump -res-usage`).
    #[arg(long)]
    jit_cache: Option<String>,
    /// Time these seg_attn configs instead of the default list: `;`-separated, each
    /// `bm,bn,p_split,warps,occupancy,d_split[,latency[,residual_f16]]` (e.g. `16,32,3,4,2,2`).
    #[arg(long)]
    configs: Option<String>,
    /// Time only seg_attn (skip the norm, RoPE, pointwise and readout kernels).
    #[arg(long)]
    attn_only: bool,
}

fn parse_config(s: &str, dt: AttnTiles) -> Result<AttnTiles> {
    let v: Vec<usize> = s.split(',').map(|x| x.trim().parse()).collect::<Result<_, _>>()?;
    anyhow::ensure!((6..=8).contains(&v.len()), "config {s:?}: expected 6 to 8 fields");
    Ok(AttnTiles {
        bm: v[0],
        bn: v[1],
        p_split: v[2],
        warps: v[3],
        occupancy: v[4],
        d_split: v[5],
        latency: v.get(6).copied().unwrap_or(dt.latency),
        residual_f16: v.get(7).is_some_and(|&r| r != 0),
    })
}

/// cutile-kernels `fmha_prefill_gqa` on one non-causal sequence: `t` queries x 16 heads over `t` keys.
fn baseline(stream: &Arc<cuda_core::Stream>, t: usize, d: usize, bm: usize, bn: usize, iters: usize) -> Result<()> {
    use cutile::half::f16;
    let (hq, hk) = (16usize, 4usize);
    let mut rng = Rng(7);
    let mk = |n: usize, rng: &mut Rng| -> Vec<f16> { rng.bf16s(n, 1.0).iter().map(|v| f16::from_f32(v.to_f32())).collect() };
    let q: Arc<Tensor<f16>> = upload(mk(t * hq * d, &mut rng), &[t, hq, d], stream)?.into();
    let k: Arc<Tensor<f16>> = upload(mk(hk * t * d, &mut rng), &[hk, t, d], stream)?.into();
    let v: Arc<Tensor<f16>> = upload(mk(hk * t * d, &mut rng), &[hk, t, d], stream)?.into();
    let mut out = api::zeros::<f16>(&[t, hq, d]).sync_on(stream)?;
    let g = hq / hk;
    let gens = [bm, bn, d, g, bm * g, 0, usize::from(t.is_multiple_of(bn)), 2].iter().map(|x| x.to_string()).collect::<Vec<_>>();
    macro_rules! launch {
        ($out:expr) => {
            unsafe {
                cutile_kernels::attention::fmha_prefill_gqa(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    $out.partition([bm, g, d]),
                    1.0 / (d as f32).sqrt(),
                    g as i32,
                    t as i32,
                    0,
                )
            }
            .generics(gens.clone())
        };
    }
    launch!(&mut out).sync_on(stream)?;
    let ms = time_graph(stream, iters, |s| s.record(launch!(&mut out)).map(|_| ()))?;
    let flops = 4.0 * (hq * d) as f64 * (t * t) as f64;
    println!("baseline fmha_prefill_gqa f16 D={d} bm={bm} bn={bn} T={t}: {ms:.3} ms  {:.1} TFLOP/s", flops / ms / 1e9);
    Ok(())
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn bf16s(&mut self, n: usize, scale: f32) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale)).collect()
    }
}

/// A packed batch of exactly `t` tokens: (state length, branches, suffix length) per request, chosen like
/// the DESIGN.md §10 sweep (states 256..12k, 3..24 questions), the last request topped up to fill `t`.
fn tree(t: usize) -> DecisionBatch {
    let reqs: &[(usize, usize, usize)] = match t {
        ..=1024 => &[(380, 3, 44), (380, 3, 44)],
        1025..=2048 => &[(1100, 6, 100), (200, 2, 58)],
        2049..=4096 => &[(2900, 8, 110), (200, 2, 58)],
        _ => &[(12000, 24, 100), (1500, 4, 121)],
    };
    let mut b = DecisionBatch::default();
    let used: usize = reqs.iter().map(|(p, n, s)| p + n * s).sum();
    for (i, &(p, n, s)) in reqs.iter().enumerate() {
        let p = if i + 1 == reqs.len() { p + t.saturating_sub(used) } else { p };
        let suf = vec![1u32; s];
        let branches: Vec<(&[u32], u16)> = (0..n).map(|_| (suf.as_slice(), 26u16)).collect();
        b.push_request(&vec![1u32; p], &branches).unwrap();
    }
    b.token_ids.truncate(t);
    b
}

fn upload<T: DType>(v: Vec<T>, shape: &[usize], stream: &Arc<cuda_core::Stream>) -> Result<Tensor<T>> {
    Ok(api::copy_host_vec_to_device(&Arc::new(v)).reshape(shape).sync_on(stream)?)
}

/// Capture `iters` copies of the ops `make` records into one graph, replay it, return ms per launch.
fn time_graph<F>(stream: &Arc<cuda_core::Stream>, iters: usize, mut make: F) -> Result<f64>
where
    F: FnMut(&cuda_async::cuda_graph::Scope) -> Result<(), cuda_async::error::DeviceError>,
{
    let graph = CudaGraph::scope(stream, |s| {
        for _ in 0..iters {
            make(s)?;
        }
        Ok(())
    })?;
    graph.launch().sync_on(stream)?; // warm
    let mut best = f64::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        graph.launch().sync_on(stream)?;
        best = best.min(start.elapsed().as_secs_f64());
    }
    Ok(best * 1e3 / iters as f64)
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn dev_err(e: anyhow::Error) -> cuda_async::error::DeviceError {
    cuda_async::error::DeviceError::Launch(e.to_string())
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("# {}", sysone::engine::platform::require_idle_gpu(args.allow_shared_gpu)?);
    if let Some(dir) = &args.jit_cache {
        let store = cutile::jit_cache::FileSystemJitStore::builder(dir).capacity_bytes(256 << 20).open()?;
        cutile::jit_cache::enable(Arc::new(store));
    }
    let device = Context::new(0)?;
    let stream = device.new_stream()?;
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let (hq, hk, d, hidden, inter) = (16usize, 4usize, 256usize, 4096usize, 12288usize);

    for &t in &args.tokens {
        if args.baseline {
            for (d, bm, bn) in [(256, 16, 64), (256, 16, 32), (128, 16, 64), (128, 32, 64)] {
                if let Err(e) = baseline(&stream, t, d, bm, bn, args.iters) {
                    println!("baseline D={d} bm={bm} bn={bn}: {e}");
                }
            }
        }
        let b = tree(t);
        b.validate()?;
        let n_req = b.segments.iter().filter(|s| s.is_prefix()).count();
        let q = b.num_branches();
        // Useful attention FLOPs: QK^T and PV over every allowed (query, key) pair, 16 heads.
        let pairs: usize = b
            .segments
            .iter()
            .map(|s| s.len as usize * (s.len as usize + if s.is_prefix() { 0 } else { s.pos0 as usize }))
            .sum();
        let flops = 4.0 * (hq * d) as f64 * pairs as f64;
        println!("\n=== T = {t}: {n_req} requests, {q} branches, {} segments, {pairs} attended pairs ===", b.segments.len());

        // Attention inputs.
        let qg = upload(rng.bf16s(t * hq * 2 * d, 2.0), &[t, hq, 2 * d], &stream)?;
        let kv = upload(rng.bf16s(t * 2 * hk * d, 2.0), &[t, 2 * hk, d], &stream)?;
        let qn = upload(rng.bf16s(t * hq * d, 2.0), &[t, hq, d], &stream)?;
        let kn = upload(rng.bf16s(t * hk * d, 1.0), &[t, hk, d], &stream)?;
        let out = api::zeros::<bf16>(&[t, hq, d]).sync_on(&stream)?;

        let dt = AttnTiles::default();
        // The tiles the engine uses at this T, then the precision knobs (DESIGN.md §1): P in 2 bf16 parts, in
        // a bf16 + f16 pair, and the classic flash-attention bf16 P (see tests/kernels_attn.rs for their error).
        let best = AttnTiles::for_tokens(t);
        let mut configs = vec![best];
        if best != dt {
            configs.push(dt);
        }
        configs.push(AttnTiles { p_split: 2, ..best });
        configs.push(AttnTiles { p_split: 2, residual_f16: true, ..dt });
        configs.push(AttnTiles { bm: 8, p_split: 1, ..dt });
        if args.sweep {
            // (bm, bn, p_split, warps, occupancy, d_split, latency); warps/occupancy 0 = compiler default.
            // warps = 16 is not offered: tileiras rejects `num_worker_warps_per_cta=16` for this kernel.
            let sweep: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
                (16, 32, 3, 8, 0, 1, 2),
                (16, 32, 3, 8, 0, 1, 1),
                (16, 32, 3, 8, 0, 1, 3),
                (16, 32, 3, 8, 1, 1, 2),
                (16, 16, 3, 8, 0, 1, 2),
                (16, 64, 3, 8, 0, 1, 2),
                (8, 32, 3, 8, 0, 1, 2),
                (8, 32, 3, 4, 2, 1, 1),
                (32, 32, 3, 8, 0, 1, 2),
                (16, 32, 3, 4, 2, 2, 2),
                (16, 32, 3, 4, 2, 2, 1),
                (16, 32, 3, 4, 2, 2, 3),
                (16, 32, 3, 8, 0, 2, 2),
                (16, 32, 3, 4, 1, 2, 2),
                (16, 16, 3, 4, 2, 2, 2),
                (16, 64, 3, 4, 2, 2, 1),
                (8, 32, 3, 4, 2, 2, 1),
                (8, 32, 3, 8, 2, 2, 1),
                (32, 32, 3, 8, 0, 2, 2),
                (32, 32, 3, 4, 0, 2, 2),
                (16, 32, 3, 4, 2, 4, 2),
                (16, 32, 2, 8, 0, 1, 2),
                (16, 32, 2, 4, 2, 2, 2),
                (8, 32, 1, 8, 0, 1, 2),
                (16, 32, 1, 8, 0, 1, 2),
            ];
            for &(bm, bn, p_split, warps, occupancy, d_split, latency) in sweep {
                configs.push(AttnTiles { bm, bn, p_split, warps, occupancy, d_split, latency, ..dt });
            }
        }
        if let Some(c) = &args.configs {
            configs = c.split(';').filter(|x| !x.trim().is_empty()).map(|x| parse_config(x, dt)).collect::<Result<_>>()?;
        }
        // Compile and check every config, then time them in `--rounds` interleaved rounds (min and median
        // per config), so a transient slowdown from other processes on the unified memory hits all configs alike.
        let mut runs = Vec::new();
        for tiles in configs {
            let mut plan = AttnPlan::new(&b.segments, tiles.bm);
            let items = plan.num_items();
            plan.pad_to(AttnPlan::max_items(t, b.segments.len(), tiles.bm));
            let work = upload(plan.work.clone(), &[plan.work.len()], &stream)?;
            let a = SegAttnArgs {
                out: out.device_pointer(),
                q: qn.device_pointer(),
                k: kn.device_pointer(),
                kv: kv.device_pointer(),
                qg: qg.device_pointer(),
                work: work.device_pointer(),
                num_items: plan.num_items(),
            };
            let res = unsafe { attn::seg_attn(a, tiles) }.sync_on(&stream);
            if let Err(e) = res {
                println!("seg_attn {tiles:?}: {e}");
                continue;
            }
            runs.push((tiles, items, a, work, Vec::new()));
        }
        for _ in 0..args.rounds {
            for (tiles, _, a, _, ms) in runs.iter_mut() {
                let (a, tiles) = (*a, *tiles);
                ms.push(time_graph(&stream, args.iters, |s| s.record(unsafe { attn::seg_attn(a, tiles) }).map(|_| ()))?);
            }
        }
        for (tiles, items, a, _, ms) in runs {
            let ms = sorted(ms);
            let (best, med) = (ms[0], ms[ms.len() / 2]);
            println!(
                "seg_attn      bm={:<3} bn={:<3} lat={} p_split={} warps={} occ={} d_split={} rf16={}  items={items:<5} (grid {}x4x{})  {best:>8.3} ms  {:>6.1} useful TFLOP/s  (median {med:.3} ms, {:.1})",
                tiles.bm,
                tiles.bn,
                tiles.latency,
                tiles.p_split,
                tiles.warps,
                tiles.occupancy,
                tiles.d_split,
                u8::from(tiles.residual_f16),
                a.num_items,
                tiles.d_split,
                flops / best / 1e9,
                flops / med / 1e9,
            );
        }

        if args.attn_only {
            continue;
        }
        // qk_norm_rope.
        let pos = upload(attn::positions(&b.segments), &[t], &stream)?;
        let qw = upload(rng.bf16s(d, 0.5), &[d], &stream)?;
        let kw = upload(rng.bf16s(d, 0.5), &[d], &stream)?;
        let inv = upload(attn::rope_inv_freq().to_vec(), &[32], &stream)?;
        let mut qo = api::zeros::<bf16>(&[t, hq, d]).sync_on(&stream)?;
        let mut ko = api::zeros::<bf16>(&[t, hk, d]).sync_on(&stream)?;
        attn::qk_norm_rope(&mut qo, &mut ko, &qg, &kv, &pos, &qw, &kw, &inv, norm::RMS_EPS)?.sync_on(&stream)?;
        let ms = time_graph(&stream, args.iters, |s| {
            s.record(attn::qk_norm_rope(&mut qo, &mut ko, &qg, &kv, &pos, &qw, &kw, &inv, norm::RMS_EPS).map_err(dev_err)?)
                .map(|_| ())
        })?;
        let bytes = (t * (hq * d + hk * d) * 2 * 2) as f64;
        println!("qk_norm_rope  {ms:>8.3} ms  {:>6.1} GB/s", bytes / ms / 1e6);
        drop((qg, kv, qn, kn, out, qo, ko));

        // Norms.
        let x = upload(rng.bf16s(t * hidden, 2.0), &[t, hidden], &stream)?;
        let w = upload(rng.bf16s(hidden, 0.5), &[hidden], &stream)?;
        let mut res = upload(rng.bf16s(t * hidden, 4.0), &[t, hidden], &stream)?;
        let mut y = api::zeros::<bf16>(&[t, hidden]).sync_on(&stream)?;
        norm::zc_rmsnorm(&mut y, &x, &w, norm::RMS_EPS)?.sync_on(&stream)?;
        let ms = time_graph(&stream, args.iters, |s| {
            s.record(norm::zc_rmsnorm(&mut y, &x, &w, norm::RMS_EPS).map_err(dev_err)?).map(|_| ())
        })?;
        println!("zc_rmsnorm    {ms:>8.3} ms  {:>6.1} GB/s", (t * hidden * 4) as f64 / ms / 1e6);
        norm::add_zc_rmsnorm(&mut res, &mut y, &x, &w, norm::RMS_EPS)?.sync_on(&stream)?;
        let ms = time_graph(&stream, args.iters, |s| {
            s.record(norm::add_zc_rmsnorm(&mut res, &mut y, &x, &w, norm::RMS_EPS).map_err(dev_err)?).map(|_| ())
        })?;
        println!("add_zc_rmsnorm{ms:>8.3} ms  {:>6.1} GB/s", (t * hidden * 8) as f64 / ms / 1e6);

        // Readout: gather the answer rows, then the slot GEMM.
        let rows = upload(b.answer_rows.clone(), &[q], &stream)?;
        let mut hs = api::zeros::<bf16>(&[q, hidden]).sync_on(&stream)?;
        pointwise::gather_rows(&mut hs, &x, &rows)?.sync_on(&stream)?;
        let ms = time_graph(&stream, args.iters, |s| {
            s.record(pointwise::gather_rows(&mut hs, &x, &rows).map_err(dev_err)?).map(|_| ())
        })?;
        println!("gather_rows   {ms:>8.3} ms  (Q = {q})");
        let wslot = upload(rng.bf16s(256 * hidden, 0.05), &[256, hidden], &stream)?;
        let mut logits = api::zeros::<f32>(&[q, 256]).sync_on(&stream)?;
        readout::slot_logits(&mut logits, &hs, &wslot)?.sync_on(&stream)?;
        let ms = time_graph(&stream, args.iters, |s| {
            s.record(readout::slot_logits(&mut logits, &hs, &wslot).map_err(dev_err)?).map(|_| ())
        })?;
        println!("slot_logits   {ms:>8.3} ms  (Q = {q})");
        drop((res, y, hs, wslot, logits));

        // Embedding gather.
        let table = upload(rng.bf16s(8192 * hidden, 1.0), &[8192, hidden], &stream)?;
        let ids = upload((0..t).map(|_| (rng.next() % 8192) as u32).collect::<Vec<_>>(), &[t], &stream)?;
        let mut emb = api::zeros::<bf16>(&[t, hidden]).sync_on(&stream)?;
        pointwise::embedding(&mut emb, &table, &ids)?.sync_on(&stream)?;
        let ms = time_graph(&stream, args.iters, |s| {
            s.record(pointwise::embedding(&mut emb, &table, &ids).map_err(dev_err)?).map(|_| ())
        })?;
        println!("embedding     {ms:>8.3} ms  {:>6.1} GB/s", (t * hidden * 4) as f64 / ms / 1e6);
        drop((x, table, ids, emb));

        // SwiGLU product.
        let gu = upload(rng.bf16s(t * 2 * inter, 4.0), &[t, 2 * inter], &stream)?;
        let mut m = api::zeros::<bf16>(&[t, inter]).sync_on(&stream)?;
        pointwise::silu_mul(&mut m, &gu)?.sync_on(&stream)?;
        let ms = time_graph(&stream, args.iters, |s| s.record(pointwise::silu_mul(&mut m, &gu).map_err(dev_err)?).map(|_| ()))?;
        println!("silu_mul      {ms:>8.3} ms  {:>6.1} GB/s", (t * inter * 6) as f64 / ms / 1e6);
    }
    Ok(())
}
