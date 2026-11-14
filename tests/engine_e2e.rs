//! End-to-end gates of the GPU engine (DESIGN §9.2; Phase 2a, I1; NVFP4: N1) on the 52-request model corpus.
//! The engine is selected by `$SYSONE_ENGINE` (`bf16`, the default, or `nvfp4`):
//!
//! - `slot_probabilities_match_goldens`: every branch's slot probabilities vs the reference.
//!   - bf16: vs qev bf16 (batched fork), gated by `[e2e.bf16]` (dp p99/max, top-1 agreement where the reference
//!     margin exceeds `top1_margin`); also reported (and gated by `[e2e.fp32_truth]`) against qev fp32.
//!   - nvfp4: vs the NVFP4 QDQ per-question reference (`nvfp4_perq_r1`, the checkpoint's own evaluation path),
//!     gated by `[nvfp4]` (dp p99/max; its envelope is the QDQ reference's cross-process drift and its
//!     per-question vs shared-prefix spread); reported vs the shared-prefix QDQ paths and, as information on the
//!     cost of quantization, vs qev bf16.
//! - `batch_invariance`: each request alone vs packed with the others and vs alone in a larger T bucket, gated by
//!   `[batch_invariance].dp_max` (bf16) or `[batch_invariance.nvfp4].dp_max` (nvfp4); bit-identity is reported.
//! - `graph_replay_equals_eager`: a captured T-bucket graph replayed with new batch metadata gives bit for bit
//!   the eager forward's logits.
//! - `f1_launches_equal_n1_launches` (both engines): F1's fused SiLU·mul GEMM and occupancy hints vs N1's launches,
//!   bit for bit.
//! - `s2_launches_equal_f1_launches` (both engines): S2's residual add in the down GEMM and its `gated_rmsnorm` /
//!   `zc_rmsnorm` launches vs F1's launches, bit for bit (logits and every debug tap kind).
//! - `nvfp4_fused_equals_unfused` (nvfp4 only): the engine with the activation quantization fused into the norm /
//!   SiLU·mul epilogues gives bit for bit the logits of the engine with standalone quantize launches.
//! - `trimmed_last_layer_equals_untrimmed` (L1): the last layer trimmed to the answer rows gives bit for bit the
//!   untrimmed logits with pinned GEMMs (the default); with cuBLAS's per-call heuristic the difference is reported.
//! - `packing_diagnostic`, `trim_diagnostic` (ignored): locate the first tap where packed vs alone, or trimmed vs
//!   untrimmed, differ.
//!
//! Needs the GPU and the checkpoint: bf16 from `$SYSONE_MODEL_DIR` (default /home/velvet/Desktop/hf_models/
//! elephant-1), NVFP4 from `$SYSONE_NVFP4_DIR` (default /home/velvet/Desktop/hf_models/elephant-1-nvfp4-mlp):
//! `scripts/gpu.sh cargo test --release --test engine_e2e -- --nocapture --test-threads=1`
//! `SYSONE_ENGINE=nvfp4 scripts/gpu.sh cargo test --release --test engine_e2e -- --nocapture --test-threads=1`

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use sysone::contract::{DecisionBatch, DecisionEngine, MAX_SLOTS, SlotLogits};
use sysone::engine::goldens::{self, GoldenResult, Planned};
use sysone::engine::config::WeightFormat;
use sysone::engine::gpu::{EngineOptions, GpuEngine, gdn_backend_from_env, weight_format_from_env};
use sysone::frontend::{Frontend, Limits};
use sysone::tolerances::{Stats, Tolerances};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The engine under test (`$SYSONE_ENGINE`, default bf16).
fn format() -> WeightFormat {
    weight_format_from_env().unwrap().unwrap_or(WeightFormat::Bf16)
}

fn model_dir() -> PathBuf {
    let (var, default) = match format() {
        WeightFormat::Bf16 => ("SYSONE_MODEL_DIR", "/home/velvet/Desktop/hf_models/elephant-1"),
        WeightFormat::Nvfp4 => ("SYSONE_NVFP4_DIR", "/home/velvet/Desktop/hf_models/elephant-1-nvfp4-mlp"),
    };
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

/// The reference run whose request statuses select the corpus (and that the gate compares against).
fn primary_run() -> &'static str {
    match format() {
        WeightFormat::Bf16 => "bf16_batched",
        WeightFormat::Nvfp4 => "nvfp4_perq_r1",
    }
}

fn engine_opts(buckets_to: usize, nvfp4_fused: bool) -> EngineOptions {
    EngineOptions { buckets: EngineOptions::buckets_up_to(buckets_to), warmup: false,
                    gdn_backend: gdn_backend_from_env().unwrap(), weights: Some(format()), nvfp4_fused,
                    ..Default::default() }
}

struct Env {
    engine: GpuEngine,
    plans: Vec<Planned>,
}

fn env() -> &'static Mutex<Env> {
    static ENV: OnceLock<Mutex<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
        let gold = goldens::golden_run(&repo(), primary_run()).unwrap();
        let cases = goldens::model_corpus(&repo()).unwrap();
        let plans = goldens::plan_corpus(&fe, &cases, |id| gold[id].status == 200).unwrap();
        let engine = GpuEngine::load(&model_dir(), &fe.slot_ids, engine_opts(32768, true)).unwrap();
        eprintln!("engine_e2e: {} engine, GDN backend {}", engine.format.name(), engine.gdn_backend().name());
        Mutex::new(Env { engine, plans })
    })
}

fn tol() -> Tolerances {
    Tolerances::load_default().unwrap()
}

/// Per-branch slot logits of one request (`first` = its first answer row in the batch).
fn rows(l: &SlotLogits, first: usize, p: &Planned) -> Vec<Vec<f64>> {
    p.plan.branches.iter().enumerate().map(|(i, b)| l.row(first + i, b.n_slots()).iter().map(|&x| x as f64).collect()).collect()
}

struct Diff {
    dp: Vec<f64>,
    dlogit: Vec<f64>,
    /// (request, branch, reference margin) of top-1 disagreements.
    flips: Vec<(String, usize, f64)>,
}

fn compare(results: &[(String, Vec<Vec<f64>>)], gold: &std::collections::HashMap<String, GoldenResult>) -> Diff {
    let mut d = Diff { dp: vec![], dlogit: vec![], flips: vec![] };
    for (id, branches) in results {
        let g = &gold[id];
        assert_eq!(g.branches.len(), branches.len(), "{id}: branch count");
        for (bi, (ours, gb)) in branches.iter().zip(&g.branches).enumerate() {
            assert_eq!(ours.len(), gb.logits.len(), "{id} branch {bi}: slot count");
            let (pa, pb) = (goldens::softmax(ours), goldens::softmax(&gb.logits));
            d.dp.extend(pa.iter().zip(&pb).map(|(a, b)| (a - b).abs()));
            d.dlogit.extend(ours.iter().zip(&gb.logits).map(|(a, b)| (a - b).abs()));
            let arg = |p: &[f64]| p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            if arg(&pa) != arg(&pb) {
                let mut s = pb.clone();
                s.sort_by(|a, b| b.total_cmp(a));
                let margin = s[0] - s.get(1).copied().unwrap_or(0.0);
                d.flips.push((id.clone(), bi, margin));
            }
        }
    }
    d
}

/// Every request alone (one step each).
fn run_alone(env: &mut Env) -> Vec<(String, Vec<Vec<f64>>)> {
    let mut out = Vec::new();
    for p in &env.plans {
        let (b, first) = goldens::batch_of(&[&p.plan]).unwrap();
        let l = env.engine.run(&b).unwrap();
        out.push((p.case.id.clone(), rows(&l, first[0], p)));
    }
    out
}

#[test]
fn slot_probabilities_match_goldens() {
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let t0 = std::time::Instant::now();
    let results = run_alone(env);
    let n_branches: usize = results.iter().map(|r| r.1.len()).sum();
    println!("ran {} requests ({n_branches} branches) alone in {:.1} s", results.len(), t0.elapsed().as_secs_f64());
    let t = tol();
    let mut failed = Vec::new();
    let runs: &[(&str, &str)] = match format() {
        WeightFormat::Bf16 => &[("bf16_batched", "e2e.bf16"), ("fp32_batched", "e2e.fp32_truth"), ("bf16_sequential", "")],
        // Gated against the checkpoint's own evaluation path; the shared-prefix QDQ paths and bf16 are reported.
        WeightFormat::Nvfp4 => &[("nvfp4_perq_r1", "nvfp4"), ("nvfp4_batched", ""), ("nvfp4_sequential", ""),
                                 ("bf16_batched", ""), ("fp32_batched", "")],
    };
    for &(run, table) in runs {
        let gold = goldens::golden_run(&repo(), run).unwrap();
        let d = compare(&results, &gold);
        let (dp, dl) = (Stats::of(&d.dp).unwrap(), Stats::of(&d.dlogit).unwrap());
        println!("sysone vs qev {run}: dp {dp}; dlogit {dl}; top-1 disagreements {}", d.flips.len());
        for (id, b, m) in &d.flips {
            println!("    flip {id} branch {b}: reference margin {m:.4}");
        }
        if table.is_empty() {
            continue;
        }
        let lim = |k: &str| t.f64(&format!("{table}.{k}")).unwrap();
        println!("    [{table}] dp_p99 {} dp_max {} dlogit_p99 {} dlogit_max {}", lim("dp_p99"), lim("dp_max"),
                 lim("dlogit_p99"), lim("dlogit_max"));
        for (what, got, max) in [("dp p99", dp.p99, lim("dp_p99")), ("dp max", dp.max, lim("dp_max"))] {
            if got > max {
                failed.push(format!("{run}: {what} {got:.4e} > {max}"));
            }
        }
        if table == "e2e.bf16" {
            let m = lim("top1_margin");
            for (id, b, margin) in d.flips.iter().filter(|f| f.2 > m) {
                failed.push(format!("{run}: top-1 flip at {id} branch {b} with reference margin {margin:.4} > {m}"));
            }
        }
    }
    assert!(failed.is_empty(), "gate failures:\n  {}", failed.join("\n  "));
}

#[test]
fn batch_invariance() {
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let t = tol();
    let alone = run_alone(env);
    // Packed: greedy groups of whole requests up to a 16k-token budget.
    let mut packed: Vec<(String, Vec<Vec<f64>>)> = Vec::new();
    let mut i = 0;
    let mut groups = 0;
    while i < env.plans.len() {
        let mut j = i;
        let mut tokens = 0;
        while j < env.plans.len() && (j == i || tokens + env.plans[j].plan.total <= 16384) {
            tokens += env.plans[j].plan.total;
            j += 1;
        }
        let group: Vec<&_> = env.plans[i..j].iter().map(|p| &p.plan).collect();
        let (b, first) = goldens::batch_of(&group).unwrap();
        let l = env.engine.run(&b).unwrap();
        println!("packed group {groups}: {} requests, {} tokens, bucket {}", j - i, b.num_tokens(), env.engine.last.bucket_tokens);
        for (k, p) in env.plans[i..j].iter().enumerate() {
            packed.push((p.case.id.clone(), rows(&l, first[k], p)));
        }
        groups += 1;
        i = j;
    }
    // Alone, but in a bucket 4x larger than it needs (different padding, grid sizes and GEMM shapes).
    let mut bigger = Vec::new();
    for p in &env.plans {
        let (b, first) = goldens::batch_of(&[&p.plan]).unwrap();
        let Some(bucket) = env.engine.pick_bucket(&b, b.num_tokens() * 4) else { continue };
        let l = env.engine.step(&b, bucket, false).unwrap();
        bigger.push((p.case.id.clone(), rows(&l, first[0], p)));
    }
    let table = match format() {
        WeightFormat::Bf16 => "batch_invariance",
        WeightFormat::Nvfp4 => "batch_invariance.nvfp4",
    };
    let lim = t.f64(&format!("{table}.dp_max")).unwrap();
    let mut worst = 0.0f64;
    for (what, other) in [("packed", &packed), ("larger bucket", &bigger)] {
        let (mut same, mut total, mut dp, mut dl) = (0usize, 0usize, Vec::new(), Vec::new());
        for (id, branches) in other.iter() {
            let a = &alone.iter().find(|x| &x.0 == id).unwrap().1;
            for (x, y) in a.iter().zip(branches) {
                total += 1;
                same += (x == y) as usize;
                let (px, py) = (goldens::softmax(x), goldens::softmax(y));
                dp.extend(px.iter().zip(&py).map(|(a, b)| (a - b).abs()));
                dl.extend(x.iter().zip(y).map(|(a, b)| (a - b).abs()));
            }
        }
        let s = Stats::of(&dp).unwrap();
        worst = worst.max(s.max);
        println!("alone vs {what}: {same}/{total} branches bit-identical; dp {s}; dlogit {}", Stats::of(&dl).unwrap());
    }
    println!("batch invariance: max |dp| {worst:.3e} (gate [{table}].dp_max {lim}; bf16's [batch_invariance].dp_max {}, \
              DESIGN start {})", t.f64("batch_invariance.dp_max").unwrap(),
             t.f64("batch_invariance.design_start_dp_max").unwrap());
    assert!(worst <= lim, "batch invariance max |dp| {worst} > {lim}");
}

/// Batches of several shapes run eagerly, then (after the bucket's graph was captured) as a replay.
#[test]
fn graph_replay_equals_eager() {
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let picks = ["m000_noul_default", "m006_choice_5_debias3", "m019_state_json", "m020_state_json_nested", "m037_len1024_debias1",
                 "m033_multi_24q_debias1"];
    let mut batches: Vec<DecisionBatch> = Vec::new();
    for id in picks {
        let p = env.plans.iter().find(|p| p.case.id == id).unwrap();
        batches.push(goldens::batch_of(&[&p.plan]).unwrap().0);
    }
    // Two requests packed.
    let two: Vec<&_> = env.plans.iter().filter(|p| picks[..2].contains(&p.case.id.as_str())).map(|p| &p.plan).collect();
    batches.push(goldens::batch_of(&two).unwrap().0);
    let mut checked = 0;
    for b in &batches {
        let bucket = env.engine.pick_bucket(b, 0).unwrap();
        let eager = env.engine.step(b, bucket, true).unwrap();
        assert!(!env.engine.last.replayed);
        // Make sure the bucket is captured (the first non-eager step captures), then replay.
        env.engine.step(b, bucket, false).unwrap();
        let replay = env.engine.step(b, bucket, false).unwrap();
        assert!(env.engine.last.replayed, "bucket {} was not replayed", bucket.tokens);
        let same = eager.data == replay.data;
        println!("T={} bucket {} Q={}: replay {} eager", b.num_tokens(), bucket.tokens, b.num_branches(),
                 if same { "bit-identical to" } else { "DIFFERS from" });
        assert!(same, "graph replay differs from eager at T={}", b.num_tokens());
        checked += 1;
    }
    // A replay after a different batch ran in the same bucket must still match eager (metadata is rewritten).
    let (a, c) = (&batches[2], &batches[1]);
    let bucket = env.engine.pick_bucket(a, 0).unwrap();
    assert_eq!(env.engine.pick_bucket(c, 0).unwrap(), bucket);
    let ea = env.engine.step(a, bucket, true).unwrap();
    env.engine.step(c, bucket, false).unwrap();
    let ra = env.engine.step(a, bucket, false).unwrap();
    assert!(env.engine.last.replayed);
    assert_eq!(ea.data, ra.data, "replay after another batch differs from eager");
    println!("{checked} batches + 1 interleaved: graph replay bit-identical to eager");
}

/// NVFP4 only: the fused-quantize engine against an engine with standalone `quantize_nvfp4` launches, bit for bit,
/// on every corpus request up to 4k tokens alone and on one packed batch (eager and replayed).
#[test]
fn nvfp4_fused_equals_unfused() {
    if format() != WeightFormat::Nvfp4 {
        println!("skipped: SYSONE_ENGINE is not nvfp4");
        return;
    }
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let mut unfused = GpuEngine::load(&model_dir(), &fe.slot_ids, engine_opts(4096, false)).unwrap();
    let small: Vec<&Planned> = env.plans.iter().filter(|p| p.plan.total <= 4096).collect();
    let (mut same, mut total) = (0usize, 0usize);
    let mut batches: Vec<DecisionBatch> = small.iter().map(|p| goldens::batch_of(&[&p.plan]).unwrap().0).collect();
    let mut packed = Vec::new();
    let mut tokens = 0;
    for p in &small {
        if tokens + p.plan.total > 4096 {
            break;
        }
        tokens += p.plan.total;
        packed.push(&p.plan);
    }
    batches.push(goldens::batch_of(&packed).unwrap().0);
    for b in &batches {
        for _ in 0..2 {
            // First pass eager (and capture), second a replay in both engines.
            let (x, y) = (env.engine.run(b).unwrap(), unfused.run(b).unwrap());
            total += 1;
            same += (x.data == y.data) as usize;
        }
    }
    println!("fused vs standalone quantize: {same}/{total} steps bit-identical ({} batches, eager and replay)",
             batches.len());
    assert_eq!(same, total, "the fused NVFP4 quantization changed the logits");
}

/// L1: the last layer trimmed to the answer rows against the untrimmed forward, on every corpus request up to 4k
/// tokens alone and on packed batches of up to 4k and 16k tokens (eager and replayed). With pinned GEMMs (the
/// default) the logits must be bit-identical. With cuBLAS's per-call heuristic the trimmed GEMMs have fewer rows
/// and may run another kernel: reported, not asserted.
#[test]
fn trimmed_last_layer_equals_untrimmed() {
    use sysone::engine::gemm::GemmMode;
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let small: Vec<&Planned> = env.plans.iter().filter(|p| p.plan.total <= 4096).collect();
    let mut batches: Vec<DecisionBatch> = small.iter().map(|p| goldens::batch_of(&[&p.plan]).unwrap().0).collect();
    for budget in [4096usize, 16384] {
        let (mut packed, mut tokens) = (Vec::new(), 0);
        for p in &env.plans {
            if tokens + p.plan.total <= budget {
                tokens += p.plan.total;
                packed.push(&p.plan);
            }
        }
        batches.push(goldens::batch_of(&packed).unwrap().0);
    }
    let run = |e: &mut GpuEngine| -> Vec<SlotLogits> {
        // Each batch twice: eager (+ capture) on its bucket's first use, then a replay.
        batches.iter().flat_map(|b| [e.run(b).unwrap(), e.run(b).unwrap()]).collect()
    };
    let load = |gemm: GemmMode, trim: bool| {
        let opts = EngineOptions { gemm, trim_last_layer: trim, ..engine_opts(16384, true) };
        let e = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
        assert_eq!(e.fwd.trimmed, trim);
        e
    };
    let compare = |a: &[SlotLogits], b: &[SlotLogits]| {
        let (mut same, mut rows, mut same_rows, mut worst) = (0usize, 0usize, 0usize, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            same += (x.data == y.data) as usize;
            for r in 0..x.rows {
                rows += 1;
                let (xr, yr) = (&x.data[r * MAX_SLOTS..(r + 1) * MAX_SLOTS], &y.data[r * MAX_SLOTS..(r + 1) * MAX_SLOTS]);
                same_rows += (xr == yr) as usize;
                for (u, v) in xr.iter().zip(yr) {
                    worst = worst.max((u - v).abs() as f64);
                }
            }
        }
        (same, a.len(), same_rows, rows, worst)
    };
    assert!(env.engine.fwd.trimmed && env.engine.opts.gemm == GemmMode::Pinned, "the default engine is trimmed and pinned");
    let pinned_trim = run(&mut env.engine);
    let pinned_full = run(&mut load(GemmMode::Pinned, false));
    let (s, n, sr, r, w) = compare(&pinned_trim, &pinned_full);
    println!("pinned GEMMs: trimmed vs untrimmed last layer: {s}/{n} steps, {sr}/{r} answer rows bit-identical \
              (max |dlogit| {w:.3e}; {} batches, eager and replay)", batches.len());
    let heur_trim = run(&mut load(GemmMode::Heuristic, true));
    let heur_full = run(&mut load(GemmMode::Heuristic, false));
    let (hs, hn, hsr, hr, hw) = compare(&heur_trim, &heur_full);
    println!("heuristic GEMMs (reported): trimmed vs untrimmed last layer: {hs}/{hn} steps, {hsr}/{hr} answer rows \
              bit-identical (max |dlogit| {hw:.3e})");
    let (ps, pn, psr, pr, pw) = compare(&pinned_full, &heur_full);
    println!("untrimmed, pinned vs heuristic GEMMs (reported): {ps}/{pn} steps, {psr}/{pr} answer rows bit-identical \
              (max |dlogit| {pw:.3e})");
    assert_eq!(s, n, "with pinned GEMMs the trimmed last layer changed the logits");
}

/// Diagnostic (ignored): where a request packed after another one first departs from the request alone. Per
/// differing pair, the first layer (residual stream after it) and sub-layer (mixer, mid, MLP) whose rows of the
/// request differ.
#[test]
#[ignore]
fn packing_diagnostic() {
    use sysone::engine::forward::Taps;
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let mut opts = engine_opts(8192, true);
    opts.taps = Taps { layers: (0..32).collect(), sublayers: true, ..Default::default() };
    let mut e = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
    let small: Vec<&Planned> = env.plans.iter().filter(|p| p.plan.total <= 2048).collect();
    let mut shown = 0;
    'outer: for (ai, a) in small.iter().enumerate() {
        for b in small.iter().skip(ai + 1).take(3) {
            let (alone, _) = goldens::batch_of(&[&a.plan]).unwrap();
            let la = e.run(&alone).unwrap();
            let ta = a.plan.total;
            let mut taps_a = Vec::new();
            for l in 0..32 {
                let attn: Vec<String> = if l % 4 == 3 { vec![format!("k{l}"), format!("q{l}"), format!("attn{l}")] } else { vec![] };
                for n in attn.into_iter().chain([format!("mix{l}"), format!("mid{l}"), format!("mlp{l}"), format!("layer{l}")]) {
                    taps_a.push((n.clone(), e.read_tap(&n, ta).unwrap()));
                }
            }
            let (packed, first) = goldens::batch_of(&[&b.plan, &a.plan]).unwrap();
            let lp = e.run(&packed).unwrap();
            let q = a.plan.branches.len();
            let same = (0..q).all(|i| la.data[i * MAX_SLOTS..(i + 1) * MAX_SLOTS]
                == lp.data[(first[1] + i) * MAX_SLOTS..(first[1] + i + 1) * MAX_SLOTS]);
            if same {
                continue;
            }
            let off = b.plan.total;
            let mut line = format!("{} after {} (offset {off}, T {}):", a.case.id, b.case.id, packed.num_tokens());
            for (n, va) in &taps_a {
                let vp = e.read_tap(n, off + ta).unwrap();
                let h = va.len() / ta;
                let bad: Vec<usize> = (0..ta).filter(|&r| va[r * h..(r + 1) * h] != vp[(off + r) * h..(off + r + 1) * h]).collect();
                if !bad.is_empty() {
                    line += &format!(" first differing tap {n}: {} of {ta} rows (first rows {:?})", bad.len(),
                                     &bad[..bad.len().min(6)]);
                    break;
                }
            }
            println!("{line}");
            shown += 1;
            if shown >= 6 {
                break 'outer;
            }
        }
    }
}

/// Diagnostic (ignored): where a trimmed step first departs from the untrimmed one. Compares the trimmed path's
/// answer-row taps (`Taps::answer_rows`) with the untrimmed last-layer taps at the answer rows, per batch.
#[test]
#[ignore]
fn trim_diagnostic() {
    use sysone::engine::forward::Taps;
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let last = 31usize;
    let mut t_opts = engine_opts(16384, true);
    t_opts.taps = Taps { answer_rows: true, ..Default::default() };
    let mut u_opts = engine_opts(16384, true);
    u_opts.taps = Taps { layers: vec![last - 1, last], norm: true, sublayers: true, ..Default::default() };
    let mut te = GpuEngine::load(&model_dir(), &fe.slot_ids, t_opts).unwrap();
    let mut ue = GpuEngine::load(&model_dir(), &fe.slot_ids, u_opts).unwrap();
    assert!(te.fwd.trimmed && !ue.fwd.trimmed);
    let small: Vec<&Planned> = env.plans.iter().filter(|p| p.plan.total <= 4096).collect();
    let mut shown = 0;
    for p in small {
        let b = goldens::batch_of(&[&p.plan]).unwrap().0;
        let (x, y) = (te.run(&b).unwrap(), ue.run(&b).unwrap());
        if x.data == y.data {
            continue;
        }
        let q = b.num_branches();
        let t = b.num_tokens();
        let h = 4096;
        let mut line = format!("{} T={t} Q={q} bucket {}:", p.case.id, te.last.bucket_tokens);
        let (ka, kf) = (te.read_tap("k31_all", t).unwrap(), ue.read_tap("k31", t).unwrap());
        let kbad = ka.chunks(1024).zip(kf.chunks(1024)).filter(|(x, y)| x != y).count();
        line += &format!(" k31: {kbad}/{t} token rows differ;");
        for (tn, un) in [("q31_a", "q31"), ("attn31_a", "attn31"), ("mix31_a", "mix31"), ("mid31_a", "mid31"), ("mlp31_a", "mlp31"),
                         ("norm_a", "norm")] {
            let a = te.read_tap(tn, q).unwrap();
            let f = ue.read_tap(un, t).unwrap();
            let mut bad = 0;
            let mut maxd = 0f32;
            for (i, &r) in b.answer_rows.iter().enumerate() {
                let (ra, rf) = (&a[i * h..(i + 1) * h], &f[r as usize * h..(r as usize + 1) * h]);
                let n = ra.iter().zip(rf).filter(|(u, v)| u != v).count();
                bad += (n > 0) as usize;
                maxd = ra.iter().zip(rf).fold(maxd, |m, (u, v)| m.max((u - v).abs()));
            }
            line += &format!(" {tn}: {bad}/{q} rows differ (max {maxd:.3e});");
        }
        println!("{line}");
        shown += 1;
        if shown >= 8 {
            break;
        }
    }
}

/// F1 (docs/F1.md): the engine as it now runs (nvfp4: SiLU·mul + quantize in the gate/up GEMM epilogue on
/// interleaved gate/up rows; both engines: occupancy hints on the residual-add norms and SiLU·mul) against the
/// engine with N1's launches (separate bf16 gate/up GEMM + `silu_mul_nvfp4`, compiler-default norms), bit for
/// bit, on every corpus request up to 4k tokens alone and on one packed batch (eager and replayed).
#[test]
fn f1_launches_equal_n1_launches() {
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let mut n1 = GpuEngine::load(&model_dir(), &fe.slot_ids, EngineOptions { nvfp4_swiglu: false,
                                                                          ..engine_opts(4096, true) }).unwrap();
    n1.set_kernel_tuning(sysone::engine::kernels::KernelTuning::N1);
    let small: Vec<&Planned> = env.plans.iter().filter(|p| p.plan.total <= 4096).collect();
    let (mut same, mut total) = (0usize, 0usize);
    let mut batches: Vec<DecisionBatch> = small.iter().map(|p| goldens::batch_of(&[&p.plan]).unwrap().0).collect();
    let mut packed = Vec::new();
    let mut tokens = 0;
    for p in &small {
        if tokens + p.plan.total > 4096 {
            break;
        }
        tokens += p.plan.total;
        packed.push(&p.plan);
    }
    batches.push(goldens::batch_of(&packed).unwrap().0);
    for b in &batches {
        for _ in 0..2 {
            let (x, y) = (env.engine.run(b).unwrap(), n1.run(b).unwrap());
            total += 1;
            same += (x.data == y.data) as usize;
        }
    }
    println!("{} engine, F1 launches vs N1 launches: {same}/{total} steps bit-identical ({} batches, eager and replay)",
             format().name(), batches.len());
    assert_eq!(same, total, "F1's launches changed the logits");
}

/// S2 (docs/S2.md): the engine as it now runs (nvfp4: the residual add after the MLP down projection in the down
/// GEMM's epilogue, `zc_rmsnorm_res` as the next norm; both engines: `gated_rmsnorm` one token's 32 heads per tile
/// block, layer 0's `zc_rmsnorm` with an occupancy hint) against the same build with F1's launches
/// (`nvfp4_residual: false`, `KernelTuning::F1`), bit for bit: logits on every corpus request up to 4k tokens alone
/// and on one packed batch (eager and replayed), then every debug tap (per-layer residuals of fused layers, the
/// sublayer taps of an unfused tapped layer, the trimmed layer's answer-row taps) on a few requests.
#[test]
fn s2_launches_equal_f1_launches() {
    use sysone::engine::forward::Taps;
    use sysone::engine::kernels::KernelTuning;
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let small: Vec<&Planned> = env.plans.iter().filter(|p| p.plan.total <= 4096).collect();
    let mut batches: Vec<DecisionBatch> = small.iter().map(|p| goldens::batch_of(&[&p.plan]).unwrap().0).collect();
    let mut packed = Vec::new();
    let mut tokens = 0;
    for p in &small {
        if tokens + p.plan.total > 4096 {
            break;
        }
        tokens += p.plan.total;
        packed.push(&p.plan);
    }
    batches.push(goldens::batch_of(&packed).unwrap().0);
    let f1_engine = |opts: EngineOptions| {
        let mut e = GpuEngine::load(&model_dir(), &fe.slot_ids, EngineOptions { nvfp4_residual: false, ..opts }).unwrap();
        e.set_kernel_tuning(KernelTuning::F1);
        e
    };
    {
        let mut f1 = f1_engine(engine_opts(4096, true));
        let (mut same, mut total) = (0usize, 0usize);
        for b in &batches {
            for _ in 0..2 {
                let (x, y) = (env.engine.run(b).unwrap(), f1.run(b).unwrap());
                total += 1;
                same += (x.data == y.data) as usize;
            }
        }
        println!("{} engine, S2 launches vs F1 launches: {same}/{total} steps bit-identical ({} batches, eager and replay)",
                 format().name(), batches.len());
        assert_eq!(same, total, "S2's launches changed the logits");
    }
    // Debug taps: (a) residual stream after layers whose residual add is fused (S2 taps it before the plain norm);
    // (b) the sublayer taps of tapped layers (no fusion there) and the trimmed last layer's answer-row taps.
    let tap_sets = [Taps { layers: vec![0, 2, 7, 29], ..Default::default() },
                    Taps { layers: vec![3, 30], sublayers: true, answer_rows: true, ..Default::default() }];
    let mut bad = Vec::new();
    for taps in tap_sets {
        let opts = || EngineOptions { taps: taps.clone(), cuda_graphs: false, ..engine_opts(4096, true) };
        let mut s2 = GpuEngine::load(&model_dir(), &fe.slot_ids, opts()).unwrap();
        let mut f1 = f1_engine(opts());
        assert!(s2.fwd.trimmed && f1.fwd.trimmed);
        let names: Vec<String> = s2.fwd.taps.iter().map(|(n, _)| n.clone()).collect();
        assert_eq!(names, f1.fwd.taps.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>());
        for b in batches.iter().step_by(10) {
            let (x, y) = (s2.run(b).unwrap(), f1.run(b).unwrap());
            if x.data != y.data {
                bad.push(format!("logits with taps {:?}", taps.layers));
            }
            for n in &names {
                let rows = if n.ends_with("_a") { b.num_branches() } else { b.num_tokens() };
                let (u, v) = (s2.read_tap(n, rows).unwrap(), f1.read_tap(n, rows).unwrap());
                let diff = u.iter().zip(&v).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
                if diff > 0 {
                    bad.push(format!("tap {n}: {diff} values differ"));
                }
            }
        }
        println!("{} engine, S2 vs F1 taps {names:?}: {} batches compared", format().name(), batches.iter().step_by(10).count());
    }
    assert!(bad.is_empty(), "S2 changed debug taps: {bad:?}");
}
