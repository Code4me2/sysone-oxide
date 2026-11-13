//! End-to-end gates of the bf16 GPU engine (DESIGN §9.2; Phase 2a, I1) on the 52-request model corpus:
//!
//! - `slot_probabilities_match_bf16_goldens`: every branch's slot probabilities vs qev bf16 (batched fork), gated
//!   by `[e2e.bf16]` (dp p99/max, top-1 agreement where the reference margin exceeds `top1_margin`); also
//!   reported (and gated by `[e2e.fp32_truth]`) against qev fp32;
//! - `batch_invariance`: each request alone vs packed with the others and vs alone in a larger T bucket, gated by
//!   `[batch_invariance].dp_max` (bit-identity is reported);
//! - `graph_replay_equals_eager`: a captured T-bucket graph replayed with new batch metadata gives bit for bit
//!   the eager forward's logits.
//!
//! Needs the GPU and the checkpoint (`$SYSONE_MODEL_DIR`, default /home/velvet/Desktop/hf_models/elephant-1):
//! `scripts/gpu.sh cargo test --release --test engine_e2e -- --nocapture --test-threads=1`

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use sysone::contract::{DecisionBatch, DecisionEngine, SlotLogits};
use sysone::engine::goldens::{self, GoldenResult, Planned};
use sysone::engine::gpu::{EngineOptions, GpuEngine};
use sysone::frontend::{Frontend, Limits};
use sysone::tolerances::{Stats, Tolerances};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"))
}

struct Env {
    engine: GpuEngine,
    plans: Vec<Planned>,
}

fn env() -> &'static Mutex<Env> {
    static ENV: OnceLock<Mutex<Env>> = OnceLock::new();
    ENV.get_or_init(|| {
        let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
        let gold = goldens::golden_run(&repo(), "bf16_batched").unwrap();
        let cases = goldens::model_corpus(&repo()).unwrap();
        let plans = goldens::plan_corpus(&fe, &cases, |id| gold[id].status == 200).unwrap();
        let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(32768), warmup: false, ..Default::default() };
        let engine = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
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
fn slot_probabilities_match_bf16_goldens() {
    let mut guard = env().lock().unwrap_or_else(|e| e.into_inner());
    let env = &mut *guard;
    let t0 = std::time::Instant::now();
    let results = run_alone(env);
    let n_branches: usize = results.iter().map(|r| r.1.len()).sum();
    println!("ran {} requests ({n_branches} branches) alone in {:.1} s", results.len(), t0.elapsed().as_secs_f64());
    let t = tol();
    let mut failed = Vec::new();
    for (run, table) in [("bf16_batched", "e2e.bf16"), ("fp32_batched", "e2e.fp32_truth"), ("bf16_sequential", "")] {
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
    let lim = t.f64("batch_invariance.dp_max").unwrap();
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
    println!("batch invariance: max |dp| {worst:.3e} (gate [batch_invariance].dp_max {lim}, DESIGN start {})",
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
