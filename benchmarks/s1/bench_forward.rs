//! Whole-forward benchmark of the bf16 GPU engine (Phase 2a, I1): packed steps of realistic request trees at
//! T = 1k, 4k, 16k (each request: a ~`state-tokens` JSON state and 3 questions — noul, a 5-option choice, a
//! 4-level score — planned by the real frontend), timed as CUDA-graph replays and as eager runs, plus a per-op
//! profile (CUDA events around every op in an eager run: GEMMs by projection, GDN stages, attention, norms,
//! pointwise, readout) so Phase 2b can target the real costs.
//!
//!     scripts/gpu.sh ./target/release/bench_forward --tokens 1024,4096,16384
//!
//! Refuses to run while another process holds a CUDA context (`--allow-shared-gpu` overrides and marks the
//! output as not authoritative).

use anyhow::{Context as _, Result, ensure};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;
use sysone::api::schema::SystemOneRequest;
use sysone::contract::DecisionBatch;
use sysone::engine::gpu::{EngineOptions, GpuEngine};
use sysone::engine::platform::require_idle_gpu;
use sysone::frontend::{Frontend, Limits};

#[derive(Parser, Debug)]
#[command(about = "bf16 engine forward benchmark and per-op profile")]
struct Args {
    #[arg(long, default_value = "/home/velvet/Desktop/hf_models/elephant-1")]
    model: PathBuf,
    /// Packed step sizes (T buckets) to measure.
    #[arg(long, value_delimiter = ',', default_value = "1024,4096,16384")]
    tokens: Vec<usize>,
    /// State length of each synthetic request, in words (about one token each).
    #[arg(long, default_value_t = 800)]
    state_tokens: usize,
    /// Timed graph replays per size.
    #[arg(long, default_value_t = 10)]
    iters: usize,
    /// Timed eager runs per size.
    #[arg(long, default_value_t = 3)]
    eager_iters: usize,
    /// Skip the per-op profile.
    #[arg(long)]
    no_profile: bool,
    #[arg(long)]
    allow_shared_gpu: bool,
}

const VOCAB: [&str; 16] = [
    "order", "refund", "late", "billing", "error", "customer", "angry", "shipment", "account", "card", "retry", "ok",
    "package", "warehouse", "address", "invoice",
];

/// The load generator's request shape (benchmarks/s1/loadgen.rs), 3 questions.
fn body(i: usize, words: usize) -> String {
    let mut x = (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    let w: Vec<&str> = (0..words)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            VOCAB[(x % VOCAB.len() as u64) as usize]
        })
        .collect();
    let q = [
        format!(r#""q0": {{"type": "noul", "instructions": "Should case {i} be escalated?"}}"#),
        format!(r#""q1": {{"type": "choice", "instructions": "Route case {i}", "criteria": {{"billing": "Billing team", "tech": "Technical support", "ship": "Shipping", "fraud": "Fraud review", "other": null}}}}"#),
        format!(r#""q2": {{"type": "score", "instructions": "Urgency of case {i}", "criteria": ["low", "medium", "high", "critical"]}}"#),
    ];
    format!(r#"{{"state": {{"ticket": "{}", "case": {i}}}, "questions": {{{}}}}}"#, w.join(" "), q.join(", "))
}

/// Requests packed until the next one would overflow `target` tokens.
fn batch(fe: &Frontend, target: usize, words: usize) -> Result<(DecisionBatch, usize)> {
    let mut b = DecisionBatch::default();
    let mut n = 0;
    loop {
        let req = SystemOneRequest::from_body(body(n, words).as_bytes()).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let plan = fe.plan(&req, 1).map_err(|e| anyhow::anyhow!("{e}"))?;
        if b.num_tokens() + plan.total > target {
            break;
        }
        plan.push_into(&mut b)?;
        n += 1;
    }
    ensure!(n > 0, "one request of {words} words does not fit {target} tokens");
    b.validate()?;
    Ok((b, n))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<()> {
    let a = Args::parse();
    let note = require_idle_gpu(a.allow_shared_gpu)?;
    println!("# bench_forward: {note}");
    let fe = Frontend::load(&a.model, Limits::default()).context("frontend")?;
    let max = *a.tokens.iter().max().context("no sizes")?;
    let opts = EngineOptions { buckets: a.tokens.clone(), warmup: false, ..Default::default() };
    let mut e = GpuEngine::load(&a.model, &fe.slot_ids, opts)?;
    let cfg = e.cfg.text_config.clone();
    println!("# arena {:.2} GiB for buckets {:?} (largest {max})", e.arena_bytes() as f64 / (1u64 << 30) as f64, a.tokens);
    for &target in &a.tokens {
        let (b, n_req) = batch(&fe, target, a.state_tokens)?;
        let bucket = e.pick_bucket(&b, 0).context("no bucket")?;
        let t = b.num_tokens();
        // First step: JIT for this bucket, then capture.
        let t0 = Instant::now();
        e.step(&b, bucket, true)?;
        let first = t0.elapsed().as_secs_f64();
        let mut eager = Vec::new();
        for _ in 0..a.eager_iters {
            let t0 = Instant::now();
            e.step(&b, bucket, true)?;
            eager.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let mut replay = Vec::new();
        for _ in 0..a.iters {
            let t0 = Instant::now();
            e.step(&b, bucket, false)?;
            ensure!(e.last.replayed, "bucket {} did not replay", bucket.tokens);
            replay.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        let (rm, em) = (median(replay.clone()), median(eager.clone()));
        println!(
            "\nT={t} (bucket {}, {n_req} requests, {} branches, {} segments): step replay median {rm:.1} ms (min {:.1}, \
             max {:.1}), eager median {em:.1} ms, first step {first:.1} s; {:.0} packed tok/s, {:.0} bucket tok/s, \
             {:.1} decisions/s",
            bucket.tokens,
            b.num_branches(),
            b.segments.len(),
            replay.iter().cloned().fold(f64::INFINITY, f64::min),
            replay.iter().cloned().fold(0.0, f64::max),
            t as f64 / rm * 1e3,
            bucket.tokens as f64 / rm * 1e3,
            (n_req * 3) as f64 / rm * 1e3,
        );
        let mut rec = serde_json::json!({
            "tokens": t, "bucket": bucket.tokens, "requests": n_req, "branches": b.num_branches(),
            "replay_ms_median": rm, "eager_ms_median": em, "replay_ms": replay, "eager_ms": eager,
        });
        if !a.no_profile {
            let (ops, total) = e.profile(&b, bucket)?;
            let tb = bucket.tokens as f64;
            // GEMM FLOPs per projection label (2·T·N·K per launch).
            let h = cfg.hidden_size as f64;
            let n_of = |label: &str| -> Option<(f64, f64)> {
                Some(match label.strip_prefix("gemm.")? {
                    "GdnInProj" => (12352.0, h),
                    "GdnOutProj" => (h, 4096.0),
                    "AttnQ" => (8192.0, h),
                    "AttnKv" => (2048.0, h),
                    "AttnO" => (h, 4096.0),
                    "MlpGateUp" => (2.0 * cfg.intermediate_size as f64, h),
                    "MlpDown" => (h, cfg.intermediate_size as f64),
                    _ => return None,
                })
            };
            println!("  per-op profile (eager, CUDA events; total {total:.1} ms):");
            println!("  {:<26} {:>9} {:>6} {:>8} {:>9}", "op", "ms", "%", "launches", "TFLOP/s");
            let mut sorted = ops.clone();
            sorted.sort_by(|x, y| y.1.total_cmp(&x.1));
            let mut classes: BTreeMap<String, f64> = BTreeMap::new();
            for (label, ms, n) in &sorted {
                if label == "end" {
                    continue;
                }
                let tf = n_of(label).map(|(nn, k)| 2.0 * tb * nn * k * *n as f64 / (ms * 1e-3) / 1e12);
                println!("  {label:<26} {ms:>9.2} {:>6.1} {n:>8} {:>9}", ms / total * 100.0,
                         tf.map(|x| format!("{x:.1}")).unwrap_or_default());
                let class = label.split('.').next().unwrap_or(label).to_string();
                *classes.entry(class).or_default() += ms;
            }
            let cl: Vec<String> = classes.iter().map(|(k, v)| format!("{k} {v:.1} ms ({:.1}%)", v / total * 100.0)).collect();
            println!("  by class: {}", cl.join(", "));
            rec["profile_total_ms"] = serde_json::json!(total);
            rec["profile"] = serde_json::json!(ops.iter().map(|(l, ms, n)| serde_json::json!([l, ms, n])).collect::<Vec<_>>());
        }
        println!("{rec}");
    }
    Ok(())
}
