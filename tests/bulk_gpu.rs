//! `sysone bulk` with the bf16 GPU engine against the HTTP server with the same engine (needs the checkpoint and the
//! GPU; run it under the GPU lock, which the bulk workers' nested gpu.sh then reuses):
//!
//! ```text
//! scripts/gpu.sh cargo test --release --test bulk_gpu -- --nocapture
//! ```
//!
//! One test, in sequence (each engine releases the GPU before the next starts):
//! 1. **alone**: bulk with a one-request window (every request alone in its step, in the smallest bucket, as the
//!    server runs a request that arrives alone) must equal the server's responses to sequential requests byte for
//!    byte apart from `qev.timing`, for every status.
//! 2. **packed**: bulk with its normal window packs several requests per step. Statuses and every non-float field
//!    must equal the server's; probabilities within `[batch_invariance].dp_max` and debug slot logits within
//!    `[batch_invariance].dlogit_max` (the existing gate for "packed with others vs alone"); a `choice` may differ
//!    only where the server's top-1 margin is at most `[e2e.bf16].top1_margin`.
//! 3. **determinism**: the packed run again gives the same bytes (apart from timing).
//! 4. **two engines on one GPU** (`--gpus 0,0`, the multi-GPU path on a single GPU): the same steps as run 2
//!    (equal limits give the same packing), so the same bytes, with both workers taking steps.
//!
//! The corpus: the 52-request model corpus (with its debias, every 4th with `debug=1`), the render corpus's error
//! cases, and synthetic documents of 300-6,000 tokens.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use sysone::api::client::Client;
use sysone::api::server::{self, AppState, ServerInfo};
use sysone::contract::DecisionEngine;
use sysone::engine::goldens;
use sysone::engine::gpu::{EngineOptions, GpuEngine};
use sysone::frontend::pyjson::{self, Json};
use sysone::frontend::{Frontend, Limits};
use sysone::sched::{EngineFactory, SchedConfig};
use sysone::tolerances::{Stats, Tolerances};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"))
}

const MAX_STEP: &str = "65536";

fn parse(b: &[u8]) -> Json {
    pyjson::parse(b).unwrap_or_else(|e| panic!("not JSON ({e:?}): {}", String::from_utf8_lossy(b)))
}

fn untimed(v: &Json) -> Json {
    let mut v = v.clone();
    if let Json::Object(m) = &mut v {
        for (k, q) in m.iter_mut() {
            if k == "qev"
                && let Json::Object(q) = q
            {
                q.retain(|(k, _)| k != "timing");
            }
        }
    }
    v
}

/// id → (status, body without timing).
fn results(path: &Path) -> BTreeMap<String, (u16, Json)> {
    let mut m = BTreeMap::new();
    for l in std::fs::read(path).unwrap().split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        let v = parse(l);
        let id = v.get("id").unwrap().as_str().unwrap().to_string();
        let status = match v.get("status") {
            Some(Json::Int(s)) => s.parse().unwrap(),
            other => panic!("{other:?}"),
        };
        assert!(m.insert(id.clone(), (status, untimed(v.get("body").unwrap()))).is_none(), "{id} twice");
    }
    m
}

fn bulk(dir: &Path, name: &str, input: &Path, extra: &[&str]) -> (BTreeMap<String, (u16, Json)>, Json) {
    let out = dir.join(format!("{name}.jsonl"));
    let _ = std::fs::remove_file(&out);
    let o = Command::new(env!("CARGO_BIN_EXE_sysone"))
        .arg("bulk")
        .arg("--model")
        .arg(model_dir())
        .args(["--max-step-tokens", MAX_STEP, "--progress-s", "0", "--input"])
        .arg(input)
        .arg("--output")
        .arg(&out)
        .args(extra)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "bulk {name} failed:\n{err}");
    let summary = parse(String::from_utf8_lossy(&o.stdout).lines().last().unwrap().as_bytes());
    println!("bulk {name}: {}", summary.dumps_compact().unwrap());
    (results(&out), summary)
}

#[derive(Default)]
struct Diffs {
    dp: Vec<f64>,
    dlogit: Vec<f64>,
    derived: Vec<f64>,
    errors: Vec<String>,
    flips: Vec<(String, f64)>,
}

fn num(j: &Json) -> Option<f64> {
    match j {
        Json::Float(x) => Some(*x),
        Json::Int(s) => s.parse().ok(),
        _ => None,
    }
}

fn top1_margin(probs: Option<&Json>) -> f64 {
    let mut p: Vec<f64> = probs.and_then(|j| j.as_object()).map(|o| o.iter().filter_map(|(_, v)| num(v)).collect()).unwrap_or_default();
    p.sort_by(|a, b| b.total_cmp(a));
    p.first().copied().unwrap_or(0.0) - p.get(1).copied().unwrap_or(0.0)
}

fn walk(path: &str, ours: &Json, theirs: &Json, margin: f64, d: &mut Diffs) {
    let in_answers = path.starts_with("answers.");
    match (ours, theirs) {
        (Json::Object(a), Json::Object(b)) => {
            let ka: Vec<&str> = a.iter().map(|(k, _)| k.as_str()).collect();
            let kb: Vec<&str> = b.iter().map(|(k, _)| k.as_str()).collect();
            if ka != kb {
                d.errors.push(format!("{path}: keys {ka:?} vs {kb:?}"));
                return;
            }
            let m = if theirs.get("probabilities").is_some() { top1_margin(theirs.get("probabilities")) } else { margin };
            for k in ka {
                let p = if path.is_empty() { k.to_string() } else { format!("{path}.{k}") };
                walk(&p, ours.get(k).unwrap(), theirs.get(k).unwrap(), m, d);
            }
        }
        (Json::Array(a), Json::Array(b)) if a.len() == b.len() => {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                walk(&format!("{path}[{i}]"), x, y, margin, d);
            }
        }
        (x, y) if matches!(x, Json::Float(_)) || matches!(y, Json::Float(_)) => {
            let (Some(a), Some(b)) = (num(x), num(y)) else {
                d.errors.push(format!("{path}: {x:?} vs {y:?}"));
                return;
            };
            let diff = (a - b).abs();
            if path.contains(".slot_logits") {
                d.dlogit.push(diff);
            } else if in_answers && (path.contains(".probabilities.") || path.ends_with(".noul")) {
                d.dp.push(diff);
            } else if in_answers {
                d.derived.push(diff);
            } else if diff != 0.0 {
                d.errors.push(format!("{path}: {x:?} vs {y:?}"));
            }
        }
        (Json::Str(x), Json::Str(y)) if in_answers && path.ends_with(".choice") && x != y => {
            d.flips.push((format!("{path}: {x} vs {y}"), margin));
        }
        (x, y) => {
            if x != y {
                d.errors.push(format!("{path}: {:.200} vs {:.200}", x.dumps_compact().unwrap(), y.dumps_compact().unwrap()));
            }
        }
    }
}

/// The corpus as (id, query, body) and a job file.
fn corpus(dir: &Path) -> (PathBuf, Vec<(String, String, String)>) {
    let mut jobs = Vec::new();
    for (k, c) in goldens::model_corpus(&repo()).unwrap().into_iter().enumerate() {
        let q = if k % 4 == 0 { format!("debug=1&debias={}", c.debias) } else { format!("debias={}", c.debias) };
        jobs.push((c.id, q, c.body));
    }
    for l in std::fs::read_to_string(repo().join("reference/corpus/requests.jsonl")).unwrap().lines() {
        let v = parse(l.as_bytes());
        let id = v.get("id").unwrap().as_str().unwrap().to_string();
        if id.starts_with("invalid") || id.starts_with("limits") {
            let debias = num(v.get("debias").unwrap()).unwrap() as i64;
            jobs.push((id, format!("debias={debias}"), v.get("body").unwrap().dumps_compact().unwrap()));
        }
    }
    let spec = sysone::bulk::synth::CorpusSpec { docs: 40, min_words: 300, max_words: 6000, min_questions: 4, max_questions: 16, seed: 7 };
    for i in 0..spec.docs {
        let (w, q) = sysone::bulk::synth::doc_shape(&spec, i);
        jobs.push((format!("doc-{i:03}"), String::new(), sysone::bulk::synth::body(i, w, q)));
    }
    let text: String = jobs
        .iter()
        .map(|(id, q, b)| {
            let id = Json::str(id.clone()).dumps_compact().unwrap();
            if q.is_empty() {
                format!("{{\"id\": {id}, \"body_text\": {}}}\n", Json::str(b.clone()).dumps_compact().unwrap())
            } else {
                format!("{{\"id\": {id}, \"body_text\": {}, \"query\": \"{q}\"}}\n", Json::str(b.clone()).dumps_compact().unwrap())
            }
        })
        .collect();
    let p = dir.join("jobs.jsonl");
    std::fs::write(&p, text).unwrap();
    (p, jobs)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bulk_on_the_gpu_matches_the_server() {
    assert!(
        std::env::var_os("SYSONE_GPU_HELD").is_some(),
        "run under scripts/gpu.sh: the bulk workers reuse its GPU lock (two engines on one GPU need it)"
    );
    let dir = std::env::temp_dir().join(format!("sysone-bulk-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (input, jobs) = corpus(&dir);
    let t = Tolerances::load_default().unwrap();

    let (alone, _) = tokio::task::block_in_place(|| bulk(&dir, "alone", &input, &["--window", "1"]));
    let (packed, s_packed) = tokio::task::block_in_place(|| bulk(&dir, "packed", &input, &["--window", "64", "--trace", dir.join("packed.trace").to_str().unwrap()]));
    let (again, _) = tokio::task::block_in_place(|| bulk(&dir, "again", &input, &["--window", "64"]));
    let (two, s_two) = tokio::task::block_in_place(|| bulk(&dir, "two-engines", &input, &["--window", "64", "--gpus", "0,0"]));

    // The server (same buckets: powers of two up to MAX_STEP), one request at a time.
    let fe = Arc::new(Frontend::load(&model_dir(), Limits::default()).unwrap());
    let slot_ids = fe.slot_ids.clone();
    let factory: EngineFactory = Box::new(move || {
        let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(MAX_STEP.parse().unwrap()), ..Default::default() };
        Ok(Box::new(GpuEngine::load(&model_dir(), &slot_ids, opts)?) as Box<dyn DecisionEngine>)
    });
    let state = AppState::start(fe, ServerInfo::new("elephant-1", "qwen3_5"), SchedConfig::default(), factory, 4).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (stop, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        server::serve(listener, state, async {
            let _ = rx.await;
        })
        .await
        .unwrap()
    });
    let mut c = Client::connect(&addr).await.unwrap();
    let mut served = BTreeMap::new();
    for (id, q, body) in &jobs {
        let path = if q.is_empty() { "/v1/systemone".to_string() } else { format!("/v1/systemone?{q}") };
        let r = c.post(&path, body.clone()).await.unwrap();
        served.insert(id.clone(), (r.status.as_u16(), untimed(&parse(&r.body))));
    }
    let _ = stop.send(());
    task.await.unwrap();

    let mut failed = Vec::new();
    let statuses: BTreeMap<u16, usize> = served.values().fold(BTreeMap::new(), |mut m, (s, _)| {
        *m.entry(*s).or_default() += 1;
        m
    });
    println!("{} jobs, server statuses {statuses:?}", jobs.len());

    // 1. Alone: byte-identical.
    let mut n_diff = 0;
    for (id, (s, body)) in &served {
        let (bs, bb) = &alone[id];
        if bs != s || bb.dumps_compact().unwrap() != body.dumps_compact().unwrap() {
            n_diff += 1;
            if n_diff <= 5 {
                println!("alone MISMATCH {id}: bulk {bs} {:.300}\n    server {s} {:.300}", bb.dumps_compact().unwrap(), body.dumps_compact().unwrap());
            }
        }
    }
    println!("alone vs server: {} of {} responses differ (byte comparison without qev.timing)", n_diff, served.len());
    if n_diff > 0 {
        failed.push(format!("alone: {n_diff} responses differ from the server's"));
    }

    // 2. Packed: the batch-invariance gate.
    let mut d = Diffs::default();
    for (id, (s, body)) in &served {
        let (bs, bb) = &packed[id];
        if bs != s {
            d.errors.push(format!("{id}: status {bs} vs {s}"));
            continue;
        }
        let before = d.errors.len();
        walk("", bb, body, 1.0, &mut d);
        for e in d.errors.iter_mut().skip(before) {
            *e = format!("{id}: {e}");
        }
    }
    let dp_max = t.f64("batch_invariance.dp_max").unwrap();
    let dl_max = t.f64("batch_invariance.dlogit_max").unwrap();
    let margin = t.f64("e2e.bf16.top1_margin").unwrap();
    let dp = Stats::of(&d.dp).unwrap();
    println!("packed vs server: probabilities {dp} (gate max {dp_max}); identical {} of {}", d.dp.iter().filter(|&&x| x == 0.0).count(), d.dp.len());
    if let Some(dl) = Stats::of(&d.dlogit) {
        println!("    debug slot_logits {dl} (gate max {dl_max})");
        if dl.max > dl_max {
            failed.push(format!("packed: slot logits max {:.3e}", dl.max));
        }
    }
    println!("    score/confidence {}", Stats::of(&d.derived).map(|s| s.to_string()).unwrap_or_default());
    println!("    steps {}, padding {:.4}", num(s_packed.get("steps").unwrap()).unwrap(), num(s_packed.get("padding_frac").unwrap()).unwrap());
    if dp.max > dp_max {
        failed.push(format!("packed: probabilities max {:.3e} > {dp_max}", dp.max));
    }
    for (f, m) in &d.flips {
        println!("    choice flip {f} (server margin {m:.4})");
        if *m > margin {
            failed.push(format!("packed: {f} with margin {m:.4}"));
        }
    }
    for e in d.errors.iter().take(10) {
        println!("    MISMATCH {e}");
    }
    if !d.errors.is_empty() {
        failed.push(format!("packed: {} mismatches", d.errors.len()));
    }

    // 3 and 4: the same packing gives the same bytes, on one engine or two.
    let same = |a: &BTreeMap<String, (u16, Json)>, b: &BTreeMap<String, (u16, Json)>| {
        a.len() == b.len() && a.iter().all(|(k, (s, v))| b.get(k).is_some_and(|(s2, v2)| s == s2 && v.dumps_compact() == v2.dumps_compact()))
    };
    if !same(&packed, &again) {
        failed.push("determinism: a second packed run differs".into());
    }
    let ws = s_two.get("workers").unwrap().as_array().unwrap();
    let steps: Vec<f64> = ws.iter().map(|w| num(w.get("steps").unwrap()).unwrap()).collect();
    println!("two engines on one GPU: steps per worker {steps:?}");
    if ws.len() != 2 || steps.iter().any(|&s| s == 0.0) {
        failed.push(format!("two engines: steps per worker {steps:?}"));
    }
    if !same(&packed, &two) {
        failed.push("two engines on one GPU: results differ from one engine's".into());
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failed.is_empty(), "bulk GPU failures:\n  {}", failed.join("\n  "));
}
