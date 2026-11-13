//! The 52-request model corpus over HTTP against a server running the bf16 GPU engine, compared with qev's
//! bf16 responses (Phase 2a, I1). Every request is POSTed as qev's golden run did (`?debug=1&debias=k`), first
//! one at a time, then all at once (so the scheduler packs them). Each response must equal the golden body
//! byte for byte as JSON (keys, order, strings, integers), except:
//! - probabilities (`answers.*.probabilities.*`, `answers.*.noul`): within `[e2e.bf16]` dp_p99/dp_max;
//! - `answers.*.choice`: may differ only where the reference's top-1 margin is ≤ `[e2e.bf16].top1_margin`;
//! - derived floats (`score`, `confidence`) and `qev.diagnostics.*.slot_logits`: reported (logits gated by
//!   `[e2e.bf16].dlogit_max`);
//! - `qev.timing` (ours) and `qev.diagnostics.*.allowed_token_mass` (qev; sysone does not compute it yet) are
//!   skipped.
//!
//! By default the test starts the server in-process on a free port (real sockets, scheduler, GPU worker).
//! With `SYSONE_SERVER_ADDR=host:port` it targets an already running `sysone --engine bf16` instead.
//!
//! `scripts/gpu.sh cargo test --release --test engine_server -- --nocapture`

use std::path::PathBuf;
use std::sync::Arc;

use sysone::api::client::Client;
use sysone::api::server::{self, AppState, ServerInfo};
use sysone::contract::DecisionEngine;
use sysone::engine::goldens;
use sysone::engine::gpu::{EngineOptions, GpuEngine, gdn_backend_from_env};
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

#[derive(Default)]
struct Diffs {
    dp: Vec<f64>,
    derived: Vec<f64>,
    dlogit: Vec<f64>,
    errors: Vec<String>,
    choice_flips: Vec<String>,
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

/// Walk `ours` against `gold` at `path` (dot-joined keys).
fn walk(path: &str, ours: &Json, gold: &Json, margin: f64, d: &mut Diffs) {
    let in_answers = path.starts_with("answers.");
    match (ours, gold) {
        (Json::Object(a), Json::Object(b)) => {
            let skip = |k: &str| k == "timing" || k == "allowed_token_mass";
            let ka: Vec<&str> = a.iter().map(|(k, _)| k.as_str()).filter(|k| !skip(k)).collect();
            let kb: Vec<&str> = b.iter().map(|(k, _)| k.as_str()).filter(|k| !skip(k)).collect();
            if ka != kb {
                d.errors.push(format!("{path}: keys {ka:?} vs {kb:?}"));
                return;
            }
            // The reference margin of this answer (for a choice flip).
            let m = if gold.get("probabilities").is_some() { top1_margin(gold.get("probabilities")) } else { margin };
            for k in ka {
                let p = if path.is_empty() { k.to_string() } else { format!("{path}.{k}") };
                walk(&p, ours.get(k).unwrap(), gold.get(k).unwrap(), m, d);
            }
        }
        (Json::Array(a), Json::Array(b)) => {
            if a.len() != b.len() {
                d.errors.push(format!("{path}: {} vs {} items", a.len(), b.len()));
                return;
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                walk(&format!("{path}[{i}]"), x, y, margin, d);
            }
        }
        (x, y) if num(x).is_some() && num(y).is_some() && (matches!(x, Json::Float(_)) || matches!(y, Json::Float(_))) => {
            let diff = (num(x).unwrap() - num(y).unwrap()).abs();
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
            d.choice_flips.push(format!("{path}: {x} vs {y} (reference margin {margin:.4})"));
        }
        (x, y) => {
            if x != y {
                d.errors.push(format!("{path}: {x:?} vs {y:?}"));
            }
        }
    }
}

async fn post_all(addr: &str, cases: &[(String, String, i64)], concurrent: bool) -> Vec<(u16, Vec<u8>)> {
    let one = |addr: String, body: String, debias: i64| async move {
        let mut c = Client::connect(&addr).await.unwrap();
        let r = c.post(&format!("/v1/systemone?debug=1&debias={debias}"), body).await.unwrap();
        (r.status.as_u16(), r.body.to_vec())
    };
    if concurrent {
        let tasks: Vec<_> = cases.iter().map(|(_, b, k)| tokio::spawn(one(addr.to_string(), b.clone(), *k))).collect();
        let mut out = Vec::new();
        for t in tasks {
            out.push(t.await.unwrap());
        }
        out
    } else {
        let mut out = Vec::new();
        for (_, b, k) in cases {
            out.push(one(addr.to_string(), b.clone(), *k).await);
        }
        out
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn corpus_over_http_matches_qev_bf16() {
    let cases: Vec<(String, String, i64)> =
        goldens::model_corpus(&repo()).unwrap().into_iter().map(|c| (c.id, c.body, c.debias)).collect();
    // Golden bodies with key order preserved.
    let runs = pyjson::parse(&std::fs::read(repo().join("tests/golden/dev/model/runs/bf16_batched.json")).unwrap()).unwrap();
    let results = runs.get("results").and_then(|r| r.as_array()).unwrap();
    let golden = |id: &str| -> (u16, &Json) {
        let r = results.iter().find(|r| r.get("id").and_then(|v| v.as_str()) == Some(id)).unwrap();
        (num(r.get("status").unwrap()).unwrap() as u16, r.get("response").unwrap())
    };

    let external = std::env::var("SYSONE_SERVER_ADDR").ok();
    let mut local: Option<(Arc<AppState>, tokio::sync::oneshot::Sender<()>, tokio::task::JoinHandle<()>)> = None;
    let addr = match &external {
        Some(a) => a.clone(),
        None => {
            let fe = Arc::new(Frontend::load(&model_dir(), Limits::default()).unwrap());
            let slot_ids = fe.slot_ids.clone();
            let factory: EngineFactory = Box::new(move || {
                let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(65536), gdn_backend: gdn_backend_from_env()?,
                                           ..Default::default() };
                let e = GpuEngine::load(&model_dir(), &slot_ids, opts)?;
                eprintln!("engine_server: GDN backend {}", e.gdn_backend().name());
                Ok(Box::new(e) as Box<dyn DecisionEngine>)
            });
            let state = AppState::start(fe, ServerInfo::new("elephant-1", "qwen3_5"), SchedConfig::default(), factory, 4)
                .unwrap();
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let (stop, rx) = tokio::sync::oneshot::channel::<()>();
            let st = state.clone();
            let task = tokio::spawn(async move {
                server::serve(listener, st, async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
            });
            local = Some((state, stop, task));
            addr
        }
    };
    println!("server at {addr} ({})", if external.is_some() { "external" } else { "in-process" });

    let t = Tolerances::load_default().unwrap();
    let lim = |k: &str| t.f64(&format!("e2e.bf16.{k}")).unwrap();
    let mut failed = Vec::new();
    for concurrent in [false, true] {
        let t0 = std::time::Instant::now();
        let replies = post_all(&addr, &cases, concurrent).await;
        let mode = if concurrent { "concurrent" } else { "sequential" };
        let mut d = Diffs::default();
        let mut statuses = (0, 0);
        for ((id, ..), (status, body)) in cases.iter().zip(&replies) {
            let (gs, gbody) = golden(id);
            if *status != gs {
                d.errors.push(format!("{id}: status {status} vs {gs}: {}", String::from_utf8_lossy(body)));
                continue;
            }
            statuses.0 += (gs == 200) as usize;
            statuses.1 += (gs != 200) as usize;
            let ours = pyjson::parse(body).unwrap();
            let before = d.errors.len();
            walk("", &ours, gbody, 1.0, &mut d);
            for e in d.errors.iter_mut().skip(before) {
                *e = format!("{id}: {e}");
            }
        }
        println!("{mode}: {} requests ({} ok, {} errors as in qev) in {:.1} s", cases.len(), statuses.0, statuses.1,
                 t0.elapsed().as_secs_f64());
        let dp = Stats::of(&d.dp).unwrap();
        println!("    probabilities vs qev bf16: {dp} (gate p99 {} max {})", lim("dp_p99"), lim("dp_max"));
        println!("    score/confidence: {}", Stats::of(&d.derived).map(|s| s.to_string()).unwrap_or_default());
        let dl = Stats::of(&d.dlogit).unwrap();
        println!("    debug slot_logits: {dl} (gate max {})", lim("dlogit_max"));
        for f in &d.choice_flips {
            println!("    choice flip {f}");
        }
        for e in d.errors.iter().take(20) {
            println!("    MISMATCH {e}");
        }
        if !d.errors.is_empty() {
            failed.push(format!("{mode}: {} JSON mismatches", d.errors.len()));
        }
        if dp.p99 > lim("dp_p99") || dp.max > lim("dp_max") {
            failed.push(format!("{mode}: probabilities p99 {:.3e} max {:.3e}", dp.p99, dp.max));
        }
        if dl.max > lim("dlogit_max") {
            failed.push(format!("{mode}: slot logits max {:.3e}", dl.max));
        }
        let m = lim("top1_margin");
        for f in &d.choice_flips {
            let margin: f64 = f.rsplit("margin ").next().unwrap().trim_end_matches(')').parse().unwrap();
            if margin > m {
                failed.push(format!("{mode}: {f}"));
            }
        }
    }
    if let Some((_state, stop, task)) = local {
        let _ = stop.send(());
        task.await.unwrap();
    }
    assert!(failed.is_empty(), "server corpus gate failures:\n  {}", failed.join("\n  "));
}
