//! End-to-end HTTP tests of the server on `MockEngine` (DESIGN.md §7, W5): real sockets, the real frontend, the
//! scheduler and a mock engine. Covers qev's routes and error bodies (replayed from the reference goldens),
//! batch invariance under concurrency, backpressure, oversize requests and the token budget, and the
//! `/v1/systemone:batch` extension. Also the S1 fixes from the W5 review: qev's timing semantics, request ids,
//! 500s without qev's headers, slash redirects, backpressure before admission (CPU queue, body memory, departed
//! clients), oversized `:batch` calls, core pinning, and `debug=1` diagnostics byte-identical to qev's (replayed
//! from the assemble goldens through HTTP).
//!
//! Needs the elephant-1 tokenizer and chat template: `$SYSONE_MODEL_DIR`, default
//! /home/velvet/Desktop/hf_models/elephant-1.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use sysone::api::client::Client;
use sysone::api::server::{self, AppState, ServerInfo, ServerOptions};
use sysone::contract::{DecisionBatch, DecisionEngine, MAX_SLOTS, MassReadout, MockEngine, SlotLogits};
use sysone::frontend::assemble::Calibration;
use sysone::frontend::pyjson::{self, Json};
use sysone::frontend::{Frontend, Limits};
use sysone::sched::{EngineFactory, SchedConfig};
use std::io::{Read, Write};
use tokio::sync::oneshot;

const SERVED: &str = "elephant-1";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"))
}

fn frontend(limits: Limits) -> Arc<Frontend> {
    type Cache = Mutex<HashMap<(usize, usize), Arc<Frontend>>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let model = model_dir();
    assert!(model.join("tokenizer.json").exists(), "no checkpoint at {} (set SYSONE_MODEL_DIR)", model.display());
    let mut c = CACHE.get_or_init(Default::default).lock().unwrap();
    c.entry((limits.max_branch_tokens, limits.max_request_tokens))
        .or_insert_with(|| Arc::new(Frontend::load(&model, limits).unwrap()))
        .clone()
}

/// A batch an engine ran: (tokens, requests).
type Log = Arc<Mutex<Vec<(usize, usize)>>>;

/// MockEngine that records each batch and can be held closed by a gate.
struct TestEngine {
    inner: MockEngine,
    log: Log,
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl DecisionEngine for TestEngine {
    fn max_tokens_per_step(&self) -> usize {
        self.inner.max_tokens_per_step()
    }

    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
        let (open, cv) = &*self.gate;
        let _g = cv.wait_while(open.lock().unwrap(), |open| !*open).unwrap();
        let requests = batch.segments.iter().filter(|s| s.is_prefix()).count();
        self.log.lock().unwrap().push((batch.num_tokens(), requests));
        self.inner.run(batch)
    }

    fn run_with_mass(&mut self, batch: &DecisionBatch) -> Result<MassReadout> {
        let logits = self.run(batch)?;
        Ok(MassReadout { logits, allowed_mass: self.inner.run_with_mass(batch)?.allowed_mass })
    }
}

struct TestServer {
    addr: String,
    state: Arc<AppState>,
    log: Log,
    gate: Arc<(Mutex<bool>, Condvar)>,
    stop: Option<oneshot::Sender<()>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl TestServer {
    async fn start(sched: SchedConfig, limits: Limits, fork_mode: &str) -> Self {
        let opts = ServerOptions { cpu_workers: 4, ..Default::default() };
        Self::start_with(sched, frontend(limits), fork_mode, opts, None).await
    }

    /// A server on `frontend` with `opts`, running `engine` (default: the gated, recording [`TestEngine`]).
    async fn start_with(
        sched: SchedConfig,
        frontend: Arc<Frontend>,
        fork_mode: &str,
        opts: ServerOptions,
        engine: Option<EngineFactory>,
    ) -> Self {
        let log: Log = Default::default();
        let gate = Arc::new((Mutex::new(true), Condvar::new()));
        let (l, g) = (log.clone(), gate.clone());
        let max_tokens = sched.token_budget.max(frontend.limits.max_request_tokens);
        let engine: EngineFactory = engine.unwrap_or_else(|| {
            Box::new(move || {
                Ok(Box::new(TestEngine { inner: MockEngine { max_tokens }, log: l, gate: g }) as Box<dyn DecisionEngine>)
            })
        });
        let mut info = ServerInfo::new(SERVED, "qwen3_5");
        info.fork_mode = fork_mode.into();
        let state = AppState::start_with(frontend, info, sched, engine, opts).unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let (stop, rx) = oneshot::channel::<()>();
        let st = state.clone();
        let task = tokio::spawn(async move {
            server::serve(listener, st, async {
                let _ = rx.await;
            })
            .await
            .unwrap();
        });
        TestServer { addr, state, log, gate, stop: Some(stop), task: Some(task) }
    }

    async fn default() -> Self {
        Self::start(SchedConfig::default(), Limits::default(), "batched").await
    }

    async fn client(&self) -> Client {
        Client::connect(&self.addr).await.unwrap()
    }

    fn set_gate(&self, open: bool) {
        let (m, cv) = &*self.gate;
        *m.lock().unwrap() = open;
        cv.notify_all();
    }

    fn batches(&self) -> Vec<(usize, usize)> {
        self.log.lock().unwrap().clone()
    }

    async fn stop(mut self) {
        self.set_gate(true);
        let _ = self.stop.take().unwrap().send(());
        self.task.take().unwrap().await.unwrap();
    }
}

impl Drop for TestServer {
    /// A failing test must not leave the engine gated shut: the scheduler joins its GPU thread on drop.
    fn drop(&mut self) {
        self.set_gate(true);
    }
}

fn parse(b: &[u8]) -> Json {
    pyjson::parse(b).unwrap_or_else(|e| panic!("not JSON ({e:?}): {}", String::from_utf8_lossy(b)))
}

/// A response body without `qev.timing` (the only part that varies between identical requests).
fn untimed(body: &[u8]) -> Json {
    let mut v = parse(body);
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

/// A synthetic request: a state of roughly `words` words, and questions of every kind.
fn request(i: usize, words: usize) -> String {
    const VOCAB: [&str; 12] =
        ["order", "refund", "late", "billing", "error", "customer", "angry", "shipment", "account", "card", "retry", "ok"];
    let mut x = (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    let text: Vec<&str> = (0..words)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            VOCAB[(x % VOCAB.len() as u64) as usize]
        })
        .collect();
    let options = 2 + i % 5;
    let criteria: Vec<String> = (0..options).map(|k| format!("\"opt{k}\": \"option number {k} for case {i}\"")).collect();
    format!(
        r#"{{"state": {{"ticket": "{}", "case": {i}}}, "questions": {{"escalate": {{"type": "noul", "instructions": "Escalate case {i}?"}}, "route": {{"type": "choice", "instructions": "Route it", "criteria": {{{}}}}}, "severity": {{"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high", "critical"]}}}}}}"#,
        text.join(" "),
        criteria.join(", ")
    )
}

fn lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path).unwrap().split('\n').filter(|l| !l.is_empty()).map(str::to_string).collect()
}

fn int(v: Option<&Json>) -> Option<u64> {
    match v {
        Some(Json::Int(s)) => s.parse().ok(),
        _ => None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn routes_and_headers_match_qev() {
    let srv = TestServer::default().await;
    let mut c = srv.client().await;

    let r = c.get("/health").await.unwrap();
    assert_eq!(r.status, 200);
    assert_eq!(r.text(), r#"{"status":"ok","model":"elephant-1","fork_mode":"batched","prompt_version":"qev-labels-v2"}"#);
    assert_eq!(r.headers["content-type"], "application/json");
    let rid = r.headers["x-typesafe-request-id"].to_str().unwrap();
    assert!(rid.len() == 20 && rid.starts_with("qev-") && rid[4..].chars().all(|c| c.is_ascii_hexdigit()), "{rid}");
    assert!(r.headers["x-qev-server-time-ms"].to_str().unwrap().parse::<f64>().is_ok());

    let r = c.get("/v1/models").await.unwrap();
    assert_eq!(
        r.text(),
        concat!(
            r#"{"models":[{"name":"elephant-1","description":"Qev decision model (qwen3_5)","release_date":"2026-09-18"},"#,
            r#"{"name":"jev-latest","description":"alias for elephant-1","release_date":"2026-09-18"},"#,
            r#"{"name":"jev-preview","description":"alias for elephant-1","release_date":"2026-09-18"},"#,
            r#"{"name":"qev-latest","description":"alias for elephant-1","release_date":"2026-09-18"}]}"#
        )
    );

    let r = c.get("/nope").await.unwrap();
    assert_eq!((r.status.as_u16(), r.text().as_str()), (404, r#"{"detail":"Not Found"}"#));
    let r = c.get("/v1/systemone").await.unwrap();
    assert_eq!((r.status.as_u16(), r.text().as_str()), (405, r#"{"detail":"Method Not Allowed"}"#));

    // The request id is echoed.
    let req = hyper::Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("host", &srv.addr)
        .header("x-typesafe-request-id", "abc-123")
        .body(http_body_util::Full::new(axum::body::Bytes::from(request(1, 20))))
        .unwrap();
    let stream = tokio::net::TcpStream::connect(&srv.addr).await.unwrap();
    let (mut send, conn) = hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(stream)).await.unwrap();
    tokio::spawn(conn);
    let resp = send.send_request(req).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.headers()["x-typesafe-request-id"], "abc-123");

    // A 200 body has qev's layout: model, answers in request order, usage, qev meta with timing.
    let r = c.post("/v1/systemone?debias=2", request(2, 30)).await.unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    let v = parse(&r.body);
    let keys: Vec<&str> = v.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["model", "answers", "usage", "qev"]);
    assert_eq!(v.get("model").unwrap().as_str(), Some(SERVED));
    let answers: Vec<&str> = v.get("answers").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(answers, ["escalate", "route", "severity"]);
    assert_eq!(int(v.get("usage").unwrap().get("output_tokens")), Some(3));
    let qev = v.get("qev").unwrap();
    let qkeys: Vec<&str> = qev.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(qkeys, ["prompt_version", "prompt_sha256", "fork_mode", "prefix_tokens", "branches", "timing"]);
    // debias=2: two shifts for each of the three questions.
    assert_eq!(int(qev.get("branches")), Some(6));
    // qev's three timings, then sysone's queue_s.
    let timing: Vec<&str> = qev.get("timing").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(timing, ["render_s", "forward_s", "total_s", "queue_s"]);

    // debug adds diagnostics (any non-empty value, as `bool(str)`); an empty value does not.
    let r = c.post("/v1/systemone?debug=0", request(2, 30)).await.unwrap();
    let d = parse(&r.body);
    let diag = d.get("qev").unwrap().get("diagnostics").expect("diagnostics with debug=0");
    assert!(diag.get("route").unwrap().get("slot_logits").is_some());
    // MockEngine reports masses, so the diagnostics carry qev's allowed_token_mass, first as in qev.
    let keys: Vec<&str> = diag.get("route").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["allowed_token_mass", "slot_logits", "temperature"]);
    let r = c.post("/v1/systemone?debug=", request(2, 30)).await.unwrap();
    assert!(parse(&r.body).get("qev").unwrap().get("diagnostics").is_none());

    // The model alias, and a bad debias: qev's `int()` raises outside its handler, so Starlette sends a 500.
    let r = c.post("/v1/systemone", request(3, 10).replacen('{', r#"{"model": "qev-latest", "#, 1)).await.unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    for bad in ["x", "", "1.5"] {
        let r = c.post(&format!("/v1/systemone?debias={bad}"), request(3, 10)).await.unwrap();
        assert_eq!((r.status.as_u16(), r.text().as_str()), (500, "Internal Server Error"), "debias={bad:?}");
        // Starlette answers unhandled errors outside qev's middleware: no request id, no server time.
        assert_eq!(r.headers["content-type"], "text/plain; charset=utf-8");
        assert!(r.headers.get("x-typesafe-request-id").is_none() && r.headers.get("x-qev-server-time-ms").is_none());
    }
    // ... but only after the model check, as in qev.
    let r = c.post("/v1/systemone?debias=x", request(3, 10).replacen('{', r#"{"model": "m", "#, 1)).await.unwrap();
    assert_eq!(r.status, 404);
    // Python int() accepts these.
    let r = c.post("/v1/systemone?debias=%20%2B0_3", request(3, 10)).await.unwrap();
    assert_eq!(int(parse(&r.body).get("qev").unwrap().get("branches")), Some(2 + 3 + 3));

    let r = c.get("/metrics").await.unwrap();
    let text = r.text();
    assert!(r.headers["content-type"].to_str().unwrap().starts_with("text/plain"));
    for name in [
        "sysone_queue_depth ",
        "sysone_batch_tokens_bucket",
        "sysone_step_seconds_bucket",
        "sysone_decisions_per_second ",
        "sysone_request_latency_seconds{quantile=\"0.5\"}",
        "sysone_request_latency_seconds{quantile=\"0.99\"}",
        "sysone_requests_total{endpoint=\"/v1/systemone\",status=\"500\"} 3",
    ] {
        assert!(text.contains(name), "metrics lack {name}:\n{text}");
    }
    srv.stop().await;
}

/// Every error case of the reference goldens through HTTP: status and body as qev sent them. 200 cases (a sample
/// of the render goldens plus the schema goldens) are checked for status, usage and the qev meta.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn error_bodies_match_qev_goldens() {
    let rdir = repo().join("tests/golden/dev/render");
    let prov = parse(&std::fs::read(rdir.join("provenance.json")).unwrap());
    let lim = prov.get("limits").unwrap();
    let limits = Limits {
        max_branch_tokens: int(lim.get("max_branch_tokens")).unwrap() as usize,
        max_request_tokens: int(lim.get("max_request_tokens")).unwrap() as usize,
    };
    // The goldens were made with qev's fork_mode "sequential".
    let srv = TestServer::start(SchedConfig::default(), limits, "sequential").await;
    let mut c = srv.client().await;
    let mut failures = Vec::new();
    let mut checked = HashMap::<u16, usize>::new();

    // Schema goldens: raw bodies (including invalid JSON, NaN, 1e400 and odd model names).
    for line in lines(&repo().join("tests/golden/dev/assemble/schema.jsonl")) {
        let rec = parse(line.as_bytes());
        let id = rec.get("id").unwrap().as_str().unwrap().to_string();
        let body = rec.get("body").unwrap().as_str().unwrap().to_string();
        let status = int(rec.get("status")).unwrap() as u16;
        let r = c.post("/v1/systemone", body).await.unwrap();
        *checked.entry(status).or_default() += 1;
        if r.status.as_u16() != status {
            failures.push(format!("{id}: status {} want {status}: {}", r.status, r.text()));
            continue;
        }
        let want = rec.get("response").unwrap();
        if !want.is_null() && parse(&r.body) != *want {
            failures.push(format!("{id}: body {} want {}", r.text(), want.dumps_compact().unwrap()));
        }
    }

    // Render goldens: every non-200 case, and every 7th 200 case (all of the limits group).
    let corpus: HashMap<String, Json> = lines(&repo().join("reference/corpus/requests.jsonl"))
        .into_iter()
        .map(|l| {
            let v = parse(l.as_bytes());
            (v.get("id").unwrap().as_str().unwrap().to_string(), v)
        })
        .collect();
    let mut n200 = 0;
    for (name, _) in prov.get("files").unwrap().as_object().unwrap() {
        for line in lines(&rdir.join(name)) {
            let rec = parse(line.as_bytes());
            let id = rec.get("id").unwrap().as_str().unwrap();
            let http = rec.get("http").unwrap();
            let status = int(http.get("status")).unwrap() as u16;
            if status == 200 {
                n200 += 1;
                if n200 % 7 != 0 && !id.starts_with("limits") {
                    continue;
                }
            }
            let case = &corpus[id];
            let debias = int(case.get("debias")).unwrap();
            let payload = case.get("body").unwrap().dumps_compact().unwrap();
            let r = c.post(&format!("/v1/systemone?debias={debias}"), payload).await.unwrap();
            *checked.entry(status).or_default() += 1;
            if r.status.as_u16() != status {
                failures.push(format!("{id}: status {} want {status}: {:.300}", r.status, r.text()));
                continue;
            }
            let mut got = untimed(&r.body);
            if let Json::Object(m) = &mut got {
                m.retain(|(k, _)| k != "answers");
            }
            let want = http.get("body").unwrap();
            if got != *want {
                failures.push(format!("{id}: body {:.400} want {:.400}", got.dumps_compact().unwrap(), want.dumps_compact().unwrap()));
            }
        }
    }
    eprintln!("checked by golden status: {checked:?}");
    assert!(checked.get(&422).copied().unwrap_or(0) >= 40 && checked.get(&404).copied().unwrap_or(0) >= 4);
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
    srv.stop().await;
}

/// The same requests sent one at a time and all at once: identical bodies (MockEngine is batch-invariant, so any
/// difference is a packing or row-mapping bug), and the concurrent run really packed several per batch.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_results_equal_alone() {
    let cfg = SchedConfig { token_budget: 16384, max_wait: Duration::from_millis(30), queue_capacity: 1024 };
    let srv = TestServer::start(cfg, Limits::default(), "batched").await;
    let n = 48;
    let bodies: Vec<(String, String)> = (0..n)
        .map(|i| (format!("/v1/systemone?debias={}", 1 + i % 3), request(i, 10 + (i * 37) % 400)))
        .collect();

    let mut c = srv.client().await;
    let mut alone = Vec::new();
    for (path, body) in &bodies {
        let r = c.post(path, body.clone()).await.unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
        alone.push(untimed(&r.body));
    }
    let sequential = srv.batches();
    assert_eq!(sequential.len(), n);
    assert!(sequential.iter().all(|b| b.1 == 1));

    let tasks: Vec<_> = bodies
        .iter()
        .cloned()
        .map(|(path, body)| {
            let addr = srv.addr.clone();
            tokio::spawn(async move {
                let mut c = Client::connect(&addr).await.unwrap();
                c.post(&path, body).await.unwrap()
            })
        })
        .collect();
    for (i, t) in tasks.into_iter().enumerate() {
        let r = t.await.unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
        assert_eq!(untimed(&r.body), alone[i], "request {i} differs when packed");
    }
    let packed = &srv.batches()[n..];
    let requests: usize = packed.iter().map(|b| b.1).sum();
    assert_eq!(requests, n);
    assert!(packed.len() < n, "no packing happened: {packed:?}");
    assert!(packed.iter().all(|b| b.0 <= 16384), "{packed:?}");
    eprintln!("{n} concurrent requests ran in {} batches: {packed:?}", packed.len());

    // The batch extension gives the same bodies too.
    let arr = format!("[{}]", bodies.iter().filter(|(p, _)| p.ends_with("debias=1")).map(|(_, b)| b.as_str()).collect::<Vec<_>>().join(","));
    let r = c.post("/v1/systemone:batch?debias=1", arr).await.unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    let v = parse(&r.body);
    let results = v.get("results").unwrap().as_array().unwrap();
    let want: Vec<&Json> = alone.iter().enumerate().filter(|(i, _)| bodies[*i].0.ends_with("debias=1")).map(|(_, a)| a).collect();
    assert_eq!(results.len(), want.len());
    for (res, w) in results.iter().zip(want) {
        assert_eq!(int(res.get("status")), Some(200));
        let body = res.get("body").unwrap().dumps_compact().unwrap();
        assert_eq!(untimed(body.as_bytes()), *w);
    }
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn full_queue_answers_503_then_recovers() {
    // A small budget, so that while the engine is held the batcher fills a batch, parks it in the slot, fills the
    // next one, blocks handing it over, and the queue fills behind it.
    let cfg = SchedConfig { token_budget: 600, max_wait: Duration::ZERO, queue_capacity: 2 };
    let srv = TestServer::start(cfg, Limits::default(), "batched").await;
    srv.set_gate(false);
    let mut pending = Vec::new();
    for i in 0..40 {
        if srv.state.sched.queue_depth() == 2 {
            break;
        }
        let addr = srv.addr.clone();
        pending.push(tokio::spawn(async move { Client::connect(&addr).await.unwrap().post("/v1/systemone", request(i, 20)).await.unwrap() }));
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert_eq!(srv.state.sched.queue_depth(), 2, "the queue never filled");
    let mut c = srv.client().await;
    let r = c.post("/v1/systemone", request(99, 20)).await.unwrap();
    assert_eq!((r.status.as_u16(), r.text().as_str()), (503, r#"{"detail":"admission queue is full"}"#));
    // A batch call that does not fit is refused whole.
    let r = c.post("/v1/systemone:batch", format!("[{}]", request(98, 5))).await.unwrap();
    assert_eq!(r.status, 503);
    let metrics = c.get("/metrics").await.unwrap().text();
    assert!(metrics.contains("sysone_queue_depth 2\n"), "{metrics}");
    assert!(metrics.contains("sysone_rejected_total 2\n"), "{metrics}");

    srv.set_gate(true);
    for t in pending {
        let r = t.await.unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
    }
    let r = c.post("/v1/systemone", request(99, 20)).await.unwrap();
    assert_eq!(r.status, 200);
    srv.stop().await;
}

/// Requests over the token budget run alone and succeed; batches with several requests stay within the budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn oversize_requests_run_alone_and_budget_holds() {
    // Size the budget from real plans: small requests are ~300-500 tokens, the big ones ~3-4k.
    let fe = frontend(Limits::default());
    let tokens = |body: &str| {
        let req = sysone::api::schema::SystemOneRequest::from_body(body.as_bytes()).unwrap();
        fe.plan(&req, 1).unwrap().total
    };
    let small: Vec<String> = (0..30).map(|i| request(i, 5 + i * 3)).collect();
    let big: Vec<String> = (0..3).map(|i| request(100 + i, 3000)).collect();
    let max_small = small.iter().map(|b| tokens(b)).max().unwrap();
    let min_big = big.iter().map(|b| tokens(b)).min().unwrap();
    let budget = 3 * max_small;
    assert!(min_big > budget, "big {min_big} vs budget {budget}");

    let cfg = SchedConfig { token_budget: budget, max_wait: Duration::from_millis(20), queue_capacity: 1024 };
    let srv = TestServer::start(cfg, Limits::default(), "batched").await;
    let mut c = srv.client().await;
    let mut alone = HashMap::new();
    for b in small.iter().chain(&big) {
        let r = c.post("/v1/systemone", b.clone()).await.unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
        alone.insert(b.clone(), untimed(&r.body));
    }
    let before = srv.batches().len();
    // Interleave the big requests with the small ones.
    let mut order: Vec<&String> = small.iter().collect();
    for (k, b) in big.iter().enumerate() {
        order.insert(5 + k * 9, b);
    }
    let tasks: Vec<_> = order
        .iter()
        .map(|b| {
            let (addr, b) = (srv.addr.clone(), (*b).clone());
            tokio::spawn(async move { (b.clone(), Client::connect(&addr).await.unwrap().post("/v1/systemone", b).await.unwrap()) })
        })
        .collect();
    for t in tasks {
        let (b, r) = t.await.unwrap();
        assert_eq!(r.status, 200, "{}", r.text());
        assert_eq!(untimed(&r.body), alone[&b]);
    }
    let batches = srv.batches()[before..].to_vec();
    eprintln!("budget {budget}: {batches:?}");
    for &(t, n) in &batches {
        assert!(t <= budget || n == 1, "a {t}-token batch over the {budget} budget holds {n} requests");
    }
    assert_eq!(batches.iter().filter(|b| b.0 > budget).count(), big.len());
    assert_eq!(batches.iter().map(|b| b.1).sum::<usize>(), order.len());
    assert!(batches.iter().any(|b| b.1 > 1), "nothing packed");
    srv.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batch_endpoint_mixes_outcomes() {
    let srv = TestServer::default().await;
    let mut c = srv.client().await;
    let ok = request(7, 40);
    let single = untimed(&c.post("/v1/systemone", ok.clone()).await.unwrap().body);
    let body = format!(r#"[{ok}, {{"questions": {{}}}}, {{"model": "other", "questions": {{"q": {{"type": "noul"}}}}}}, 3, {ok}]"#);
    let r = c.post("/v1/systemone:batch", body).await.unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    let v = parse(&r.body);
    let res = v.get("results").unwrap().as_array().unwrap();
    let statuses: Vec<u64> = res.iter().map(|x| int(x.get("status")).unwrap()).collect();
    assert_eq!(statuses, [200, 422, 404, 422, 200]);
    for i in [0, 4] {
        let b = res[i].get("body").unwrap().dumps_compact().unwrap();
        assert_eq!(untimed(b.as_bytes()), single);
    }
    assert_eq!(
        res[1].get("body").unwrap().dumps_compact().unwrap(),
        r#"{"detail":[{"loc":["body"],"msg":"Value error, at least one question is required","type":"value_error"}]}"#
    );
    assert_eq!(
        res[3].get("body").unwrap().dumps_compact().unwrap(),
        r#"{"detail":[{"loc":["body"],"msg":"Input should be a valid dictionary or instance of SystemOneRequest","type":"model_type"}]}"#
    );
    let r = c.post("/v1/systemone:batch", "{}").await.unwrap();
    assert_eq!(
        (r.status.as_u16(), r.text().as_str()),
        (422, r#"{"detail":[{"loc":["body"],"msg":"Input should be a valid list","type":"list_type"}]}"#)
    );
    let r = c.post("/v1/systemone:batch", "[").await.unwrap();
    assert_eq!(r.status, 422);
    let r = c.post("/v1/systemone:batch", "[]").await.unwrap();
    assert_eq!(r.text(), r#"{"results":[]}"#);
    srv.stop().await;
}

/// Limit errors come from the frontend with qev's text, before admission.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn limits_answer_422() {
    let body = request(1, 120);
    let req = sysone::api::schema::SystemOneRequest::from_body(body.as_bytes()).unwrap();
    let plan = frontend(Limits::default()).plan(&req, 3).unwrap();
    // The branch limit is checked first.
    let limits = Limits { max_branch_tokens: plan.longest - 1, max_request_tokens: plan.total - 1 };
    let srv = TestServer::start(SchedConfig::default(), limits, "batched").await;
    let mut c = srv.client().await;
    let r = c.post("/v1/systemone?debias=3", body.clone()).await.unwrap();
    assert_eq!(
        (r.status.as_u16(), r.text()),
        (
            422,
            format!(
                r#"{{"detail":[{{"loc":["body"],"msg":"state plus longest question is {} tokens, limit is {}","type":"value_error"}}]}}"#,
                plan.longest,
                plan.longest - 1
            )
        )
    );
    srv.stop().await;
    let limits = Limits { max_branch_tokens: plan.longest, max_request_tokens: plan.total - 1 };
    let srv = TestServer::start(SchedConfig::default(), limits, "batched").await;
    let mut c = srv.client().await;
    let r = c.post("/v1/systemone?debias=3", body.clone()).await.unwrap();
    assert_eq!(
        (r.status.as_u16(), r.text()),
        (
            422,
            format!(
                r#"{{"detail":[{{"loc":["body"],"msg":"request is {} tokens, limit is {}","type":"value_error"}}]}}"#,
                plan.total,
                plan.total - 1
            )
        )
    );
    // At the limit it passes.
    let r = c.post("/v1/systemone?debias=1", body).await.unwrap();
    assert_eq!(r.status, 200);
    assert_eq!(srv.batches().len(), 1, "only the accepted request reaches the engine");
    srv.stop().await;
}

async fn send_with_headers(addr: &str, method: &str, path: &str, headers: &[(&str, &str)], body: String) -> hyper::Response<hyper::body::Incoming> {
    let mut req = hyper::Request::builder().method(method).uri(path).header("host", addr);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let req = req.body(http_body_util::Full::new(axum::body::Bytes::from(body))).unwrap();
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut send, conn) = hyper::client::conn::http1::handshake(hyper_util::rt::TokioIo::new(stream)).await.unwrap();
    tokio::spawn(conn);
    send.send_request(req).await.unwrap()
}

fn is_minted_id(rid: &str) -> bool {
    rid.len() == 20 && rid.starts_with("qev-") && rid[4..].chars().all(|c| c.is_ascii_hexdigit()) && &rid[16..17] == "4"
}

/// An empty `x-typesafe-request-id` is replaced (qev: `headers.get(...) or f"qev-{uuid}"`), a 307 for a trailing
/// slash, and the ids of 4xx/2xx responses.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn request_ids_and_slash_redirects_match_qev() {
    let srv = TestServer::default().await;
    let r = send_with_headers(&srv.addr, "POST", "/v1/systemone", &[("x-typesafe-request-id", "")], request(1, 10)).await;
    assert_eq!(r.status(), 200);
    let rid = r.headers()["x-typesafe-request-id"].to_str().unwrap();
    assert!(is_minted_id(rid), "{rid:?}");
    let r = send_with_headers(&srv.addr, "GET", "/nope", &[("x-typesafe-request-id", "keep-me")], String::new()).await;
    assert_eq!((r.status().as_u16(), &r.headers()["x-typesafe-request-id"]), (404, &axum::http::HeaderValue::from_static("keep-me")));

    let mut c = srv.client().await;
    let r = c.get("/health/").await.unwrap();
    assert_eq!(r.status, 307);
    assert_eq!(r.headers["location"], format!("http://{}/health", srv.addr).as_str());
    assert!(r.body.is_empty() && r.headers.get("content-type").is_none());
    assert!(is_minted_id(r.headers["x-typesafe-request-id"].to_str().unwrap()));
    // Any method, the query kept, several slashes.
    let r = c.post("/v1/systemone//?debias=2", request(1, 10)).await.unwrap();
    assert_eq!(r.status, 307);
    assert_eq!(r.headers["location"], format!("http://{}/v1/systemone?debias=2", srv.addr).as_str());
    // A path that matches nothing either way stays a 404.
    let r = c.get("/nope/").await.unwrap();
    assert_eq!((r.status.as_u16(), r.text().as_str()), (404, r#"{"detail":"Not Found"}"#));
    srv.stop().await;
}

fn timing(body: &[u8]) -> (f64, f64, f64, f64) {
    let v = parse(body);
    let t = v.get("qev").unwrap().get("timing").unwrap();
    let f = |k: &str| match t.get(k).unwrap() {
        Json::Float(x) => *x,
        other => panic!("{k}: {other:?}"),
    };
    (f("render_s"), f("forward_s"), f("total_s"), f("queue_s"))
}

/// `forward_s` is the engine step only (qev's `t2 - t1`); time spent waiting for the engine is `queue_s`.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn timing_separates_queue_from_forward() {
    let srv = TestServer::default().await;
    let hold = Duration::from_millis(300);
    srv.set_gate(false);
    // A enters the engine and is held there; B waits behind it in the scheduler.
    let addr = srv.addr.clone();
    let a = tokio::spawn(async move { Client::connect(&addr).await.unwrap().post("/v1/systemone", request(1, 20)).await.unwrap() });
    tokio::time::sleep(Duration::from_millis(50)).await;
    let addr = srv.addr.clone();
    let b = tokio::spawn(async move { Client::connect(&addr).await.unwrap().post("/v1/systemone", request(2, 20)).await.unwrap() });
    tokio::time::sleep(hold).await;
    srv.set_gate(true);
    let (a, b) = (a.await.unwrap(), b.await.unwrap());
    assert_eq!((a.status.as_u16(), b.status.as_u16()), (200, 200));
    let (ra, fa, ta, qa) = timing(&a.body);
    let (rb, fb, tb, qb) = timing(&b.body);
    eprintln!("A: render {ra:.4} queue {qa:.4} forward {fa:.4} total {ta:.4}; B: render {rb:.4} queue {qb:.4} forward {fb:.4} total {tb:.4}");
    // A's forward holds the gate; B's queue does.
    assert!(fa >= 0.3 && qa < 0.1, "A");
    assert!(qb >= 0.2 && fb < 0.1, "B");
    for (r, f, t, q) in [(ra, fa, ta, qa), (rb, fb, tb, qb)] {
        assert!(r + q + f <= t + 1e-6 && t < r + q + f + 0.1, "parts {r} + {q} + {f} vs total {t}");
    }
    srv.stop().await;
}

/// A full CPU new-work queue answers 503 before planning; finishing work still gets through.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn full_cpu_queue_answers_503() {
    let opts = ServerOptions { cpu_workers: 1, cpu_queue_capacity: Some(1), ..Default::default() };
    let srv = TestServer::start_with(SchedConfig::default(), frontend(Limits::default()), "batched", opts, None).await;
    let (go, wait) = std::sync::mpsc::channel::<()>();
    let st = srv.state.clone();
    let blocker = tokio::spawn(async move { st.pool.run(move || wait.recv().unwrap()).await });
    while srv.state.pool.queue_depth() > 0 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    let addr = srv.addr.clone();
    let queued = tokio::spawn(async move { Client::connect(&addr).await.unwrap().post("/v1/systemone", request(1, 10)).await.unwrap() });
    let t0 = Instant::now();
    while srv.state.pool.queue_depth() < 1 {
        assert!(t0.elapsed() < Duration::from_secs(5), "the request never queued");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let mut c = srv.client().await;
    let r = c.post("/v1/systemone", request(2, 10)).await.unwrap();
    assert_eq!((r.status.as_u16(), r.text().as_str()), (503, r#"{"detail":"admission queue is full"}"#));
    assert!(is_minted_id(r.headers["x-typesafe-request-id"].to_str().unwrap()));
    let r = c.post("/v1/systemone:batch", format!("[{}]", request(3, 10))).await.unwrap();
    assert_eq!(r.status, 503);
    go.send(()).unwrap();
    blocker.await.unwrap().unwrap();
    assert_eq!(queued.await.unwrap().status, 200);
    let m = c.get("/metrics").await.unwrap().text();
    assert!(m.contains("sysone_overload_total{reason=\"cpu_queue\"} 2\n"), "{m}");
    assert!(m.contains("sysone_cpu_queue_capacity 1\n"), "{m}");
    srv.stop().await;
}

/// Request bodies in flight are capped (503), and a declared body over the per-request limit is refused (413)
/// before it is read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn body_memory_is_capped() {
    let small = request(1, 10);
    let big = request(2, 2000);
    let opts = ServerOptions { cpu_workers: 2, max_inflight_body_bytes: small.len() + 100, ..Default::default() };
    assert!(big.len() > opts.max_inflight_body_bytes);
    let srv = TestServer::start_with(SchedConfig::default(), frontend(Limits::default()), "batched", opts, None).await;
    let mut c = srv.client().await;
    assert_eq!(c.post("/v1/systemone", small.clone()).await.unwrap().status, 200);
    let r = c.post("/v1/systemone", big).await.unwrap();
    assert_eq!((r.status.as_u16(), r.text().as_str()), (503, r#"{"detail":"admission queue is full"}"#));
    // The lease was returned: the small one still fits, again and again.
    for _ in 0..3 {
        assert_eq!(c.post("/v1/systemone", small.clone()).await.unwrap().status, 200);
    }
    let m = c.get("/metrics").await.unwrap().text();
    assert!(m.contains("sysone_overload_total{reason=\"body_memory\"} 1\n"), "{m}");
    assert!(m.contains("sysone_inflight_body_bytes 0\n"), "{m}");

    // A raw socket (std, blocking: a few bytes each way).
    let mut s = std::net::TcpStream::connect(&srv.addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.write_all(b"POST /v1/systemone HTTP/1.1\r\nhost: x\r\ncontent-length: 300000000\r\n\r\n{").unwrap();
    let mut buf = vec![0u8; 4096];
    let n = s.read(&mut buf).unwrap();
    let head = String::from_utf8_lossy(&buf[..n]);
    assert!(head.starts_with("HTTP/1.1 413"), "{head}");
    assert!(head.ends_with("Failed to buffer the request body: length limit exceeded"), "{head}");
    srv.stop().await;
}

/// A request whose client disconnects while it waits for a CPU thread is never planned nor run.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn departed_clients_are_not_planned() {
    let opts = ServerOptions { cpu_workers: 1, ..Default::default() };
    let srv = TestServer::start_with(SchedConfig::default(), frontend(Limits::default()), "batched", opts, None).await;
    let (go, wait) = std::sync::mpsc::channel::<()>();
    let st = srv.state.clone();
    let blocker = tokio::spawn(async move { st.pool.run(move || wait.recv().unwrap()).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    // Write a whole request on a raw socket, wait until it is queued, and hang up.
    let body = request(1, 10);
    let mut s = std::net::TcpStream::connect(&srv.addr).unwrap();
    let head = format!("POST /v1/systemone HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n", body.len());
    s.write_all(head.as_bytes()).unwrap();
    s.write_all(body.as_bytes()).unwrap();
    let t0 = Instant::now();
    while srv.state.pool.queue_depth() < 1 {
        assert!(t0.elapsed() < Duration::from_secs(5), "the request never queued");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    drop(s);
    // hyper notices the closed connection and drops the handler, which drops the pool task's receiver.
    tokio::time::sleep(Duration::from_millis(100)).await;
    go.send(()).unwrap();
    blocker.await.unwrap().unwrap();
    let mut c = srv.client().await;
    let t0 = Instant::now();
    loop {
        let m = c.get("/metrics").await.unwrap().text();
        if m.contains("sysone_skipped_total 1\n") {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(5), "the departed request was planned:\n{m}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(srv.batches().is_empty(), "{:?}", srv.batches());
    srv.stop().await;
}

/// A `:batch` call larger than the admission queue could never be admitted: a 413 with a clear message, decided
/// before anything is planned.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batch_over_queue_capacity_is_413() {
    let cfg = SchedConfig { queue_capacity: 3, ..SchedConfig::default() };
    let srv = TestServer::start(cfg, Limits::default(), "batched").await;
    let mut c = srv.client().await;
    let items = |n: usize| format!("[{}]", (0..n).map(|i| request(i, 5)).collect::<Vec<_>>().join(","));
    let r = c.post("/v1/systemone:batch", items(4)).await.unwrap();
    assert_eq!(
        (r.status.as_u16(), r.text().as_str()),
        (413, r#"{"detail":"batch of 4 requests exceeds the admission queue capacity of 3; send at most 3 per call"}"#)
    );
    assert!(srv.batches().is_empty());
    let r = c.post("/v1/systemone:batch", items(3)).await.unwrap();
    assert_eq!(r.status, 200);
    srv.stop().await;
}

/// With the threads pinned to the performance cores (where the machine has them) the server works as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinned_server_answers() {
    let pin = sysone::sched::affinity::performance_cores();
    eprintln!("performance cores: {pin:?}");
    let opts = ServerOptions { cpu_workers: 2, pin_cores: pin.clone(), ..Default::default() };
    let srv = TestServer::start_with(SchedConfig::default(), frontend(Limits::default()), "batched", opts, None).await;
    let mut c = srv.client().await;
    assert_eq!(c.post("/v1/systemone", request(1, 10)).await.unwrap().status, 200);
    if let Some(cores) = pin {
        let got = srv.state.pool.run(sysone::sched::affinity::allowed_cores).await.unwrap();
        assert_eq!(got, Some(cores));
    }
    srv.stop().await;
}

/// Replays fixed slot logits and masses, one request per step, in order.
type Feed = Arc<Mutex<VecDeque<(Vec<Vec<f32>>, Option<Vec<f64>>)>>>;

struct Replay {
    feed: Feed,
}

impl DecisionEngine for Replay {
    fn max_tokens_per_step(&self) -> usize {
        1 << 17
    }

    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
        Ok(self.run_with_mass(batch)?.logits)
    }

    fn run_with_mass(&mut self, batch: &DecisionBatch) -> Result<MassReadout> {
        let (rows, masses) = self.feed.lock().unwrap().pop_front().expect("no logits queued");
        anyhow::ensure!(rows.len() == batch.num_branches(), "{} rows for {} branches", rows.len(), batch.num_branches());
        let mut data = vec![0.0f32; rows.len() * MAX_SLOTS];
        for (i, r) in rows.iter().enumerate() {
            data[i * MAX_SLOTS..i * MAX_SLOTS + r.len()].copy_from_slice(r);
        }
        Ok(MassReadout { logits: SlotLogits { rows: rows.len(), data }, allowed_mass: masses })
    }
}

fn f64_of(v: &Json) -> f64 {
    match v {
        Json::Float(f) => *f,
        Json::Int(s) => s.parse().unwrap(),
        _ => panic!("not a number: {v:?}"),
    }
}

/// `?debug=1` through HTTP, against qev's own `decide()` on the same slot logits and masses
/// (tests/golden/dev/assemble/answers.jsonl): the answers and the diagnostics, including `allowed_token_mass`
/// when the engine reports masses, are byte-identical to Starlette's JSON of qev's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn debug_diagnostics_are_byte_identical_to_qev() {
    let corpus: HashMap<String, Json> = lines(&repo().join("reference/corpus/requests.jsonl"))
        .into_iter()
        .map(|l| {
            let v = parse(l.as_bytes());
            (v.get("id").unwrap().as_str().unwrap().to_string(), v)
        })
        .collect();
    // One server per (max_slots, calibration): the frontend holds both.
    let mut groups: Vec<((usize, String), Vec<Json>)> = Vec::new();
    for line in lines(&repo().join("tests/golden/dev/assemble/answers.jsonl")) {
        let r = parse(line.as_bytes());
        if r.get("engine_direct") == Some(&Json::Bool(true)) || r.get("error").is_some() {
            continue;
        }
        let key = (f64_of(r.get("max_slots").unwrap()) as usize, r.get("calibration").unwrap().dumps_compact().unwrap());
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(r),
            None => groups.push((key, vec![r])),
        }
    }
    let (mut with_mass, mut without) = (0, 0);
    let mut failures = Vec::new();
    for ((max_slots, _), recs) in &groups {
        let mut fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
        fe.max_slots = *max_slots;
        let mut cal = Calibration::none();
        for (k, v) in recs[0].get("calibration").unwrap().as_object().unwrap() {
            cal.temperatures.insert(k.clone(), f64_of(v));
        }
        fe.calibration = cal;
        let feed: Feed = Default::default();
        let f = feed.clone();
        let engine: EngineFactory = Box::new(move || Ok(Box::new(Replay { feed: f }) as Box<dyn DecisionEngine>));
        let opts = ServerOptions { cpu_workers: 2, ..Default::default() };
        let srv = TestServer::start_with(SchedConfig::default(), Arc::new(fe), "batched", opts, Some(engine)).await;
        let mut c = srv.client().await;
        for r in recs {
            let id = r.get("id").unwrap().as_str().unwrap();
            let case = &corpus[r.get("case").unwrap().as_str().unwrap()];
            let logits: Vec<Vec<f32>> = r
                .get("logits")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row.as_array().unwrap().iter().map(|x| f64_of(x) as f32).collect())
                .collect();
            let masses: Option<Vec<f64>> = match r.get("masses").unwrap() {
                Json::Null => None,
                m => Some(m.as_array().unwrap().iter().map(f64_of).collect()),
            };
            if masses.is_some() {
                with_mass += 1;
            } else {
                without += 1;
            }
            feed.lock().unwrap().push_back((logits, masses));
            let debias = f64_of(r.get("debias").unwrap()) as i64;
            let body = case.get("body").unwrap().dumps_compact().unwrap();
            let resp = c.post(&format!("/v1/systemone?debias={debias}&debug=1"), body).await.unwrap();
            if resp.status != 200 {
                failures.push(format!("{id}: status {}: {:.300}", resp.status, resp.text()));
                continue;
            }
            let text = resp.text();
            let answers = format!("\"answers\":{},\"usage\":", r.get("answers_json").unwrap().as_str().unwrap());
            let diag = format!(",\"diagnostics\":{}}}}}", r.get("diagnostics_json").unwrap().as_str().unwrap());
            if !text.contains(&answers) {
                failures.push(format!("{id}: answers differ: {text:.600}"));
            }
            if !text.ends_with(&diag) {
                failures.push(format!("{id}: diagnostics differ:\n got {text:.2000}\nwant ...{diag:.2000}"));
            }
            let usage = parse(&resp.body).get("usage").cloned();
            if usage.as_ref() != r.get("usage") {
                failures.push(format!("{id}: usage {usage:?}"));
            }
        }
        srv.stop().await;
    }
    eprintln!("{with_mass} cases with masses, {without} without, in {} server configurations", groups.len());
    assert!(failures.is_empty(), "{} mismatches:\n{}", failures.len(), failures.join("\n"));
    assert!(with_mass >= 150 && without >= 150, "{with_mass} / {without}");
}
