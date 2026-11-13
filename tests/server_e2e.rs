//! End-to-end HTTP tests of the server on `MockEngine` (DESIGN.md §7, W5): real sockets, the real frontend, the
//! scheduler and a mock engine. Covers qev's routes and error bodies (replayed from the reference goldens),
//! batch invariance under concurrency, backpressure, oversize requests and the token budget, and the
//! `/v1/systemone:batch` extension.
//!
//! Needs the elephant-1 tokenizer and chat template: `$SYSONE_MODEL_DIR`, default
//! /home/velvet/Desktop/hf_models/elephant-1.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use anyhow::Result;
use sysone::api::client::Client;
use sysone::api::server::{self, AppState, ServerInfo};
use sysone::contract::{DecisionBatch, DecisionEngine, MockEngine, SlotLogits};
use sysone::frontend::pyjson::{self, Json};
use sysone::frontend::{Frontend, Limits};
use sysone::sched::{EngineFactory, SchedConfig};
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
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize), Arc<Frontend>>>> = OnceLock::new();
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
        let log: Log = Default::default();
        let gate = Arc::new((Mutex::new(true), Condvar::new()));
        let (l, g) = (log.clone(), gate.clone());
        let max_tokens = sched.token_budget.max(limits.max_request_tokens);
        let engine: EngineFactory = Box::new(move || {
            Ok(Box::new(TestEngine { inner: MockEngine { max_tokens }, log: l, gate: g }) as Box<dyn DecisionEngine>)
        });
        let mut info = ServerInfo::new(SERVED, "qwen3_5");
        info.fork_mode = fork_mode.into();
        let state = AppState::start(frontend(limits), info, sched, engine, 4).unwrap();
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
    let timing: Vec<&str> = qev.get("timing").unwrap().as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(timing, ["render_s", "forward_s", "total_s"]);

    // debug adds diagnostics (any non-empty value, as `bool(str)`); an empty value does not.
    let r = c.post("/v1/systemone?debug=0", request(2, 30)).await.unwrap();
    let d = parse(&r.body);
    let diag = d.get("qev").unwrap().get("diagnostics").expect("diagnostics with debug=0");
    assert!(diag.get("route").unwrap().get("slot_logits").is_some());
    let r = c.post("/v1/systemone?debug=", request(2, 30)).await.unwrap();
    assert!(parse(&r.body).get("qev").unwrap().get("diagnostics").is_none());

    // The model alias, and a bad debias: qev's `int()` raises outside its handler, so Starlette sends a 500.
    let r = c.post("/v1/systemone", request(3, 10).replacen('{', r#"{"model": "qev-latest", "#, 1)).await.unwrap();
    assert_eq!(r.status, 200, "{}", r.text());
    for bad in ["x", "", "1.5"] {
        let r = c.post(&format!("/v1/systemone?debias={bad}"), request(3, 10)).await.unwrap();
        assert_eq!((r.status.as_u16(), r.text().as_str()), (500, "Internal Server Error"), "debias={bad:?}");
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
    let cfg = SchedConfig { token_budget: 16384, max_wait: Duration::ZERO, queue_capacity: 2 };
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
