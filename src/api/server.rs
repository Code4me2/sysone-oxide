//! The HTTP server: qev's routes with its status codes, bodies, headers and query parameters
//! (`qev/server.py` at OneJev@81ce62f), plus `/metrics` and the additive `/v1/systemone:batch`.
//!
//! A `/v1/systemone` request goes: read the body (within the in-flight body budget) → CPU pool, new-work queue
//! (parse, validate, route the model, parse `debug`/`debias`, render, tokenize, plan, check limits) → admission
//! queue → batcher → GPU worker → CPU pool, finishing queue (answer assembly, response JSON).
//!
//! As in qev (Starlette):
//! - `x-typesafe-request-id` is echoed when present and non-empty, else minted as `qev-` + 16 hex digits of a
//!   random UUID4; `x-qev-server-time-ms` is the server time. Neither is set on a 500 for an unhandled error,
//!   because Starlette's `ServerErrorMiddleware` answers those outside qev's `http` middleware.
//! - A path that matches no route but would with its trailing slash toggled (`/health/`, `/v1/systemone/`) gets
//!   Starlette's `redirect_slashes` answer: 307 with an absolute `location`, an empty body, and no content type.
//! - `qev.timing` keeps qev's `render_s`, `forward_s` (the engine step only, as qev's `t2 - t1`) and `total_s`
//!   (wall time from planning to the response, which here includes waiting for a batch). It adds `queue_s`: from
//!   the end of planning to the start of the engine step. qev's meta is a plain dict that its client reads with
//!   `data.get("qev", {})`, so the extra key breaks nothing.
//!
//! Differences from qev that a client can see (qev has no queues and no limits on bodies):
//! - Overload answers 503 `{"detail": "admission queue is full"}`: when the admission queue is full, when the CPU
//!   pool's new-work queue is full ([`ServerOptions::cpu_queue_capacity`]), or when the request bodies in flight
//!   would exceed [`ServerOptions::max_inflight_body_bytes`]. `/metrics` counts each reason
//!   (`sysone_overload_total{reason=...}`).
//! - A body over [`MAX_BODY_BYTES`] gets 413 `Failed to buffer the request body: length limit exceeded`.
//! - A `/v1/systemone:batch` call with more items than the admission queue holds can never be admitted (calls are
//!   admitted whole), so it gets 413 `{"detail": "batch of N requests exceeds the admission queue capacity of M;
//!   send at most M per call"}` without planning anything.
//! - `qev.diagnostics.*.allowed_token_mass` (with `?debug=1`) is present only when the engine reports masses
//!   ([`crate::contract::DecisionEngine::run_with_mass`]); `MockEngine` does, an engine that does not leaves it
//!   out.
//! - A request whose client disconnects before its planning starts is not planned (`sysone_skipped_total`).

use std::hash::{BuildHasher, Hasher};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{RawQuery, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};
use http_body_util::BodyExt;

use crate::api::query::{self, py_int};
use crate::api::schema::{ALIASES, ApiError, ErrorItem, SystemOneRequest, check_model};
use crate::contract::DecisionBatch;
use crate::frontend::pyjson::{self, Json};
use crate::frontend::render::PROMPT_VERSION;
use crate::frontend::{Frontend, Plan, PlanError};
use crate::sched::{CpuPool, EngineFactory, Metrics, Packable, PoolError, SchedConfig, Scheduler, Scored};

/// What `/health` and `/v1/models` report, and request defaults.
#[derive(Clone, Debug)]
pub struct ServerInfo {
    /// The served model name (qev `--name`, default the checkpoint directory's name).
    pub served_name: String,
    /// `config.json` `model_type`, for the `/v1/models` description.
    pub model_type: String,
    /// The `fork_mode` reported in `/health` and the `qev` meta.
    pub fork_mode: String,
    /// `debias` when the query has none (qev: `$QEV_DEBIAS`, default 1).
    pub default_debias: i64,
    pub description: String,
    pub release_date: String,
}

impl ServerInfo {
    pub fn new(served_name: &str, model_type: &str) -> Self {
        ServerInfo {
            served_name: served_name.into(),
            model_type: model_type.into(),
            fork_mode: "batched".into(),
            default_debias: 1,
            description: String::new(),
            release_date: String::new(),
        }
    }

    /// What the `sysone` CLI serves: the model name (`name`, else the resolved model directory's last component),
    /// `fork_mode`, and the default `debias` (`debias`, else `$QEV_DEBIAS` as qev reads it, else 1). Shared by
    /// the server and `sysone bulk`, so both answer the same body the same way.
    pub fn for_cli(
        model_dir: &std::path::Path,
        model_type: &str,
        name: Option<&str>,
        fork_mode: &str,
        debias: Option<i64>,
    ) -> anyhow::Result<Self> {
        use anyhow::Context as _;
        let served = match name {
            Some(n) => n.to_string(),
            None => model_dir.canonicalize()?.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        };
        anyhow::ensure!(!served.is_empty(), "cannot derive a model name from {}; pass --name", model_dir.display());
        let mut info = ServerInfo::new(&served, model_type);
        info.fork_mode = fork_mode.to_string();
        info.default_debias = match (debias, std::env::var("QEV_DEBIAS")) {
            (Some(d), _) => d,
            (None, Ok(v)) => py_int(&v).with_context(|| format!("QEV_DEBIAS={v:?} is not an integer"))?,
            (None, Err(_)) => 1,
        };
        Ok(info)
    }
}

/// Server resources beyond the scheduler's (see [`AppState::start_with`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerOptions {
    /// CPU pool threads.
    pub cpu_workers: usize,
    /// New requests that may wait for a CPU thread before 503 (default: 16 per thread).
    pub cpu_queue_capacity: Option<usize>,
    /// Request bytes that may be held at once, from reading the body to the response (503 beyond).
    pub max_inflight_body_bytes: usize,
    /// Pin the GPU worker, the batcher and the CPU pool to these cores (docs/PLATFORM.md P7).
    pub pin_cores: Option<Vec<usize>>,
}

/// Default for [`ServerOptions::max_inflight_body_bytes`]: 1 GiB (four maximum-size bodies).
pub const DEFAULT_MAX_INFLIGHT_BODY_BYTES: usize = 1 << 30;

impl Default for ServerOptions {
    fn default() -> Self {
        ServerOptions { cpu_workers: 4, cpu_queue_capacity: None, max_inflight_body_bytes: DEFAULT_MAX_INFLIGHT_BODY_BYTES, pin_cores: None }
    }
}

/// The request bytes held by in-flight requests, against a cap.
pub struct BodyBudget {
    used: AtomicUsize,
    cap: usize,
}

impl BodyBudget {
    pub fn new(cap: usize) -> Self {
        BodyBudget { used: AtomicUsize::new(0), cap }
    }

    pub fn in_use(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    fn try_take(&self, n: usize) -> bool {
        self.used.fetch_update(Ordering::AcqRel, Ordering::Acquire, |u| u.checked_add(n).filter(|&t| t <= self.cap)).is_ok()
    }
}

/// Bytes taken from a [`BodyBudget`], given back on drop.
struct BodyLease<'a> {
    budget: &'a BodyBudget,
    n: usize,
}

impl BodyLease<'_> {
    fn grow(&mut self, n: usize) -> bool {
        let ok = self.budget.try_take(n);
        if ok {
            self.n += n;
        }
        ok
    }
}

impl Drop for BodyLease<'_> {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.n, Ordering::AcqRel);
    }
}

pub struct AppState {
    pub frontend: Arc<Frontend>,
    pub sched: Scheduler,
    pub pool: CpuPool,
    pub metrics: Arc<Metrics>,
    pub info: ServerInfo,
    pub bodies: BodyBudget,
    /// The listening address, for redirects to clients that send no `Host` (set by [`serve`]).
    pub local_addr: OnceLock<SocketAddr>,
}

impl AppState {
    /// Start the scheduler (building the engine on its GPU thread) and a CPU pool of `cpu_workers` threads, with
    /// the other [`ServerOptions`] at their defaults.
    pub fn start(
        frontend: Arc<Frontend>,
        info: ServerInfo,
        sched: SchedConfig,
        engine: EngineFactory,
        cpu_workers: usize,
    ) -> anyhow::Result<Arc<Self>> {
        Self::start_with(frontend, info, sched, engine, ServerOptions { cpu_workers, ..Default::default() })
    }

    pub fn start_with(
        frontend: Arc<Frontend>,
        info: ServerInfo,
        sched: SchedConfig,
        engine: EngineFactory,
        opts: ServerOptions,
    ) -> anyhow::Result<Arc<Self>> {
        let metrics = Arc::new(Metrics::new());
        let sched = Scheduler::start_with(sched, engine, metrics.clone(), opts.pin_cores.clone())?;
        let workers = opts.cpu_workers.max(1);
        let queue = opts.cpu_queue_capacity.unwrap_or(workers * crate::sched::pool::DEFAULT_QUEUE_PER_THREAD);
        let pool = CpuPool::with_options(workers, queue, opts.pin_cores);
        let bodies = BodyBudget::new(opts.max_inflight_body_bytes);
        Ok(Arc::new(AppState { frontend, sched, pool, metrics, info, bodies, local_addr: OnceLock::new() }))
    }
}

pub const SYSTEMONE: &str = "/v1/systemone";
pub const SYSTEMONE_BATCH: &str = "/v1/systemone:batch";

/// Every route, for Starlette's slash redirect.
const ROUTES: [&str; 6] = ["/health", "/v1/models", "/metrics", SYSTEMONE, SYSTEMONE_BATCH, "/"];

/// Largest request body accepted (axum's default of 2 MB is too small for long states; uvicorn has no limit).
pub const MAX_BODY_BYTES: usize = 256 << 20;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route(SYSTEMONE, post(systemone))
        .route(SYSTEMONE_BATCH, post(systemone_batch))
        .route("/", get(index))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn(request_id))
        .with_state(state)
}

// ---------------------------------------------------------------------------------------------------------------
// Responses

fn json_bytes(status: StatusCode, body: impl Into<Body>) -> Response {
    let mut r = Response::new(body.into());
    *r.status_mut() = status;
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}

fn json(status: StatusCode, v: &Json) -> Response {
    match v.dumps_compact() {
        Ok(s) => json_bytes(status, s),
        Err(_) => internal_error(),
    }
}

/// Marks a response as Starlette's answer to an unhandled exception, which qev's middleware never sees.
#[derive(Clone, Copy, Debug)]
struct Unhandled;

/// Starlette's response to an unhandled exception.
fn internal_error() -> Response {
    let mut r = text(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error");
    r.extensions_mut().insert(Unhandled);
    r
}

fn text(status: StatusCode, body: &'static str) -> Response {
    let mut r = Response::new(Body::from(body));
    *r.status_mut() = status;
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    r
}

fn overloaded_body() -> Json {
    Json::Object(vec![("detail".into(), Json::str("admission queue is full"))])
}

fn detail(msg: &str) -> Json {
    Json::Object(vec![("detail".into(), Json::str(msg))])
}

async fn not_found(State(st): State<Arc<AppState>>, req: Request) -> Response {
    if let Some(location) = slash_redirect(&req, st.local_addr.get()) {
        let mut r = Response::new(Body::empty());
        *r.status_mut() = StatusCode::TEMPORARY_REDIRECT;
        if let Ok(v) = HeaderValue::from_str(&location) {
            r.headers_mut().insert(header::LOCATION, v);
            return r;
        }
    }
    json(StatusCode::NOT_FOUND, &detail("Not Found"))
}

async fn method_not_allowed() -> Response {
    json(StatusCode::METHOD_NOT_ALLOWED, &detail("Method Not Allowed"))
}

/// `urllib.parse.unquote` of a path (invalid UTF-8 replaced, as uvicorn decodes it).
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%'
            && i + 2 < b.len()
            && let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2]))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `urllib.parse.quote(url, safe=":/%#?=@[]!$&'()*+,;")`, as Starlette's `RedirectResponse` sets `location`.
fn quote_location(url: &str) -> String {
    const SAFE: &[u8] = b":/%#?=@[]!$&'()*+,;_.-~";
    let mut out = String::with_capacity(url.len());
    for &c in url.as_bytes() {
        if c.is_ascii_alphanumeric() || SAFE.contains(&c) {
            out.push(c as char);
        } else {
            out.push_str(&format!("%{c:02X}"));
        }
    }
    out
}

/// Starlette's `Router(redirect_slashes=True)`: when the path matches no route but does with its trailing
/// slashes removed (or one added), the absolute URL to redirect to.
fn slash_redirect(req: &Request, local: Option<&SocketAddr>) -> Option<String> {
    let path = percent_decode(req.uri().path());
    if path == "/" {
        return None;
    }
    let other = if path.ends_with('/') { path.trim_end_matches('/').to_string() } else { format!("{path}/") };
    if !ROUTES.contains(&other.as_str()) {
        return None;
    }
    // Starlette takes the netloc from the Host header, else the server's own address.
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .filter(|h| !h.is_empty() && !h.contains(['/', '?', '#', '@']))
        .map(str::to_string)
        .or_else(|| local.map(|a| if a.port() == 80 { a.ip().to_string() } else { a.to_string() }))?;
    let query = req.uri().query().filter(|q| !q.is_empty()).map(|q| format!("?{q}")).unwrap_or_default();
    Some(quote_location(&format!("http://{host}{other}{query}")))
}

/// A fresh request id as qev mints it: `f"qev-{uuid.uuid4().hex[:16]}"` (so the 13th hex digit is the UUID
/// version, 4).
fn new_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // A randomly keyed SipHash (std's RandomState draws its keys from the OS) of a counter.
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    let x = h.finish();
    format!("qev-{:012x}4{:03x}", x >> 16, x & 0xfff)
}

/// qev's middleware: echo or mint `x-typesafe-request-id`, and report the server time.
async fn request_id(req: Request, next: Next) -> Response {
    let t0 = Instant::now();
    // `request.headers.get(...) or f"qev-..."`: an empty header counts as absent.
    let rid = req.headers().get("x-typesafe-request-id").filter(|v| !v.is_empty()).cloned();
    let mut resp = next.run(req).await;
    if resp.extensions().get::<Unhandled>().is_some() {
        return resp;
    }
    let rid = rid.unwrap_or_else(|| HeaderValue::from_str(&new_request_id()).unwrap());
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    resp.headers_mut().insert("x-typesafe-request-id", rid);
    resp.headers_mut().insert("x-qev-server-time-ms", HeaderValue::from_str(&format!("{ms:.1}")).unwrap());
    resp
}

// ---------------------------------------------------------------------------------------------------------------
// Simple routes

async fn health(State(st): State<Arc<AppState>>) -> Response {
    json(
        StatusCode::OK,
        &Json::Object(vec![
            ("status".into(), Json::str("ok")),
            ("model".into(), Json::str(st.info.served_name.as_str())),
            ("fork_mode".into(), Json::str(st.info.fork_mode.as_str())),
            ("prompt_version".into(), Json::str(PROMPT_VERSION)),
        ]),
    )
}

async fn models(State(st): State<Arc<AppState>>) -> Response {
    let info = &st.info;
    let release = if info.release_date.is_empty() { "2026-09-18".to_string() } else { info.release_date.clone() };
    let item = |name: &str, desc: String| {
        Json::Object(vec![
            ("name".into(), Json::str(name)),
            ("description".into(), Json::str(desc)),
            ("release_date".into(), Json::str(release.as_str())),
        ])
    };
    let desc = if info.description.is_empty() {
        format!("Qev decision model ({})", info.model_type)
    } else {
        info.description.clone()
    };
    let mut items = vec![item(&info.served_name, desc)];
    items.extend(ALIASES.iter().map(|a| item(a, format!("alias for {}", info.served_name))));
    json(StatusCode::OK, &Json::Object(vec![("models".into(), Json::Array(items))]))
}

async fn metrics(State(st): State<Arc<AppState>>) -> Response {
    let mut text = st.metrics.render(st.sched.queue_depth(), st.sched.queue_capacity(), st.sched.policy().token_budget);
    crate::sched::metrics::render_front(
        &mut text,
        &crate::sched::metrics::Front {
            cpu_queue_depth: st.pool.queue_depth(),
            cpu_queue_capacity: st.pool.queue_capacity(),
            skipped: st.pool.skipped(),
            body_bytes: st.bodies.in_use(),
            body_capacity: st.bodies.capacity(),
        },
    );
    let mut r = Response::new(Body::from(text));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"));
    r
}

async fn index() -> Response {
    let mut r = Response::new(Body::from("<h1>sysone</h1><p>POST /v1/systemone</p>"));
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    r
}

// ---------------------------------------------------------------------------------------------------------------
// Decisions

/// The outcome of a request that never reaches the engine, or of one that did (shared with [`crate::bulk`]).
pub(crate) enum Outcome {
    /// Status and JSON body.
    Json(StatusCode, Json),
    /// A 500 (Starlette's plain-text body).
    Internal,
    /// Turned away for load ([`overloaded_body`]); the reason is counted in `/metrics`.
    Overloaded(&'static str),
}

impl Outcome {
    pub(crate) fn status(&self) -> u16 {
        match self {
            Outcome::Json(s, _) => s.as_u16(),
            Outcome::Internal => 500,
            Outcome::Overloaded(_) => 503,
        }
    }

    fn into_response(self) -> Response {
        match self {
            Outcome::Json(s, v) => json(s, &v),
            Outcome::Internal => internal_error(),
            Outcome::Overloaded(_) => json(StatusCode::SERVICE_UNAVAILABLE, &overloaded_body()),
        }
    }

    /// One entry of a `:batch` response (and of a bulk result line: `status` and `body`).
    pub(crate) fn into_item(self) -> Json {
        let status = self.status();
        let body = match self {
            Outcome::Json(_, v) => v,
            Outcome::Internal => detail("Internal Server Error"),
            Outcome::Overloaded(_) => overloaded_body(),
        };
        Json::Object(vec![("status".into(), Json::int(status as i64)), ("body".into(), body)])
    }

    fn api(e: &ApiError) -> Self {
        Outcome::Json(StatusCode::from_u16(e.status()).unwrap(), e.body())
    }
}

/// What the scheduler packs for one request: its plan, and whether it is a debug request (which asks the engine
/// for `allowed_token_mass`).
struct Work {
    plan: Arc<Plan>,
    debug: bool,
}

impl Packable for Work {
    fn num_tokens(&self) -> usize {
        self.plan.total
    }

    fn num_branches(&self) -> usize {
        self.plan.branches.len()
    }

    fn push_into(&self, batch: &mut DecisionBatch) -> anyhow::Result<usize> {
        self.plan.push_into(batch)
    }

    fn wants_mass(&self) -> bool {
        self.debug
    }
}

/// A planned request, ready for the engine.
pub(crate) struct Ready {
    pub(crate) req: SystemOneRequest,
    pub(crate) plan: Arc<Plan>,
    pub(crate) debug: bool,
    pub(crate) started: Instant,
    /// When planning ended (the start of `queue_s`).
    pub(crate) planned: Instant,
}

/// The query parameters of a decision call, as qev reads them.
#[derive(Clone, Debug)]
pub(crate) struct Params {
    debug: bool,
    /// None: present but not an integer (qev raises inside the handler: 500).
    debias: Option<i64>,
}

pub(crate) fn params(raw: Option<&str>, default_debias: i64) -> Params {
    let qs = query::parse_qsl(raw.unwrap_or(""));
    // `bool(request.query_params.get("debug"))`: any non-empty value, even "0", turns it on.
    let debug = query::get(&qs, "debug").is_some_and(|v| !v.is_empty());
    let debias = match query::get(&qs, "debias") {
        None => Some(default_debias),
        Some(v) => py_int(v),
    };
    Params { debug, debias }
}

/// Everything before the engine, in qev's order: validate, route the model, read `debug`/`debias`, plan.
pub(crate) fn prepare(frontend: &Frontend, info: &ServerInfo, body: Result<Json, ApiError>, p: &Params) -> Result<Ready, Outcome> {
    let started = Instant::now();
    let req = body.and_then(|v| SystemOneRequest::validate(&v)).map_err(|e| Outcome::api(&e))?;
    check_model(&req.model, &info.served_name).map_err(|e| Outcome::api(&e))?;
    let Some(debias) = p.debias else { return Err(Outcome::Internal) };
    let plan = match frontend.plan(&req, debias) {
        Ok(plan) => plan,
        Err(PlanError::Value(msg)) => return Err(Outcome::api(&ApiError::value_error(msg))),
        Err(PlanError::Internal(e)) => {
            eprintln!("sysone: planning failed: {e:#}");
            return Err(Outcome::Internal);
        }
    };
    Ok(Ready { req, plan: Arc::new(plan), debug: p.debug, started, planned: Instant::now() })
}

/// Answer assembly and the response body.
pub(crate) fn finish(frontend: &Frontend, info: &ServerInfo, r: &Ready, scored: &Scored) -> Outcome {
    let plan = &r.plan;
    let masses = if r.debug { scored.allowed_mass.as_deref() } else { None };
    let out = match frontend.assemble(&r.req, plan, |i| scored.row(i, plan.branches[i].n_slots()), masses) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("sysone: answer assembly failed: {e:#}");
            return Outcome::Internal;
        }
    };
    let resp = frontend.response(plan, out.answers, &info.served_name);
    let mut meta = frontend.meta(plan, &info.fork_mode).to_json();
    if let Json::Object(m) = &mut meta {
        let render_s = r.planned.saturating_duration_since(r.started).as_secs_f64();
        let queue_s = scored.step_started.saturating_duration_since(r.planned).as_secs_f64();
        m.push((
            "timing".into(),
            Json::Object(vec![
                ("render_s".into(), Json::Float(render_s)),
                ("forward_s".into(), Json::Float(scored.step.as_secs_f64())),
                ("total_s".into(), Json::Float(r.started.elapsed().as_secs_f64())),
                ("queue_s".into(), Json::Float(queue_s)),
            ]),
        ));
        if r.debug {
            m.push(("diagnostics".into(), out.diagnostics));
        }
    }
    Outcome::Json(StatusCode::OK, resp.to_json(Some(meta)))
}

pub(crate) fn parse_body(body: &[u8]) -> Result<Json, ApiError> {
    pyjson::parse(body).map_err(|_| ApiError::invalid_json())
}

/// Admit planned requests (all or none), wait for their logits, and assemble each answer.
async fn run_ready(st: &Arc<AppState>, ready: Vec<Ready>) -> Result<Vec<Outcome>, Outcome> {
    let works: Vec<Arc<dyn Packable>> =
        ready.iter().map(|r| Arc::new(Work { plan: r.plan.clone(), debug: r.debug }) as Arc<dyn Packable>).collect();
    let rxs = st.sched.submit(works).map_err(|_| Outcome::Overloaded("queue"))?;
    let mut scored = Vec::with_capacity(rxs.len());
    for rx in rxs {
        scored.push(match rx.await {
            Ok(Ok(s)) => Some(s),
            Ok(Err(msg)) => {
                eprintln!("sysone: {msg}");
                None
            }
            Err(_) => None,
        });
    }
    let st2 = st.clone();
    let done = st
        .pool
        .run(move || {
            ready
                .iter()
                .zip(&scored)
                .map(|(r, s)| match s {
                    Some(s) => {
                        let out = finish(&st2.frontend, &st2.info, r, s);
                        if out.status() == 200 {
                            st2.metrics.record_decided(r.started.elapsed(), r.plan.num_questions);
                        }
                        out
                    }
                    None => Outcome::Internal,
                })
                .collect::<Vec<_>>()
        })
        .await;
    done.map_err(|_| Outcome::Internal)
}

/// Why a body could not be read.
enum BodyError {
    TooLarge,
    Overloaded,
    Broken,
}

impl BodyError {
    fn into_response(self, st: &AppState, endpoint: &str) -> Response {
        let r = match self {
            BodyError::TooLarge => text(StatusCode::PAYLOAD_TOO_LARGE, "Failed to buffer the request body: length limit exceeded"),
            BodyError::Broken => text(StatusCode::BAD_REQUEST, "Failed to buffer the request body"),
            BodyError::Overloaded => {
                st.metrics.record_overload("body_memory");
                json(StatusCode::SERVICE_UNAVAILABLE, &overloaded_body())
            }
        };
        st.metrics.record_response(endpoint, r.status().as_u16());
        r
    }
}

/// Read a request body of at most [`MAX_BODY_BYTES`], holding its size against the in-flight body budget (a
/// declared `content-length` is reserved before anything is read).
async fn read_body<'a>(st: &'a AppState, headers: &HeaderMap, body: Body) -> Result<(Bytes, BodyLease<'a>), BodyError> {
    let declared = headers.get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<usize>().ok());
    if declared.is_some_and(|n| n > MAX_BODY_BYTES) {
        return Err(BodyError::TooLarge);
    }
    let mut lease = BodyLease { budget: &st.bodies, n: 0 };
    if let Some(n) = declared
        && !lease.grow(n)
    {
        return Err(BodyError::Overloaded);
    }
    let mut buf = Vec::with_capacity(declared.unwrap_or(0));
    let mut body = body;
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|_| BodyError::Broken)?;
        let Ok(data) = frame.into_data() else { continue };
        if buf.len() + data.len() > MAX_BODY_BYTES {
            return Err(BodyError::TooLarge);
        }
        let need = (buf.len() + data.len()).saturating_sub(lease.n);
        if need > 0 && !lease.grow(need) {
            return Err(BodyError::Overloaded);
        }
        buf.extend_from_slice(&data);
    }
    Ok((Bytes::from(buf), lease))
}

async fn systemone(State(st): State<Arc<AppState>>, RawQuery(q): RawQuery, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let (body, _lease) = match read_body(&st, &parts.headers, body).await {
        Ok(b) => b,
        Err(e) => return e.into_response(&st, SYSTEMONE),
    };
    let out = decide_one(&st, q, body).await;
    respond(&st, SYSTEMONE, out)
}

fn respond(st: &AppState, endpoint: &str, out: Outcome) -> Response {
    if let Outcome::Overloaded(reason) = out {
        st.metrics.record_overload(reason);
    }
    st.metrics.record_response(endpoint, out.status());
    out.into_response()
}

async fn decide_one(st: &Arc<AppState>, q: Option<String>, body: Bytes) -> Outcome {
    let p = params(q.as_deref(), st.info.default_debias);
    let st2 = st.clone();
    let prepared = st.pool.try_run(move || prepare(&st2.frontend, &st2.info, parse_body(&body), &p)).await;
    let ready = match prepared {
        Ok(Ok(r)) => r,
        Ok(Err(o)) => return o,
        Err(PoolError::Full) => return Outcome::Overloaded("cpu_queue"),
        Err(PoolError::Panicked) => return Outcome::Internal,
    };
    match run_ready(st, vec![ready]).await {
        Ok(mut v) => v.pop().unwrap_or(Outcome::Internal),
        Err(o) => o,
    }
}

/// `POST /v1/systemone:batch` (sysone extension): the body is a JSON array of `/v1/systemone` request bodies; the
/// query parameters apply to each. The answer is `{"results": [{"status": s, "body": b}, ...]}` in request order,
/// where each status and body are what `/v1/systemone` would have answered for that request. The valid requests
/// are admitted together: if the queue cannot take all of them now, the whole call is a 503; if the array has
/// more items than the queue can ever hold, the call is a 413 (split it).
async fn systemone_batch(State(st): State<Arc<AppState>>, RawQuery(q): RawQuery, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let (body, _lease) = match read_body(&st, &parts.headers, body).await {
        Ok(b) => b,
        Err(e) => return e.into_response(&st, SYSTEMONE_BATCH),
    };
    let out = decide_batch(&st, q, body).await;
    respond(&st, SYSTEMONE_BATCH, out)
}

async fn decide_batch(st: &Arc<AppState>, q: Option<String>, body: Bytes) -> Outcome {
    let p = params(q.as_deref(), st.info.default_debias);
    let st2 = st.clone();
    let capacity = st.sched.queue_capacity();
    let prepared = st
        .pool
        .try_run(move || -> Result<Vec<Result<Ready, Outcome>>, Outcome> {
            let items = match parse_body(&body) {
                Ok(Json::Array(items)) => items,
                Ok(_) => {
                    let e = ErrorItem { loc: vec!["body".into()], msg: "Input should be a valid list".into(), kind: "list_type".into() };
                    return Err(Outcome::api(&ApiError::Unprocessable(vec![e])));
                }
                Err(e) => return Err(Outcome::api(&e)),
            };
            if items.len() > capacity {
                let msg = format!(
                    "batch of {} requests exceeds the admission queue capacity of {capacity}; send at most {capacity} per call",
                    items.len()
                );
                return Err(Outcome::Json(StatusCode::PAYLOAD_TOO_LARGE, detail(&msg)));
            }
            Ok(items.into_iter().map(|v| prepare(&st2.frontend, &st2.info, Ok(v), &p)).collect())
        })
        .await;
    let prepared = match prepared {
        Ok(Ok(v)) => v,
        Ok(Err(o)) => return o,
        Err(PoolError::Full) => return Outcome::Overloaded("cpu_queue"),
        Err(PoolError::Panicked) => return Outcome::Internal,
    };
    let mut slots: Vec<Option<Outcome>> = Vec::with_capacity(prepared.len());
    let mut ready = Vec::new();
    for r in prepared {
        match r {
            Ok(r) => {
                ready.push(r);
                slots.push(None);
            }
            Err(o) => slots.push(Some(o)),
        }
    }
    let mut answered = if ready.is_empty() {
        Vec::new().into_iter()
    } else {
        match run_ready(st, ready).await {
            Ok(v) => v.into_iter(),
            Err(o) => return o,
        }
    };
    let results = slots
        .into_iter()
        .map(|s| {
            let o = s.unwrap_or_else(|| answered.next().unwrap_or(Outcome::Internal));
            o.into_item()
        })
        .collect();
    Outcome::Json(StatusCode::OK, Json::Object(vec![("results".into(), Json::Array(results))]))
}

/// Serve until `shutdown` resolves.
pub async fn serve(
    listener: tokio::net::TcpListener,
    state: Arc<AppState>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    if let Ok(a) = listener.local_addr() {
        let _ = state.local_addr.set(a);
    }
    axum::serve(listener, router(state)).with_graceful_shutdown(shutdown).await
}

/// How long to wait between polls in [`wait_ready`].
const POLL: Duration = Duration::from_millis(10);

/// Wait until `addr` accepts connections (for tests and tools that start a server in-process).
pub async fn wait_ready(addr: &str, timeout: Duration) -> bool {
    let end = Instant::now() + timeout;
    while Instant::now() < end {
        if tokio::net::TcpStream::connect(addr).await.is_ok() {
            return true;
        }
        tokio::time::sleep(POLL).await;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_ids_look_like_qev() {
        let a = new_request_id();
        let b = new_request_id();
        assert_ne!(a, b);
        for id in [&a, &b] {
            assert_eq!(id.len(), 20, "{id}");
            assert!(id.starts_with("qev-") && id[4..].chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()), "{id}");
            // uuid4().hex[12] is the version digit.
            assert_eq!(&id[16..17], "4", "{id}");
        }
    }

    #[test]
    fn location_is_quoted_like_starlette() {
        assert_eq!(quote_location("http://h:1/health?a=b c&d=é"), "http://h:1/health?a=b%20c&d=%C3%A9");
        assert_eq!(percent_decode("/heal%74h/%2"), "/health/%2");
        assert_eq!(percent_decode("/%ff"), "/\u{fffd}");
    }

    fn req(path: &str, host: Option<&str>) -> Request {
        let mut b = axum::http::Request::builder().uri(path);
        if let Some(h) = host {
            b = b.header("host", h);
        }
        b.body(Body::empty()).unwrap()
    }

    #[test]
    fn slash_redirects_follow_starlette() {
        let local: SocketAddr = "127.0.0.1:8000".parse().unwrap();
        let r = |p: &str, h: Option<&str>| slash_redirect(&req(p, h), Some(&local));
        assert_eq!(r("/health/", Some("example:9")).as_deref(), Some("http://example:9/health"));
        assert_eq!(r("/health//?x=1", Some("e")).as_deref(), Some("http://e/health?x=1"));
        assert_eq!(r("/v1/systemone/", None).as_deref(), Some("http://127.0.0.1:8000/v1/systemone"));
        assert_eq!(r("/heal%74h/", Some("e")).as_deref(), Some("http://e/health"));
        assert_eq!(r("/v1/systemone:batch/", Some("e")).as_deref(), Some("http://e/v1/systemone:batch"));
        for p in ["/", "/nope", "/nope/", "//", "/health"] {
            assert_eq!(r(p, Some("e")), None, "{p}");
        }
    }

    #[test]
    fn body_budget_is_a_hard_cap() {
        let b = BodyBudget::new(100);
        let mut l1 = BodyLease { budget: &b, n: 0 };
        assert!(l1.grow(60));
        {
            let mut l2 = BodyLease { budget: &b, n: 0 };
            assert!(!l2.grow(41));
            assert!(l2.grow(40));
            assert_eq!(b.in_use(), 100);
        }
        assert_eq!(b.in_use(), 60);
        drop(l1);
        assert_eq!(b.in_use(), 0);
    }
}
