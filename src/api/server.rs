//! The HTTP server: qev's routes with its status codes, bodies, headers and query parameters
//! (`qev/server.py` at OneJev@81ce62f), plus `/metrics` and the additive `/v1/systemone:batch`.
//!
//! A `/v1/systemone` request goes: CPU pool (parse, validate, route the model, parse `debug`/`debias`, render,
//! tokenize, plan, check limits) → admission queue (503 when full) → batcher → GPU worker → CPU pool (answer
//! assembly, response JSON).
//!
//! Differences from qev that a client can see: a full admission queue answers 503
//! `{"detail": "admission queue is full"}` (qev has no queue), and `qev.diagnostics` (with `?debug=1`) has no
//! `allowed_token_mass`, because the engine contract returns slot logits only.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, RawQuery, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::routing::{get, post};

use crate::api::query::{self, py_int};
use crate::api::schema::{ALIASES, ApiError, ErrorItem, SystemOneRequest, check_model};
use crate::frontend::pyjson::{self, Json};
use crate::frontend::render::PROMPT_VERSION;
use crate::frontend::{Frontend, Plan, PlanError};
use crate::sched::{CpuPool, EngineFactory, Metrics, Packable, SchedConfig, Scheduler, Scored};

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
}

pub struct AppState {
    pub frontend: Arc<Frontend>,
    pub sched: Scheduler,
    pub pool: CpuPool,
    pub metrics: Arc<Metrics>,
    pub info: ServerInfo,
}

impl AppState {
    /// Start the scheduler (building the engine on its GPU thread) and a CPU pool of `cpu_workers` threads.
    pub fn start(
        frontend: Arc<Frontend>,
        info: ServerInfo,
        sched: SchedConfig,
        engine: EngineFactory,
        cpu_workers: usize,
    ) -> anyhow::Result<Arc<Self>> {
        let metrics = Arc::new(Metrics::new());
        let sched = Scheduler::start(sched, engine, metrics.clone())?;
        Ok(Arc::new(AppState { frontend, sched, pool: CpuPool::new(cpu_workers), metrics, info }))
    }
}

pub const SYSTEMONE: &str = "/v1/systemone";
pub const SYSTEMONE_BATCH: &str = "/v1/systemone:batch";

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
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
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

/// Starlette's response to an unhandled exception.
fn internal_error() -> Response {
    let mut r = Response::new(Body::from("Internal Server Error"));
    *r.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; charset=utf-8"));
    r
}

fn overloaded_body() -> Json {
    Json::Object(vec![("detail".into(), Json::str("admission queue is full"))])
}

fn detail(msg: &str) -> Json {
    Json::Object(vec![("detail".into(), Json::str(msg))])
}

async fn not_found() -> Response {
    json(StatusCode::NOT_FOUND, &detail("Not Found"))
}

async fn method_not_allowed() -> Response {
    json(StatusCode::METHOD_NOT_ALLOWED, &detail("Method Not Allowed"))
}

/// qev's middleware: echo or mint `x-typesafe-request-id`, and report the server time.
async fn request_id(req: Request, next: Next) -> Response {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let t0 = Instant::now();
    let rid = req.headers().get("x-typesafe-request-id").cloned();
    let mut resp = next.run(req).await;
    let rid = rid.unwrap_or_else(|| {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        let mut x = nanos ^ COUNTER.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9e3779b97f4a7c15);
        // splitmix64
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        HeaderValue::from_str(&format!("qev-{x:016x}")).unwrap()
    });
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
    let text = st.metrics.render(st.sched.queue_depth(), st.sched.queue_capacity(), st.sched.policy().token_budget);
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

/// The outcome of a request that never reaches the engine, or of one that did.
enum Outcome {
    /// Status and JSON body.
    Json(StatusCode, Json),
    /// A 500 (Starlette's plain-text body).
    Internal,
}

impl Outcome {
    fn status(&self) -> u16 {
        match self {
            Outcome::Json(s, _) => s.as_u16(),
            Outcome::Internal => 500,
        }
    }

    fn into_response(self) -> Response {
        match self {
            Outcome::Json(s, v) => json(s, &v),
            Outcome::Internal => internal_error(),
        }
    }

    /// One entry of a `:batch` response.
    fn into_item(self) -> Json {
        let status = self.status();
        let body = match self {
            Outcome::Json(_, v) => v,
            Outcome::Internal => detail("Internal Server Error"),
        };
        Json::Object(vec![("status".into(), Json::int(status as i64)), ("body".into(), body)])
    }

    fn api(e: &ApiError) -> Self {
        Outcome::Json(StatusCode::from_u16(e.status()).unwrap(), e.body())
    }
}

/// A planned request, ready for the engine.
struct Ready {
    req: SystemOneRequest,
    plan: Arc<Plan>,
    debug: bool,
    started: Instant,
    render_s: f64,
}

/// The query parameters of a decision call, as qev reads them.
#[derive(Clone, Debug)]
struct Params {
    debug: bool,
    /// None: present but not an integer (qev raises inside the handler: 500).
    debias: Option<i64>,
}

fn params(raw: Option<&str>, default_debias: i64) -> Params {
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
fn prepare(frontend: &Frontend, info: &ServerInfo, body: Result<Json, ApiError>, p: &Params) -> Result<Ready, Outcome> {
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
    Ok(Ready { req, plan: Arc::new(plan), debug: p.debug, started, render_s: started.elapsed().as_secs_f64() })
}

/// Answer assembly and the response body.
fn finish(frontend: &Frontend, info: &ServerInfo, r: &Ready, scored: &Scored, forward_s: f64) -> Outcome {
    let plan = &r.plan;
    let out = match frontend.assemble(&r.req, plan, |i| scored.row(i, plan.branches[i].n_slots()), None) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("sysone: answer assembly failed: {e:#}");
            return Outcome::Internal;
        }
    };
    let resp = frontend.response(plan, out.answers, &info.served_name);
    let mut meta = frontend.meta(plan, &info.fork_mode).to_json();
    if let Json::Object(m) = &mut meta {
        m.push((
            "timing".into(),
            Json::Object(vec![
                ("render_s".into(), Json::Float(r.render_s)),
                ("forward_s".into(), Json::Float(forward_s)),
                ("total_s".into(), Json::Float(r.started.elapsed().as_secs_f64())),
            ]),
        ));
        if r.debug {
            m.push(("diagnostics".into(), out.diagnostics));
        }
    }
    Outcome::Json(StatusCode::OK, resp.to_json(Some(meta)))
}

fn parse_body(body: &[u8]) -> Result<Json, ApiError> {
    pyjson::parse(body).map_err(|_| ApiError::invalid_json())
}

/// Admit planned requests (all or none), wait for their logits, and assemble each answer.
async fn run_ready(st: &Arc<AppState>, ready: Vec<Ready>) -> Result<Vec<Outcome>, Outcome> {
    let works: Vec<Arc<dyn Packable>> = ready.iter().map(|r| r.plan.clone() as Arc<dyn Packable>).collect();
    let submitted = Instant::now();
    let rxs = st.sched.submit(works).map_err(|_| Outcome::Json(StatusCode::SERVICE_UNAVAILABLE, overloaded_body()))?;
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
    let forward_s = submitted.elapsed().as_secs_f64();
    let st2 = st.clone();
    let done = st
        .pool
        .run(move || {
            ready
                .iter()
                .zip(&scored)
                .map(|(r, s)| match s {
                    Some(s) => {
                        let out = finish(&st2.frontend, &st2.info, r, s, forward_s);
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

async fn systemone(State(st): State<Arc<AppState>>, RawQuery(q): RawQuery, body: Bytes) -> Response {
    let out = decide_one(&st, q, body).await;
    st.metrics.record_response(SYSTEMONE, out.status());
    out.into_response()
}

async fn decide_one(st: &Arc<AppState>, q: Option<String>, body: Bytes) -> Outcome {
    let p = params(q.as_deref(), st.info.default_debias);
    let st2 = st.clone();
    let prepared = st.pool.run(move || prepare(&st2.frontend, &st2.info, parse_body(&body), &p)).await;
    let ready = match prepared {
        Ok(Ok(r)) => r,
        Ok(Err(o)) => return o,
        Err(_) => return Outcome::Internal,
    };
    match run_ready(st, vec![ready]).await {
        Ok(mut v) => v.pop().unwrap_or(Outcome::Internal),
        Err(o) => o,
    }
}

/// `POST /v1/systemone:batch` (sysone extension): the body is a JSON array of `/v1/systemone` request bodies; the
/// query parameters apply to each. The answer is `{"results": [{"status": s, "body": b}, ...]}` in request order,
/// where each status and body are what `/v1/systemone` would have answered for that request. The valid requests
/// are admitted together: if the queue cannot take all of them, the whole call is a 503.
async fn systemone_batch(State(st): State<Arc<AppState>>, RawQuery(q): RawQuery, body: Bytes) -> Response {
    let out = decide_batch(&st, q, body).await;
    st.metrics.record_response(SYSTEMONE_BATCH, out.status());
    out.into_response()
}

async fn decide_batch(st: &Arc<AppState>, q: Option<String>, body: Bytes) -> Outcome {
    let p = params(q.as_deref(), st.info.default_debias);
    let st2 = st.clone();
    let prepared = st
        .pool
        .run(move || -> Result<Vec<Result<Ready, Outcome>>, Outcome> {
            let items = match parse_body(&body) {
                Ok(Json::Array(items)) => items,
                Ok(_) => {
                    let e = ErrorItem { loc: vec!["body".into()], msg: "Input should be a valid list".into(), kind: "list_type".into() };
                    return Err(Outcome::api(&ApiError::Unprocessable(vec![e])));
                }
                Err(e) => return Err(Outcome::api(&e)),
            };
            Ok(items.into_iter().map(|v| prepare(&st2.frontend, &st2.info, Ok(v), &p)).collect())
        })
        .await;
    let prepared = match prepared {
        Ok(Ok(v)) => v,
        Ok(Err(o)) => return o,
        Err(_) => return Outcome::Internal,
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
