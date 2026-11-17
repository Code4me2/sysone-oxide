//! Server metrics, rendered as Prometheus text at `/metrics` (DESIGN.md §7): queue depth, batch tokens, step time,
//! decisions/s, and request latency quantiles.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How far back the latency quantiles and the decisions/s rate look.
pub const LATENCY_WINDOW: Duration = Duration::from_secs(60);
pub const RATE_WINDOW: Duration = Duration::from_secs(10);
/// At most this many latency samples are kept for the quantiles.
const MAX_LATENCY_SAMPLES: usize = 1 << 16;

const TOKEN_BUCKETS: [f64; 10] = [256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0];
const COUNT_BUCKETS: [f64; 9] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];
const SECONDS_BUCKETS: [f64; 13] =
    [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0];

#[derive(Clone, Debug)]
struct Histogram {
    bounds: &'static [f64],
    counts: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Histogram { bounds, counts: vec![0; bounds.len()], sum: 0.0, count: 0 }
    }

    fn observe(&mut self, x: f64) {
        for (b, c) in self.bounds.iter().zip(&mut self.counts) {
            if x <= *b {
                *c += 1;
            }
        }
        self.sum += x;
        self.count += 1;
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        for (b, c) in self.bounds.iter().zip(&self.counts) {
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {c}");
        }
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {}", self.count);
        let _ = writeln!(out, "{name}_sum {}", self.sum);
        let _ = writeln!(out, "{name}_count {}", self.count);
    }
}

struct Inner {
    /// (endpoint, status) → responses.
    requests: BTreeMap<(String, u16), u64>,
    rejected_full: u64,
    /// 503s by reason ("queue", "cpu_queue", "body_memory").
    overload: BTreeMap<&'static str, u64>,
    decisions_total: u64,
    branches_total: u64,
    batches_total: u64,
    batch_tokens_total: u64,
    /// Bucket (padded) tokens of the steps whose engine reports them (`DecisionEngine::last_bucket_tokens`).
    bucket_tokens_total: u64,
    /// Packed tokens of those same steps, so padding = 1 - this / `bucket_tokens_total`.
    bucket_batch_tokens_total: u64,
    engine_busy_seconds: f64,
    /// Start of the step running now: `/metrics` counts its time so far, so the busy counter never jumps by a
    /// whole step at its end (a scrape window then never sees more busy time than wall time).
    running_since: Option<Instant>,
    batch_tokens: Histogram,
    batch_requests: Histogram,
    step_seconds: Histogram,
    queue_seconds: Histogram,
    latency: VecDeque<(Instant, f64)>,
    latency_sum: f64,
    latency_count: u64,
    decisions: VecDeque<(Instant, u64)>,
}

/// Shared by the HTTP handlers, the batcher and the GPU worker. One mutex: every update is a few adds, far below
/// the per-request cost.
pub struct Metrics {
    started: Instant,
    inner: Mutex<Inner>,
    /// `DecisionEngine::info` of the running engine, rendered as `sysone_engine_info` labels.
    engine_info: Mutex<Vec<(&'static str, String)>>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// The value at quantile `q` of sorted samples (nearest rank).
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            engine_info: Mutex::new(Vec::new()),
            started: Instant::now(),
            inner: Mutex::new(Inner {
                requests: BTreeMap::new(),
                rejected_full: 0,
                overload: BTreeMap::new(),
                decisions_total: 0,
                branches_total: 0,
                batches_total: 0,
                batch_tokens_total: 0,
                bucket_tokens_total: 0,
                bucket_batch_tokens_total: 0,
                engine_busy_seconds: 0.0,
                running_since: None,
                batch_tokens: Histogram::new(&TOKEN_BUCKETS),
                batch_requests: Histogram::new(&COUNT_BUCKETS),
                step_seconds: Histogram::new(&SECONDS_BUCKETS),
                queue_seconds: Histogram::new(&SECONDS_BUCKETS),
                latency: VecDeque::new(),
                latency_sum: 0.0,
                latency_count: 0,
                decisions: VecDeque::new(),
            }),
        }
    }

    /// One HTTP response (a `:batch` call counts once per item under its own endpoint label).
    pub fn record_response(&self, endpoint: &str, status: u16) {
        let mut m = self.inner.lock().unwrap();
        *m.requests.entry((endpoint.to_string(), status)).or_default() += 1;
        if status == 503 {
            m.rejected_full += 1;
        }
    }

    /// A request turned away for load (its 503 is also counted by [`record_response`](Self::record_response)).
    pub fn record_overload(&self, reason: &'static str) {
        *self.inner.lock().unwrap().overload.entry(reason).or_default() += 1;
    }

    /// One answered request: its end-to-end latency and the number of questions (decisions) it answered.
    pub fn record_decided(&self, latency: Duration, questions: usize) {
        let now = Instant::now();
        let mut m = self.inner.lock().unwrap();
        let s = latency.as_secs_f64();
        m.latency_sum += s;
        m.latency_count += 1;
        m.latency.push_back((now, s));
        if m.latency.len() > MAX_LATENCY_SAMPLES {
            m.latency.pop_front();
        }
        m.decisions_total += questions as u64;
        m.decisions.push_back((now, questions as u64));
        Self::trim(&mut m, now);
    }

    /// An engine step starts now (ended by [`record_batch`](Self::record_batch)).
    pub fn record_step_start(&self, at: Instant) {
        self.inner.lock().unwrap().running_since = Some(at);
    }

    /// One engine step.
    pub fn record_batch(&self, tokens: usize, requests: usize, branches: usize, step: Duration) {
        let mut m = self.inner.lock().unwrap();
        m.batches_total += 1;
        m.batch_tokens_total += tokens as u64;
        m.branches_total += branches as u64;
        // The part of the step already counted while it ran (see `record_step_start`) is not counted again.
        match m.running_since.take() {
            Some(t) => m.engine_busy_seconds += t.elapsed().as_secs_f64(),
            None => m.engine_busy_seconds += step.as_secs_f64(),
        }
        m.batch_tokens.observe(tokens as f64);
        m.batch_requests.observe(requests as f64);
        m.step_seconds.observe(step.as_secs_f64());
    }

    /// The bucket capacity a step of `tokens` packed tokens ran in (the engine computed `bucket - tokens` rows of
    /// padding).
    pub fn record_bucket(&self, tokens: usize, bucket_tokens: usize) {
        let mut m = self.inner.lock().unwrap();
        m.bucket_tokens_total += bucket_tokens as u64;
        m.bucket_batch_tokens_total += tokens as u64;
    }

    /// Time a request spent between admission and the start of its engine step.
    pub fn record_queue_wait(&self, wait: Duration) {
        self.inner.lock().unwrap().queue_seconds.observe(wait.as_secs_f64());
    }

    fn trim(m: &mut Inner, now: Instant) {
        while m.latency.front().is_some_and(|(t, _)| now.duration_since(*t) > LATENCY_WINDOW) {
            m.latency.pop_front();
        }
        while m.decisions.front().is_some_and(|(t, _)| now.duration_since(*t) > RATE_WINDOW) {
            m.decisions.pop_front();
        }
    }

    /// Latency quantiles (p50, p99) in seconds over the last [`LATENCY_WINDOW`]; NaN when there is no sample.
    pub fn latency_quantiles(&self) -> (f64, f64) {
        let mut m = self.inner.lock().unwrap();
        Self::trim(&mut m, Instant::now());
        let mut v: Vec<f64> = m.latency.iter().map(|(_, s)| *s).collect();
        v.sort_by(f64::total_cmp);
        (quantile(&v, 0.5), quantile(&v, 0.99))
    }

    /// Decisions per second over the last [`RATE_WINDOW`] (or since start, if shorter).
    pub fn decisions_per_second(&self) -> f64 {
        let now = Instant::now();
        let mut m = self.inner.lock().unwrap();
        Self::trim(&mut m, now);
        let window = now.duration_since(self.started).min(RATE_WINDOW).as_secs_f64().max(1e-3);
        m.decisions.iter().map(|(_, n)| *n).sum::<u64>() as f64 / window
    }

    /// Record the engine's static facts (`DecisionEngine::info`) for `/metrics`.
    pub fn set_engine_info(&self, info: Vec<(&'static str, String)>) {
        *self.engine_info.lock().unwrap() = info;
    }

    pub fn batches_total(&self) -> u64 {
        self.inner.lock().unwrap().batches_total
    }

    /// Prometheus text exposition (format 0.0.4).
    pub fn render(&self, queue_depth: usize, queue_capacity: usize, token_budget: usize) -> String {
        let (p50, p99) = self.latency_quantiles();
        let rate = self.decisions_per_second();
        let m = self.inner.lock().unwrap();
        let mut out = String::new();
        let gauge =|out: &mut String, name: &str, help: &str, v: f64| {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
        };
        gauge(&mut out, "sysone_queue_depth", "Requests admitted and waiting for a batch.", queue_depth as f64);
        gauge(&mut out, "sysone_queue_capacity", "Admission queue capacity (503 beyond it).", queue_capacity as f64);
        gauge(&mut out, "sysone_token_budget", "Packed tokens per engine step.", token_budget as f64);
        gauge(&mut out, "sysone_decisions_per_second", "Questions answered per second over the last 10 s.", rate);
        gauge(&mut out, "sysone_uptime_seconds", "Seconds since the server started.", self.started.elapsed().as_secs_f64());
        let info = self.engine_info.lock().unwrap();
        if !info.is_empty() {
            let labels: Vec<String> =
                info.iter().map(|(k, v)| format!("{k}=\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\""))).collect();
            let _ = writeln!(out, "# HELP sysone_engine_info Engine configuration (constant 1; the labels carry it).");
            let _ = writeln!(out, "# TYPE sysone_engine_info gauge\nsysone_engine_info{{{}}} 1", labels.join(","));
        }
        drop(info);
        let counter = |out: &mut String, name: &str, help: &str, v: f64| {
            let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}");
        };
        counter(&mut out, "sysone_decisions_total", "Questions answered.", m.decisions_total as f64);
        counter(&mut out, "sysone_branches_total", "Branches (answer rows) scored.", m.branches_total as f64);
        counter(&mut out, "sysone_batches_total", "Engine steps.", m.batches_total as f64);
        counter(&mut out, "sysone_batch_tokens_total", "Tokens packed into engine steps.", m.batch_tokens_total as f64);
        counter(&mut out, "sysone_bucket_tokens_total", "Bucket (padded) tokens of engine steps that report them.",
                m.bucket_tokens_total as f64);
        counter(&mut out, "sysone_bucket_batch_tokens_total", "Packed tokens of the steps counted in sysone_bucket_tokens_total.",
                m.bucket_batch_tokens_total as f64);
        let busy = m.engine_busy_seconds + m.running_since.map_or(0.0, |t| t.elapsed().as_secs_f64());
        counter(&mut out, "sysone_engine_busy_seconds_total", "Seconds the engine spent in steps (the running one so far).", busy);
        counter(&mut out, "sysone_rejected_total", "Requests refused with 503 (overload).", m.rejected_full as f64);
        let _ = writeln!(out, "# HELP sysone_overload_total Requests refused with 503, by the resource that was full.");
        let _ = writeln!(out, "# TYPE sysone_overload_total counter");
        for reason in ["queue", "cpu_queue", "body_memory"] {
            let n = m.overload.get(reason).copied().unwrap_or(0);
            let _ = writeln!(out, "sysone_overload_total{{reason=\"{reason}\"}} {n}");
        }
        let _ = writeln!(out, "# HELP sysone_requests_total HTTP responses by endpoint and status.");
        let _ = writeln!(out, "# TYPE sysone_requests_total counter");
        for ((endpoint, status), n) in &m.requests {
            let _ = writeln!(out, "sysone_requests_total{{endpoint=\"{endpoint}\",status=\"{status}\"}} {n}");
        }
        m.batch_tokens.render(&mut out, "sysone_batch_tokens", "Packed tokens per engine step.");
        m.batch_requests.render(&mut out, "sysone_batch_requests", "Requests per engine step.");
        m.step_seconds.render(&mut out, "sysone_step_seconds", "Engine step wall time.");
        m.queue_seconds.render(&mut out, "sysone_queue_wait_seconds", "Admission to engine step start.");
        let name = "sysone_request_latency_seconds";
        let _ = writeln!(out, "# HELP {name} End-to-end latency of answered requests (quantiles over the last 60 s).");
        let _ = writeln!(out, "# TYPE {name} summary");
        let _ = writeln!(out, "{name}{{quantile=\"0.5\"}} {p50}");
        let _ = writeln!(out, "{name}{{quantile=\"0.99\"}} {p99}");
        let _ = writeln!(out, "{name}_sum {}", m.latency_sum);
        let _ = writeln!(out, "{name}_count {}", m.latency_count);
        out
    }
}

/// The HTTP front's own state, rendered after [`Metrics::render`] by the server.
#[derive(Clone, Copy, Debug, Default)]
pub struct Front {
    pub cpu_queue_depth: usize,
    pub cpu_queue_capacity: usize,
    /// Pool tasks skipped because their client had disconnected.
    pub skipped: u64,
    pub body_bytes: usize,
    pub body_capacity: usize,
}

pub fn render_front(out: &mut String, f: &Front) {
    let mut metric = |name: &str, kind: &str, help: &str, v: f64| {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
    };
    metric("sysone_cpu_queue_depth", "gauge", "New requests waiting for a CPU worker.", f.cpu_queue_depth as f64);
    metric("sysone_cpu_queue_capacity", "gauge", "CPU new-work queue capacity (503 beyond it).", f.cpu_queue_capacity as f64);
    metric("sysone_skipped_total", "counter", "CPU tasks skipped because the client had disconnected.", f.skipped as f64);
    metric("sysone_inflight_body_bytes", "gauge", "Request body bytes held by in-flight requests.", f.body_bytes as f64);
    metric("sysone_inflight_body_capacity_bytes", "gauge", "Cap on in-flight request body bytes (503 beyond it).", f.body_capacity as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_are_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(quantile(&v, 0.5), 50.0);
        assert_eq!(quantile(&v, 0.99), 99.0);
        assert_eq!(quantile(&v[..1], 0.99), 1.0);
        assert!(quantile(&[], 0.5).is_nan());
    }

    fn busy(m: &Metrics) -> f64 {
        let t = m.render(0, 1, 1);
        t.lines().find_map(|l| l.strip_prefix("sysone_engine_busy_seconds_total ")).unwrap().parse().unwrap()
    }

    /// The busy counter grows while a step runs, so busy time over a scrape window never exceeds the window
    /// (loadgen's `server_engine_busy_frac` was > 1 when a whole step landed at its end).
    #[test]
    fn busy_time_counts_the_running_step() {
        let m = Metrics::new();
        let t0 = Instant::now();
        m.record_step_start(t0);
        std::thread::sleep(Duration::from_millis(30));
        let (scrape, during) = (Instant::now(), busy(&m));
        assert!(during >= 0.03 && during <= t0.elapsed().as_secs_f64(), "{during}");
        std::thread::sleep(Duration::from_millis(20));
        m.record_batch(10, 1, 1, t0.elapsed());
        let after = busy(&m);
        assert!(after - during <= scrape.elapsed().as_secs_f64() + 1e-9 && after >= 0.05, "{during} {after}");
        // Without a step start (old callers), the step's duration is counted at its end.
        m.record_batch(10, 1, 1, Duration::from_millis(100));
        assert!((busy(&m) - after - 0.1).abs() < 1e-9);
        m.record_bucket(10, 16);
        assert!(m.render(0, 1, 1).contains("sysone_bucket_tokens_total 16\n"));
    }

    #[test]
    fn renders_prometheus_text() {
        let m = Metrics::new();
        m.record_response("/v1/systemone", 200);
        m.record_response("/v1/systemone", 503);
        m.record_overload("cpu_queue");
        m.record_decided(Duration::from_millis(10), 3);
        m.record_decided(Duration::from_millis(30), 1);
        m.record_batch(1000, 2, 4, Duration::from_millis(7));
        m.set_engine_info(vec![("engine", "bf16".into()), ("execution_mode", "exact_eager".into())]);
        let mut t = m.render(5, 64, 16384);
        render_front(&mut t, &Front { cpu_queue_depth: 3, skipped: 2, ..Default::default() });
        assert!(t.contains("sysone_overload_total{reason=\"cpu_queue\"} 1\n"));
        assert!(t.contains("sysone_overload_total{reason=\"queue\"} 0\n"));
        assert!(t.contains("sysone_cpu_queue_depth 3\n"));
        assert!(t.contains("sysone_skipped_total 2\n"));
        assert!(t.contains("sysone_queue_depth 5\n"));
        assert!(t.contains("sysone_requests_total{endpoint=\"/v1/systemone\",status=\"503\"} 1\n"));
        assert!(t.contains("sysone_rejected_total 1\n"));
        assert!(t.contains("sysone_decisions_total 4\n"));
        assert!(t.contains("sysone_batch_tokens_bucket{le=\"1024\"} 1\n"));
        assert!(t.contains("sysone_batch_tokens_bucket{le=\"512\"} 0\n"));
        assert!(t.contains("sysone_request_latency_seconds{quantile=\"0.5\"} 0.01\n"));
        assert!(t.contains("sysone_request_latency_seconds{quantile=\"0.99\"} 0.03\n"));
        assert!(t.contains("sysone_engine_info{engine=\"bf16\",execution_mode=\"exact_eager\"} 1\n"));
        assert!(m.decisions_per_second() > 0.0);
        // Every non-comment line is `name[{labels}] value`.
        for line in t.lines().filter(|l| !l.starts_with('#')) {
            let (_, v) = line.rsplit_once(' ').unwrap();
            assert!(v.parse::<f64>().is_ok() || v == "NaN", "{line}");
        }
    }
}
