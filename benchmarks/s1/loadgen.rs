//! Closed-loop load generator for `/v1/systemone` (DESIGN.md §10). `--concurrency` clients each keep one
//! keep-alive connection and send the next request as soon as the previous one returns. Reports requests/s,
//! decisions/s, latency quantiles and status counts, plus the server's batch statistics from `/metrics`, as one
//! JSON line on stdout: also the p50 of the responses' `qev.timing` parts (`queue_s` waiting for a batch,
//! `forward_s` the engine step), the server's 503s by reason, the engine's busy fraction and the padding of its
//! steps (from `sysone_bucket_tokens_total`).
//!
//! Two modes:
//! - synthetic (default): `--distinct` bodies of `--state-tokens` words and `--questions` questions
//!   (`sysone::bulk::synth::body`), for `--duration-s` after `--warmup-s`;
//! - `--corpus FILE`: every job of a bulk job file (`sysone bulk` input format) sent once, then stop; the rates
//!   are over the time from the first request to the last response. `--responses FILE` writes each response as a
//!   bulk result line (`{"id", "status", "body"}`), for comparing the server with `sysone bulk` (docs/BULK.md).
//!
//! ```text
//! cargo run --release --bin loadgen -- --addr 127.0.0.1:8000 --concurrency 64 --state-tokens 1000 \
//!     --questions 8 --duration-s 60
//! ./target/release/loadgen --addr 127.0.0.1:8000 --concurrency 256 --corpus corpus.jsonl
//! ```

use anyhow::{Context, Result, ensure};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use sysone::api::client::Client;
use sysone::bulk::job::{Line, parse_line, result_line};
use sysone::frontend::pyjson::{self, Json};
use sysone::sched::metrics::quantile;

#[derive(Parser, Debug, Clone)]
#[command(about = "Load generator for /v1/systemone")]
struct Args {
    /// Server address (host:port).
    #[arg(long, default_value = "127.0.0.1:8000")]
    addr: String,
    /// Concurrent clients (one connection each).
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
    /// State length in words (about one token each).
    #[arg(long, default_value_t = 1000)]
    state_tokens: usize,
    /// Questions per request (cycling noul, choice with 5 options, score with 4 levels).
    #[arg(long, default_value_t = 3)]
    questions: usize,
    /// `?debias=` on every request (synthetic mode; corpus jobs use their own `query`, else this).
    #[arg(long, default_value_t = 1)]
    debias: i64,
    /// Measured duration after warmup (synthetic mode).
    #[arg(long, default_value_t = 30.0)]
    duration_s: f64,
    #[arg(long, default_value_t = 3.0)]
    warmup_s: f64,
    /// Distinct request bodies (each client cycles through them from its own offset).
    #[arg(long, default_value_t = 64)]
    distinct: usize,
    /// Send every job of this bulk job file once (corpus mode).
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// Corpus mode: write every response as a bulk result line here.
    #[arg(long)]
    responses: Option<PathBuf>,
}

#[derive(Default)]
struct Tally {
    latencies: Vec<f64>,
    /// `qev.timing` `queue_s` and `forward_s` of each 200.
    queue_s: Vec<f64>,
    forward_s: Vec<f64>,
    status: BTreeMap<u16, u64>,
    decisions: u64,
    input_tokens: u64,
    /// Corpus mode: result lines.
    lines: Vec<u8>,
}

impl Tally {
    fn record(&mut self, status: u16, lat: f64, body: &[u8]) -> Result<()> {
        *self.status.entry(status).or_default() += 1;
        if status != 200 {
            return Ok(());
        }
        self.latencies.push(lat);
        let v = pyjson::parse(body).context("response is not JSON")?;
        let usage = v.get("usage");
        for (key, out) in [("input_tokens", &mut self.input_tokens), ("output_tokens", &mut self.decisions)] {
            if let Some(Json::Int(n)) = usage.and_then(|u| u.get(key)) {
                *out += n.parse::<u64>().unwrap_or(0);
            }
        }
        let timing = v.get("qev").and_then(|q| q.get("timing"));
        for (key, out) in [("queue_s", &mut self.queue_s), ("forward_s", &mut self.forward_s)] {
            if let Some(Json::Float(x)) = timing.and_then(|tm| tm.get(key)) {
                out.push(*x);
            }
        }
        Ok(())
    }

    fn merge(&mut self, t: Tally) {
        self.latencies.extend(t.latencies);
        self.queue_s.extend(t.queue_s);
        self.forward_s.extend(t.forward_s);
        for (s, n) in t.status {
            *self.status.entry(s).or_default() += n;
        }
        self.decisions += t.decisions;
        self.input_tokens += t.input_tokens;
        self.lines.extend(t.lines);
    }
}

async fn client(id: usize, a: Args, bodies: Arc<Vec<String>>, start: Instant) -> Result<Tally> {
    let mut c = Client::connect(&a.addr).await?;
    let path = format!("/v1/systemone?debias={}", a.debias);
    let warm_end = start + Duration::from_secs_f64(a.warmup_s);
    let end = warm_end + Duration::from_secs_f64(a.duration_s);
    let mut t = Tally::default();
    let mut k = id;
    while Instant::now() < end {
        let b = bodies[k % bodies.len()].clone();
        k += 1;
        let t0 = Instant::now();
        let r = c.post(&path, b).await?;
        let lat = t0.elapsed().as_secs_f64();
        if t0 < warm_end {
            continue;
        }
        t.record(r.status.as_u16(), lat, &r.body)?;
    }
    Ok(t)
}

/// A corpus job: its id, the path with its query, and the raw body bytes.
struct CorpusJob {
    id: Json,
    path: String,
    body: Vec<u8>,
}

fn load_corpus(path: &PathBuf, debias: i64) -> Result<Vec<CorpusJob>> {
    let text = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut jobs = Vec::new();
    for (i, l) in text.split(|&b| b == b'\n').enumerate() {
        match parse_line(i as u64 + 1, l) {
            Line::Job(j) => {
                // The raw body bytes: `body_text` as given, `body` re-serialised compactly (same JSON value).
                let raw = pyjson::parse(l)?;
                let body = match (raw.get("body_text"), raw.get("body")) {
                    (Some(Json::Str(t)), _) => t.clone().into_bytes(),
                    (_, Some(b)) => b.dumps_compact().map_err(|e| anyhow::anyhow!(e))?.into_bytes(),
                    _ => continue,
                };
                let query = j.query.unwrap_or_else(|| format!("debias={debias}"));
                jobs.push(CorpusJob { id: j.id, path: format!("/v1/systemone?{query}"), body });
            }
            Line::Blank => {}
            Line::Bad { line, msg } => anyhow::bail!("{} line {line}: {msg}", path.display()),
        }
    }
    Ok(jobs)
}

async fn corpus_client(a: Args, jobs: Arc<Vec<CorpusJob>>, next: Arc<AtomicUsize>, keep: bool) -> Result<Tally> {
    let mut c = Client::connect(&a.addr).await?;
    let mut t = Tally::default();
    loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        let Some(job) = jobs.get(i) else { break };
        let t0 = Instant::now();
        let r = c.post(&job.path, job.body.clone()).await?;
        let lat = t0.elapsed().as_secs_f64();
        t.record(r.status.as_u16(), lat, &r.body)?;
        if keep {
            // Non-JSON bodies (Starlette's plain-text 500) become `{"detail": text}`, as `:batch` reports them.
            let body = pyjson::parse(&r.body).unwrap_or_else(|_| Json::Object(vec![("detail".into(), Json::str(r.text()))]));
            t.lines.extend(result_line(&job.id, r.status.as_u16(), &body).map_err(|e| anyhow::anyhow!(e))?);
        }
    }
    Ok(t)
}

/// One value from a Prometheus text page.
fn metric(text: &str, name: &str) -> Option<f64> {
    text.lines().find_map(|l| l.strip_prefix(name).and_then(|r| r.strip_prefix(' ')).and_then(|v| v.parse().ok()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let a = Args::parse();
    ensure!(a.concurrency > 0 && a.questions > 0 && a.distinct > 0, "concurrency, questions and distinct must be > 0");
    let scrape = || async {
        let mut c = Client::connect(&a.addr).await?;
        anyhow::Ok(c.get("/metrics").await?.text())
    };
    scrape().await.context("scraping /metrics")?;
    let mut all = Tally::default();
    let (before, before_at, secs, corpus_jobs);
    if let Some(path) = &a.corpus {
        let jobs = Arc::new(load_corpus(path, a.debias)?);
        corpus_jobs = Some(jobs.len());
        before = scrape().await.context("scraping /metrics before the run")?;
        before_at = Instant::now();
        let next = Arc::new(AtomicUsize::new(0));
        let tasks: Vec<_> = (0..a.concurrency)
            .map(|_| tokio::spawn(corpus_client(a.clone(), jobs.clone(), next.clone(), a.responses.is_some())))
            .collect();
        for t in tasks {
            all.merge(t.await??);
        }
        secs = before_at.elapsed().as_secs_f64();
        if let Some(p) = &a.responses {
            std::fs::write(p, &all.lines).with_context(|| format!("writing {}", p.display()))?;
        }
    } else {
        corpus_jobs = None;
        let bodies = Arc::new((0..a.distinct).map(|i| sysone::bulk::synth::body(i, a.state_tokens, a.questions)).collect::<Vec<_>>());
        let start = Instant::now();
        let tasks: Vec<_> = (0..a.concurrency).map(|i| tokio::spawn(client(i, a.clone(), bodies.clone(), start))).collect();
        // Server counters over the measured window only (from the end of warmup).
        tokio::time::sleep(Duration::from_secs_f64(a.warmup_s)).await;
        before = scrape().await.context("scraping /metrics after warmup")?;
        before_at = Instant::now();
        for t in tasks {
            all.merge(t.await??);
        }
        secs = a.duration_s;
    }
    let after = scrape().await.context("scraping /metrics after the run")?;
    // The server counters are deltas between two scrapes; divide them by the time between those scrapes (the
    // clients' last requests can end after `duration_s`, which made the busy fraction exceed 1).
    let scraped_s = before_at.elapsed().as_secs_f64();
    let delta = |n: &str| Some(metric(&after, n)? - metric(&before, n)?);
    all.latencies.sort_by(f64::total_cmp);
    all.queue_s.sort_by(f64::total_cmp);
    all.forward_s.sort_by(f64::total_cmp);
    let ok = all.latencies.len() as f64;
    let batches = delta("sysone_batches_total").unwrap_or(f64::NAN);
    let batch_tokens = delta("sysone_batch_tokens_total").unwrap_or(f64::NAN);
    let busy = delta("sysone_engine_busy_seconds_total").unwrap_or(f64::NAN);
    let bucket = delta("sysone_bucket_tokens_total").unwrap_or(f64::NAN);
    let bucket_batch = delta("sysone_bucket_batch_tokens_total").unwrap_or(f64::NAN);
    // Non-finite values (no samples, no server metrics) become null.
    let num = |x: f64| if x.is_finite() { serde_json::json!(x) } else { serde_json::Value::Null };
    let status: serde_json::Map<String, serde_json::Value> =
        all.status.iter().map(|(s, n)| (s.to_string(), serde_json::json!(n))).collect();
    let overload: serde_json::Map<String, serde_json::Value> = ["queue", "cpu_queue", "body_memory"]
        .iter()
        .map(|r| (r.to_string(), num(delta(&format!("sysone_overload_total{{reason=\"{r}\"}}")).unwrap_or(f64::NAN))))
        .collect();
    let report = serde_json::json!({
        "addr": a.addr,
        "concurrency": a.concurrency,
        "corpus": a.corpus.as_ref().map(|p| p.display().to_string()),
        "corpus_jobs": corpus_jobs,
        "state_tokens": if a.corpus.is_some() { serde_json::Value::Null } else { serde_json::json!(a.state_tokens) },
        "questions": if a.corpus.is_some() { serde_json::Value::Null } else { serde_json::json!(a.questions) },
        "debias": a.debias,
        "duration_s": secs,
        "requests_per_s": num(ok / secs),
        "decisions_per_s": num(all.decisions as f64 / secs),
        "input_tokens_per_s": num(all.input_tokens as f64 / secs),
        "latency_p50_ms": num(quantile(&all.latencies, 0.5) * 1e3),
        "latency_p99_ms": num(quantile(&all.latencies, 0.99) * 1e3),
        "latency_max_ms": num(all.latencies.last().copied().unwrap_or(f64::NAN) * 1e3),
        "status": status,
        "queue_p50_ms": num(quantile(&all.queue_s, 0.5) * 1e3),
        "forward_p50_ms": num(quantile(&all.forward_s, 0.5) * 1e3),
        "server_overload": overload,
        "server_batches": num(batches),
        "server_mean_batch_tokens": num(batch_tokens / batches),
        "server_window_s": scraped_s,
        "server_engine_busy_frac": num(busy / scraped_s),
        "server_padding_frac": num(1.0 - bucket_batch / bucket),
    });
    println!("{report}");
    Ok(())
}
