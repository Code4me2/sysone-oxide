//! Closed-loop load generator for `/v1/systemone` (DESIGN.md §10). `--concurrency` clients each keep one
//! keep-alive connection and send the next request as soon as the previous one returns. Reports requests/s,
//! decisions/s, latency quantiles and status counts, plus the server's batch statistics from `/metrics`, as one
//! JSON line on stdout: also the p50 of the responses' `qev.timing` parts (`queue_s` waiting for a batch,
//! `forward_s` the engine step) and the server's 503s by reason. W6 extends it into the sweep (state length × questions × concurrency) and the baselines.
//!
//!     cargo run --release --bin loadgen -- --addr 127.0.0.1:8000 --concurrency 64 --state-tokens 1000 \
//!         --questions 8 --duration-s 60

use anyhow::{Context, Result, ensure};
use clap::Parser;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use sysone::api::client::Client;
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
    /// Approximate state length in tokens (about one token per word of the synthetic state).
    #[arg(long, default_value_t = 1000)]
    state_tokens: usize,
    /// Questions per request (cycling noul, choice with 5 options, score with 4 levels).
    #[arg(long, default_value_t = 3)]
    questions: usize,
    /// `?debias=` on every request.
    #[arg(long, default_value_t = 1)]
    debias: i64,
    /// Measured duration after warmup.
    #[arg(long, default_value_t = 30.0)]
    duration_s: f64,
    #[arg(long, default_value_t = 3.0)]
    warmup_s: f64,
    /// Distinct request bodies (each client cycles through them from its own offset).
    #[arg(long, default_value_t = 64)]
    distinct: usize,
}

const VOCAB: [&str; 16] = [
    "order", "refund", "late", "billing", "error", "customer", "angry", "shipment", "account", "card", "retry", "ok",
    "package", "warehouse", "address", "invoice",
];

fn body(i: usize, a: &Args) -> String {
    let mut x = (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    let words: Vec<&str> = (0..a.state_tokens)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            VOCAB[(x % VOCAB.len() as u64) as usize]
        })
        .collect();
    let questions: Vec<String> = (0..a.questions)
        .map(|q| match q % 3 {
            0 => format!(r#""q{q}": {{"type": "noul", "instructions": "Should case {i} be escalated (check {q})?"}}"#),
            1 => format!(
                r#""q{q}": {{"type": "choice", "instructions": "Route case {i}", "criteria": {{"billing": "Billing team", "tech": "Technical support", "ship": "Shipping", "fraud": "Fraud review", "other": null}}}}"#
            ),
            _ => format!(
                r#""q{q}": {{"type": "score", "instructions": "Urgency of case {i}", "criteria": ["low", "medium", "high", "critical"]}}"#
            ),
        })
        .collect();
    format!(r#"{{"state": {{"ticket": "{}", "case": {i}}}, "questions": {{{}}}}}"#, words.join(" "), questions.join(", "))
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
}

async fn client(id: usize, a: Args, bodies: std::sync::Arc<Vec<String>>, start: Instant) -> Result<Tally> {
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
        *t.status.entry(r.status.as_u16()).or_default() += 1;
        if r.status == 200 {
            t.latencies.push(lat);
            t.decisions += a.questions as u64;
            let v = sysone::frontend::pyjson::parse(&r.body).context("response is not JSON")?;
            if let Some(sysone::frontend::pyjson::Json::Int(n)) = v.get("usage").and_then(|u| u.get("input_tokens")) {
                t.input_tokens += n.parse::<u64>().unwrap_or(0);
            }
            let timing = v.get("qev").and_then(|q| q.get("timing"));
            for (key, out) in [("queue_s", &mut t.queue_s), ("forward_s", &mut t.forward_s)] {
                if let Some(sysone::frontend::pyjson::Json::Float(x)) = timing.and_then(|tm| tm.get(key)) {
                    out.push(*x);
                }
            }
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
    let bodies = std::sync::Arc::new((0..a.distinct).map(|i| body(i, &a)).collect::<Vec<_>>());
    let scrape = || async {
        let mut c = Client::connect(&a.addr).await?;
        anyhow::Ok(c.get("/metrics").await?.text())
    };
    scrape().await.context("scraping /metrics")?;
    let start = Instant::now();
    let tasks: Vec<_> = (0..a.concurrency).map(|i| tokio::spawn(client(i, a.clone(), bodies.clone(), start))).collect();
    // Server counters over the measured window only (from the end of warmup).
    tokio::time::sleep(Duration::from_secs_f64(a.warmup_s)).await;
    let before = scrape().await.context("scraping /metrics after warmup")?;
    let mut all = Tally::default();
    for t in tasks {
        let t = t.await??;
        all.latencies.extend(t.latencies);
        all.queue_s.extend(t.queue_s);
        all.forward_s.extend(t.forward_s);
        for (s, n) in t.status {
            *all.status.entry(s).or_default() += n;
        }
        all.decisions += t.decisions;
        all.input_tokens += t.input_tokens;
    }
    let after = scrape().await.context("scraping /metrics after the run")?;
    let delta = |n: &str| Some(metric(&after, n)? - metric(&before, n)?);
    all.latencies.sort_by(f64::total_cmp);
    all.queue_s.sort_by(f64::total_cmp);
    all.forward_s.sort_by(f64::total_cmp);
    let ok = all.latencies.len() as f64;
    let secs = a.duration_s;
    let batches = delta("sysone_batches_total").unwrap_or(f64::NAN);
    let batch_tokens = delta("sysone_batch_tokens_total").unwrap_or(f64::NAN);
    let busy = delta("sysone_engine_busy_seconds_total").unwrap_or(f64::NAN);
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
        "state_tokens": a.state_tokens,
        "questions": a.questions,
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
        "server_engine_busy_frac": num(busy / secs),
    });
    println!("{report}");
    Ok(())
}
