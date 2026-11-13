use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use sysone::api::query::py_int;
use sysone::api::server::{self, AppState, ServerInfo, ServerOptions};
use sysone::contract::{DecisionBatch, DecisionEngine, MassReadout, MockEngine, SlotLogits};
use sysone::engine::config::ModelConfig;
use sysone::frontend::{Frontend, Limits};
use sysone::sched::{EngineFactory, SchedConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EngineKind {
    /// The deterministic stand-in (`contract::MockEngine`): real frontend and scheduler, fake logits.
    Mock,
}

#[derive(Parser, Debug)]
#[command(version, about = "System One decision-model server (drop-in for qev's /v1/systemone)")]
struct Args {
    /// Model directory (config.json, model.safetensors, tokenizer.json, chat_template.jinja).
    #[arg(long)]
    model: PathBuf,
    /// Address to listen on (qev: --host 0.0.0.0 --port 8000).
    #[arg(long, default_value = "0.0.0.0:8000")]
    listen: String,
    /// Model name reported by the API. Default: the last component of the model directory after resolving it
    /// (`canonicalize`, so symlinks and `..` are followed).
    ///
    /// qev's `serve` (cli.py) instead derives the default from the `--model` *argument string*: it splits it on
    /// "/" and, if that gives more than two parts, joins all parts but the first with "-"; otherwise it takes
    /// `Path(arg).name`. So both agree for `elephant-1`, `./elephant-1` and `baselabs/elephant-1`, but differ for
    /// absolute or deeper paths (qev: `/home/u/m/elephant-1` → `home-u-m-elephant-1`, `a/b/c` → `b-c`), for a
    /// trailing slash (qev: `m/elephant-1/` → `elephant-1-`), and for symlinks (qev keeps the link's name).
    /// Pass `--name` to pin the name a client expects.
    #[arg(long)]
    name: Option<String>,
    /// Engine. Only the mock exists until the real engine lands (DESIGN.md §11, W4).
    #[arg(long, value_enum, default_value_t = EngineKind::Mock)]
    engine: EngineKind,
    /// Close a batch once it holds this many packed tokens (a larger request runs alone).
    #[arg(long, default_value_t = 16384)]
    token_budget: usize,
    /// Close a batch once its oldest request has waited this long.
    #[arg(long, default_value_t = 2000)]
    max_wait_us: u64,
    /// Requests the admission queue holds; more are refused with 503.
    #[arg(long, default_value_t = 1024)]
    queue_capacity: usize,
    /// CPU threads for rendering, tokenization and answer assembly (default: available cores minus 2; with
    /// --pin-cores, performance cores minus 2).
    #[arg(long)]
    cpu_workers: Option<usize>,
    /// New requests that may wait for a CPU thread; more are refused with 503 (default: 16 per CPU thread).
    #[arg(long)]
    cpu_queue_capacity: Option<usize>,
    /// Request body memory held by in-flight requests, in MiB; more are refused with 503.
    #[arg(long, default_value_t = 1024)]
    max_inflight_body_mb: usize,
    /// Pin the GPU worker, the batcher and the CPU threads to the performance cores (read from sysfs
    /// `cpu_capacity` or `cpuinfo_max_freq`; docs/PLATFORM.md P7). Does nothing where all cores look alike.
    /// Costs CPU-pool throughput when the frontend is the bottleneck: on GB10 with the zero-latency mock engine,
    /// 8 pinned workers served ~4.9k req/s of 1k tokens against ~7.6k for 18 unpinned ones (2026-11-13), far
    /// above what the real engine consumes.
    #[arg(long)]
    pin_cores: bool,
    /// Longest state plus one question accepted (qev default).
    #[arg(long, default_value_t = 32768)]
    max_branch_tokens: usize,
    /// Longest whole request accepted (qev default).
    #[arg(long, default_value_t = 65536)]
    max_request_tokens: usize,
    /// `debias` when a request has none (default: $QEV_DEBIAS as qev reads it, else 1).
    #[arg(long, allow_negative_numbers = true)]
    debias: Option<i64>,
    /// `fork_mode` reported in /health and the qev meta (sysone always runs prefix and branches in one packed
    /// forward, which qev calls "batched").
    #[arg(long, default_value = "batched")]
    fork_mode: String,
    /// Mock engine only: simulated fixed time per step (GB10 bf16 weight floor ≈ 57,000 us, docs/PHASE0.md).
    #[arg(long, default_value_t = 0)]
    mock_step_us: u64,
    /// Mock engine only: simulated time per packed token (GB10 bf16 compute-bound ≈ 143,000 ns/token at ~7k tok/s).
    #[arg(long, default_value_t = 0)]
    mock_ns_per_token: u64,
}

/// `MockEngine` that sleeps like a real step would, for scheduler and load-generator work without a GPU.
struct SimulatedEngine {
    inner: MockEngine,
    step: Duration,
    per_token: Duration,
}

impl DecisionEngine for SimulatedEngine {
    fn max_tokens_per_step(&self) -> usize {
        self.inner.max_tokens_per_step()
    }

    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
        let t = self.step + self.per_token * batch.num_tokens() as u32;
        if !t.is_zero() {
            std::thread::sleep(t);
        }
        self.inner.run(batch)
    }

    fn run_with_mass(&mut self, batch: &DecisionBatch) -> Result<MassReadout> {
        let logits = self.run(batch)?;
        let allowed_mass = self.inner.run_with_mass(batch)?.allowed_mass;
        Ok(MassReadout { logits, allowed_mass })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = ModelConfig::load(&args.model)?;
    let limits = Limits { max_branch_tokens: args.max_branch_tokens, max_request_tokens: args.max_request_tokens };
    let frontend = Arc::new(Frontend::load(&args.model, limits).context("loading the frontend")?);
    let served = match &args.name {
        Some(n) => n.clone(),
        None => args.model.canonicalize()?.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
    };
    ensure!(!served.is_empty(), "cannot derive a model name from {}; pass --name", args.model.display());
    let mut info = ServerInfo::new(&served, &cfg.model_type);
    info.fork_mode = args.fork_mode.clone();
    info.default_debias = match (args.debias, std::env::var("QEV_DEBIAS")) {
        (Some(d), _) => d,
        (None, Ok(v)) => py_int(&v).with_context(|| format!("QEV_DEBIAS={v:?} is not an integer"))?,
        (None, Err(_)) => 1,
    };

    // The engine must take the largest request the limits admit in one step (requests are never split).
    let step = args.token_budget.max(args.max_request_tokens);
    let engine: EngineFactory = match args.engine {
        EngineKind::Mock => {
            let (fixed, per_token) = (Duration::from_micros(args.mock_step_us), Duration::from_nanos(args.mock_ns_per_token));
            Box::new(move || {
                let inner = MockEngine { max_tokens: step };
                Ok(Box::new(SimulatedEngine { inner, step: fixed, per_token }) as Box<dyn DecisionEngine>)
            })
        }
    };
    let sched = SchedConfig {
        token_budget: args.token_budget,
        max_wait: Duration::from_micros(args.max_wait_us),
        queue_capacity: args.queue_capacity,
    };
    let pin_cores = if args.pin_cores {
        let cores = sysone::sched::affinity::performance_cores();
        match &cores {
            Some(c) => eprintln!("sysone: pinning the GPU worker, batcher and CPU threads to cores {c:?}"),
            None => eprintln!("sysone: --pin-cores: no performance/efficiency core split detected; not pinning"),
        }
        cores
    } else {
        None
    };
    let cpu = args.cpu_workers.unwrap_or_else(|| {
        let cores = pin_cores.as_ref().map(Vec::len);
        cores.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get())).saturating_sub(2).max(1)
    });
    let opts = ServerOptions {
        cpu_workers: cpu,
        cpu_queue_capacity: args.cpu_queue_capacity,
        max_inflight_body_bytes: args.max_inflight_body_mb.saturating_mul(1 << 20),
        pin_cores,
    };
    let state = AppState::start_with(frontend, info, sched, engine, opts)?;
    let policy = state.sched.policy();
    eprintln!(
        "sysone: serving {} ({} layers, engine {:?}) on {} — token budget {}, max wait {} us, queue {}, {} CPU workers",
        served,
        cfg.text_config.num_hidden_layers,
        args.engine,
        args.listen,
        policy.token_budget,
        args.max_wait_us,
        args.queue_capacity,
        cpu
    );
    if policy.max_tokens < args.max_request_tokens {
        eprintln!(
            "sysone: warning: the engine takes {} tokens per step but requests may hold {}; larger ones fail with 500",
            policy.max_tokens, args.max_request_tokens
        );
    }

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&args.listen).await.with_context(|| format!("binding {}", args.listen))?;
        let shutdown = async {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("sysone: shutting down");
        };
        server::serve(listener, state, shutdown).await?;
        Ok(())
    })
}
