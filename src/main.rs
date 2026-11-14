//! `sysone`: the server binary. Parses the CLI, loads the model config and the [frontend], builds the engine
//! factory (`--engine mock`, `bf16` or `nvfp4`), starts the scheduler and CPU pool (`AppState::start_with`) and
//! serves HTTP until Ctrl-C.
//!
//! The engine is built inside the scheduler's GPU worker thread (the factory closure runs there), because a CUDA
//! context belongs to the thread that creates it. `--max-step-tokens` must cover both `--token-budget` and
//! `--max-request-tokens`: requests are never split across steps. `--engine mock` wraps `MockEngine` in
//! `SimulatedEngine`, which can sleep like a real step (`--mock-step-us`, `--mock-ns-per-token`) for scheduler
//! and load-generator work without a GPU.
//!
//! See docs/ARCHITECTURE.md §2 for the path of a request, and `sysone --help` for every flag.
//!
//! [frontend]: sysone::frontend

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use sysone::api::server::{self, AppState, ServerInfo, ServerOptions};
use sysone::contract::{DecisionBatch, DecisionEngine, MassReadout, MockEngine, SlotLogits};
use sysone::engine::config::{ModelConfig, WeightFormat};
use sysone::engine::gpu::{EngineOptions, GpuEngine, parse_gdn_backend};
use sysone::frontend::{Frontend, Limits};
use sysone::sched::{EngineFactory, SchedConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EngineKind {
    /// The deterministic stand-in (`contract::MockEngine`): real frontend and scheduler, fake logits.
    Mock,
    /// The bf16 GPU engine (`engine::gpu::GpuEngine`) on device 0; needs a bf16 checkpoint.
    Bf16,
    /// The GPU engine with the MLP projections on NVFP4 kernels (W4A4); needs an NVFP4-MLP checkpoint
    /// (`quantization_config.format = nvfp4-pack-quantized`, e.g. elephant-1-NVFP4-MLP).
    Nvfp4,
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
    /// Engine: `bf16` runs a bf16 checkpoint on the GPU, `nvfp4` an NVFP4-MLP checkpoint (MLP GEMMs on FP4 tensor
    /// cores); `mock` returns deterministic fake logits. The checkpoint must match the engine (checked at load).
    #[arg(long, value_enum, default_value_t = EngineKind::Mock)]
    engine: EngineKind,
    /// GPU engines: largest packed step (T bucket) in tokens; it must cover --max-request-tokens and
    /// --token-budget, since a request is never split across steps (default: the larger of the two).
    #[arg(long)]
    max_step_tokens: Option<usize>,
    /// GPU engines: run every step eagerly instead of replaying a CUDA graph per T bucket.
    #[arg(long)]
    no_cuda_graphs: bool,
    /// GPU engines: most requests (prefix segments) in one step; each needs 2 MiB of GDN state per step.
    #[arg(long, default_value_t = 256)]
    max_prefixes: usize,
    /// GPU engines: implementation of the GatedDeltaNet delta rule: `tile` (cuTile), `simt` (cuda-oxide SIMT
    /// kernels, docs/G1.md; needs a kernel artifact for the GPU's architecture, else the load fails), or `auto`
    /// (the architecture's validated default, see docs/E2.md). The choice is logged at startup and reported in
    /// `/metrics` (`sysone_engine_info`).
    #[arg(long, default_value = "auto", value_parser = ["auto", "tile", "simt"])]
    gdn_backend: String,
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
    /// Longest state plus one question accepted. Default: the checkpoint profile's (qev's 32,768 for bf16
    /// checkpoints; 16,384 for the NVFP4 checkpoint, its documented limit).
    #[arg(long)]
    max_branch_tokens: Option<usize>,
    /// Longest whole request accepted. Default: the checkpoint profile's (qev's 65,536).
    #[arg(long)]
    max_request_tokens: Option<usize>,
    /// nvfp4 engine: run the activation quantization as a standalone kernel instead of fusing it into the
    /// preceding norm / SiLU·mul (bit-identical; for benchmarks and diagnostics).
    #[arg(long)]
    nvfp4_unfused: bool,
    /// bf16 GEMM kernel choice: `pinned` (one cuBLASLt algorithm per weight shape at every batch size, so a
    /// request's results do not depend on what it is packed with; default) or `heuristic` (cuBLAS per call).
    #[arg(long, default_value = "pinned")]
    gemm: String,
    /// Compute the last layer for every token instead of only the answer rows (diagnostics).
    #[arg(long)]
    no_trim_last_layer: bool,
    /// nvfp4 engine: run SiLU·mul + quantize as its own kernel after a bf16 gate/up GEMM instead of in the GEMM's
    /// epilogue (bit-identical; for benchmarks and diagnostics).
    #[arg(long)]
    nvfp4_separate_silu: bool,
    /// nvfp4 engine: kernel of the MLP down projection's FP4 GEMM: `cutile` (default) or `cublaslt` (cuBLASLt's
    /// block-scaled FP4 GEMM, one pinned algorithm per weight shape, batch-invariant; docs/C1.md). Default:
    /// $SYSONE_FP4_GEMM, else cutile.
    #[arg(long)]
    fp4_gemm: Option<String>,
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
    // `sysone bulk ...`: the bulk tagging path (docs/BULK.md); everything else is the server.
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("bulk") {
        let code = sysone::bulk::cli::main(&argv[2..])?;
        std::process::exit(code);
    }
    let args = Args::parse();
    let cfg = ModelConfig::load(&args.model)?;
    let profile = cfg.profile()?;
    let limits = Limits {
        max_branch_tokens: args.max_branch_tokens.unwrap_or(profile.max_branch_tokens),
        max_request_tokens: args.max_request_tokens.unwrap_or(profile.max_request_tokens),
    };
    eprintln!(
        "sysone: {} checkpoint; limits {} tokens per branch, {} per request ({})",
        profile.weights.name(),
        limits.max_branch_tokens,
        limits.max_request_tokens,
        if args.max_branch_tokens.is_some() || args.max_request_tokens.is_some() { "flags" } else { profile.limits_source }
    );
    if limits.max_branch_tokens > profile.max_branch_tokens || limits.max_request_tokens > profile.max_request_tokens {
        eprintln!(
            "sysone: warning: limits above the checkpoint profile's ({} per branch, {} per request; {})",
            profile.max_branch_tokens, profile.max_request_tokens, profile.limits_source
        );
    }
    let frontend = Arc::new(Frontend::load(&args.model, limits).context("loading the frontend")?);
    let info = ServerInfo::for_cli(&args.model, &cfg.model_type, args.name.as_deref(), &args.fork_mode, args.debias)?;
    let served = info.served_name.clone();

    // The engine must take the largest request the limits admit in one step (requests are never split).
    let step = args.token_budget.max(limits.max_request_tokens);
    let engine: EngineFactory = match args.engine {
        EngineKind::Mock => {
            let (fixed, per_token) = (Duration::from_micros(args.mock_step_us), Duration::from_nanos(args.mock_ns_per_token));
            Box::new(move || {
                let inner = MockEngine { max_tokens: step };
                Ok(Box::new(SimulatedEngine { inner, step: fixed, per_token }) as Box<dyn DecisionEngine>)
            })
        }
        EngineKind::Bf16 | EngineKind::Nvfp4 => {
            let max_step = args.max_step_tokens.unwrap_or(step);
            ensure!(
                max_step >= step,
                "--max-step-tokens {max_step} is below the largest request the limits admit ({} tokens) or the token \
                 budget ({}); requests are never split across steps, so raise it or lower --max-request-tokens",
                limits.max_request_tokens,
                args.token_budget
            );
            let weights = Some(if args.engine == EngineKind::Nvfp4 { WeightFormat::Nvfp4 } else { WeightFormat::Bf16 });
            let opts = EngineOptions {
                buckets: EngineOptions::buckets_up_to(max_step),
                cuda_graphs: !args.no_cuda_graphs,
                max_prefixes: args.max_prefixes,
                gdn_backend: parse_gdn_backend(&args.gdn_backend)?,
                weights,
                nvfp4_fused: !args.nvfp4_unfused,
                gemm: args.gemm.parse()?,
                trim_last_layer: !args.no_trim_last_layer,
                nvfp4_swiglu: !args.nvfp4_separate_silu,
                fp4_gemm: args.fp4_gemm.as_deref().map(str::parse).transpose()?,
                ..Default::default()
            };
            let (model, slot_ids) = (args.model.clone(), frontend.slot_ids.clone());
            Box::new(move || {
                let e = GpuEngine::load(&model, &slot_ids, opts)?;
                Ok(Box::new(e) as Box<dyn DecisionEngine>)
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
    if policy.max_tokens < limits.max_request_tokens {
        eprintln!(
            "sysone: warning: the engine takes {} tokens per step but requests may hold {}; larger ones fail with 500",
            policy.max_tokens, limits.max_request_tokens
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
