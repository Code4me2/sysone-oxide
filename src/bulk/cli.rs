//! `sysone bulk ...`: argument parsing for the bulk subcommands (docs/BULK.md).

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use super::pack::{PackPolicy, PackStats, Size, Throughput, ladder, simulate_bulk, simulate_fifo};
use super::run::{RunOptions, Worker, WorkerStart};
use super::worker::{DEFAULT_MAX_STEP, EngineSpec};
use crate::api::server::ServerInfo;
use crate::contract::StepLimits;
use crate::engine::config::ModelConfig;
use crate::frontend::{Frontend, Limits};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EngineKind {
    Bf16,
    Mock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum WorkerMode {
    /// One worker process per GPU (started through scripts/gpu.sh, which takes the GPU's lock).
    Process,
    /// Workers as threads of this process (mock engine; or one GPU when this process holds its lock).
    Thread,
}

#[derive(Parser, Debug)]
#[command(name = "sysone bulk", about = "Tag a JSONL file of /v1/systemone requests at full throughput (docs/BULK.md)")]
pub struct RunArgs {
    /// Model directory (config.json, model.safetensors, tokenizer.json, chat_template.jinja).
    #[arg(long)]
    pub model: PathBuf,
    /// Jobs: one `{"id": ..., "body": {...}}` per line (also `body_text`, `query`).
    #[arg(long)]
    pub input: PathBuf,
    /// Results: one `{"id": ..., "status": ..., "body": ...}` per line, appended; ids already there are skipped.
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, value_enum, default_value_t = EngineKind::Bf16)]
    pub engine: EngineKind,
    /// GPUs (nvidia-smi indices or UUIDs, comma-separated), one worker each; `auto`: $CUDA_VISIBLE_DEVICES if set,
    /// else every GPU nvidia-smi lists. The same GPU twice runs two engines on it (only when this process already
    /// holds its lock, i.e. runs under scripts/gpu.sh).
    /// With `--engine mock`, `--gpus` starts one mock worker per listed GPU through gpu.sh (lock tests).
    #[arg(long)]
    pub gpus: Option<String>,
    /// Default: process for bf16, thread for mock.
    #[arg(long, value_enum)]
    pub workers: Option<WorkerMode>,
    /// scripts/gpu.sh, to start each GPU worker under its GPU's lock (default: $SYSONE_GPU_SH, else the
    /// repository's scripts/gpu.sh).
    #[arg(long)]
    pub gpu_sh: Option<PathBuf>,
    /// Planned requests the packer chooses each step from (more: fuller steps, more memory).
    #[arg(long, default_value_t = 1024)]
    pub window: usize,
    /// Steps queued per worker (one running, the rest waiting in its pipe).
    #[arg(long, default_value_t = 2)]
    pub depth: usize,
    /// While input is still coming, run a step only once the window fills the largest bucket this much.
    #[arg(long, default_value_t = 0.98)]
    pub min_fill: f64,
    /// Largest bucket (tokens per step); then cut to what fits the device's memory.
    #[arg(long)]
    pub max_step_tokens: Option<usize>,
    /// Buckets at powers of two and their midpoints (less padding in the last steps; one more graph capture each).
    #[arg(long)]
    pub fine_ladder: bool,
    /// Keep buckets whose activations fit this fraction of the device memory free after the weights.
    #[arg(long, default_value_t = 0.85)]
    pub memory_fraction: f64,
    /// Most requests (prefix segments) per step; 2 MiB of GDN state each.
    #[arg(long, default_value_t = 256)]
    pub max_prefixes: usize,
    #[arg(long, default_value = "auto", value_parser = ["auto", "tile", "simt"])]
    pub gdn_backend: String,
    #[arg(long)]
    pub no_cuda_graphs: bool,
    /// Platform profiles (`hwprobe --profile-out`); each GPU worker uses the one for its device name.
    #[arg(long)]
    pub profile: Vec<PathBuf>,
    /// GPU workers refuse to start while another process uses their GPU (for benchmarks).
    #[arg(long)]
    pub require_idle: bool,
    /// Query string for jobs without their own `query` (e.g. `debias=2`).
    #[arg(long)]
    pub query: Option<String>,
    /// Model name in responses (as `sysone --name`).
    #[arg(long)]
    pub name: Option<String>,
    /// Default debias (as `sysone --debias`; else $QEV_DEBIAS, else 1).
    #[arg(long, allow_negative_numbers = true)]
    pub debias: Option<i64>,
    #[arg(long, default_value = "batched")]
    pub fork_mode: String,
    #[arg(long, default_value_t = 32768)]
    pub max_branch_tokens: usize,
    #[arg(long, default_value_t = 65536)]
    pub max_request_tokens: usize,
    /// Planning threads (default: available cores minus 2).
    #[arg(long)]
    pub cpu_workers: Option<usize>,
    /// Seconds between progress lines on stderr (0: none).
    #[arg(long, default_value_t = 30.0)]
    pub progress_s: f64,
    /// Process at most this many jobs (not counting skipped ones).
    #[arg(long)]
    pub limit: Option<u64>,
    /// Also write the final summary JSON here.
    #[arg(long)]
    pub summary: Option<PathBuf>,
    /// Write one JSON line per step (worker, bucket, tokens, ids) here.
    #[arg(long)]
    pub trace: Option<PathBuf>,
    /// Mock engine: workers.
    #[arg(long, default_value_t = 1)]
    pub mock_workers: usize,
    /// Mock engine: simulated ns per bucket token, per worker (comma-separated, cycled).
    #[arg(long, default_value = "0")]
    pub mock_ns_per_token: String,
    /// Mock engine: largest bucket per worker (comma-separated, cycled; default --max-step-tokens or 32768).
    #[arg(long)]
    pub mock_max_tokens: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub mock_step_us: u64,
    /// Mock engine: worker k stops after this many steps, as if it crashed (comma-separated, cycled; 0 = never).
    #[arg(long)]
    pub mock_exit_after: Option<String>,
}

fn list<T: std::str::FromStr>(s: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    s.split(',').map(|x| x.trim().parse::<T>().map_err(|e| anyhow::anyhow!("{x:?}: {e}"))).collect()
}

fn default_gpu_sh() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SYSONE_GPU_SH") {
        return Some(PathBuf::from(p));
    }
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/gpu.sh");
    p.exists().then_some(p)
}

/// The GPUs to use: `--gpus` entries, or `auto`.
pub fn gpu_list(sel: &str) -> Result<Vec<String>> {
    if sel != "auto" {
        let v: Vec<String> = sel.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        ensure!(!v.is_empty(), "--gpus is empty");
        return Ok(v);
    }
    if let Ok(v) = std::env::var("CUDA_VISIBLE_DEVICES") {
        let v: Vec<String> = v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        ensure!(!v.is_empty(), "CUDA_VISIBLE_DEVICES is empty: no GPU selected");
        return Ok(v);
    }
    let smi = std::env::var("SYSONE_NVIDIA_SMI").unwrap_or_else(|_| "nvidia-smi".into());
    let out = std::process::Command::new(&smi)
        .args(["--query-gpu=index,uuid", "--format=csv,noheader"])
        .output()
        .with_context(|| format!("running {smi} to list the GPUs (or pass --gpus)"))?;
    ensure!(out.status.success(), "{smi} failed; pass --gpus");
    // "index, uuid" per line (the query gpu.sh makes); only the index is used.
    let v: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.split(',').next().map(|s| s.trim().to_string()))
        .filter(|l| !l.is_empty())
        .collect();
    ensure!(!v.is_empty(), "{smi} lists no GPU");
    Ok(v)
}

/// A GPU listed twice means two engines on it: allowed only when this process already holds the locks (it runs
/// under scripts/gpu.sh), because the second worker's gpu.sh would otherwise wait for the first one's lock forever.
fn check_gpus(gpus: &[String]) -> Result<()> {
    let held = std::env::var("SYSONE_GPU_HELD").unwrap_or_default();
    let mut sorted = gpus.to_vec();
    sorted.sort();
    sorted.dedup();
    ensure!(
        sorted.len() == gpus.len() || !held.is_empty(),
        "--gpus {} names a GPU twice: two engines on one GPU need this process to hold its lock already \
         (run it under scripts/gpu.sh)",
        gpus.join(",")
    );
    Ok(())
}

/// Worker names: `<prefix><gpu>`, with `.k` added for a GPU listed more than once.
fn gpu_name(prefix: &str, gpus: &[String], k: usize) -> String {
    let g = &gpus[k];
    if gpus.iter().filter(|x| *x == g).count() > 1 { format!("{prefix}{g}.{k}") } else { format!("{prefix}{g}") }
}

pub fn run_main(argv: &[String]) -> Result<i32> {
    let a = RunArgs::try_parse_from(std::iter::once("sysone bulk".to_string()).chain(argv.iter().cloned()))
        .unwrap_or_else(|e| e.exit());
    let cfg = ModelConfig::load(&a.model)?;
    let limits = Limits { max_branch_tokens: a.max_branch_tokens, max_request_tokens: a.max_request_tokens };
    let frontend = Arc::new(Frontend::load(&a.model, limits).context("loading the frontend")?);
    let info = ServerInfo::for_cli(&a.model, &cfg.model_type, a.name.as_deref(), &a.fork_mode, a.debias)?;
    let base = EngineSpec {
        engine: match a.engine {
            EngineKind::Bf16 => "bf16".into(),
            EngineKind::Mock => "mock".into(),
        },
        model: a.model.canonicalize()?,
        slot_ids: frontend.slot_ids.clone(),
        max_step_tokens: a.max_step_tokens,
        fine_ladder: a.fine_ladder,
        memory_fraction: a.memory_fraction,
        max_prefixes: a.max_prefixes,
        gdn_backend: a.gdn_backend.clone(),
        cuda_graphs: !a.no_cuda_graphs,
        profiles: a.profile.iter().map(|p| p.canonicalize()).collect::<std::io::Result<_>>()?,
        require_idle: a.require_idle,
        mock_step_us: a.mock_step_us,
        mock_ns_per_token: 0,
        mock_exit_after: None,
    };
    let exe = std::env::current_exe()?;
    let mut workers = Vec::new();
    match a.engine {
        EngineKind::Mock => {
            let mode = a.workers.unwrap_or(WorkerMode::Thread);
            let ns: Vec<u64> = list(&a.mock_ns_per_token)?;
            let max: Vec<usize> = match &a.mock_max_tokens {
                Some(m) => list(m)?,
                None => vec![a.max_step_tokens.unwrap_or(DEFAULT_MAX_STEP)],
            };
            let exits: Vec<u64> = match &a.mock_exit_after {
                Some(e) => list(e)?,
                None => vec![0],
            };
            ensure!(a.mock_workers > 0 && !ns.is_empty() && !max.is_empty(), "mock workers need at least one setting each");
            let gpus = a.gpus.as_deref().map(gpu_list).transpose()?;
            if let Some(g) = &gpus {
                check_gpus(g)?;
            }
            let n = gpus.as_ref().map_or(a.mock_workers, Vec::len);
            for i in 0..n {
                let mut spec = base.clone();
                spec.mock_ns_per_token = ns[i % ns.len()];
                spec.max_step_tokens = Some(max[i % max.len()]);
                spec.mock_exit_after = Some(exits[i % exits.len()]).filter(|&e| e > 0);
                let (name, start) = match &gpus {
                    // Mock workers placed like GPU workers: through gpu.sh, one per GPU, each under its lock.
                    Some(g) => {
                        let sh = a.gpu_sh.clone().or_else(default_gpu_sh).context("scripts/gpu.sh not found: pass --gpu-sh")?;
                        (gpu_name("mock-gpu", g, i), WorkerStart::Process(super::worker::command(&exe, Some(&sh), Some(&g[i]), &spec)?))
                    }
                    None => (
                        format!("mock{i}"),
                        match mode {
                            WorkerMode::Thread => WorkerStart::Thread(spec),
                            WorkerMode::Process => WorkerStart::Process(super::worker::command(&exe, None, None, &spec)?),
                        },
                    ),
                };
                workers.push(Worker { name, start });
            }
        }
        EngineKind::Bf16 => {
            let gpus = gpu_list(a.gpus.as_deref().unwrap_or("auto"))?;
            check_gpus(&gpus)?;
            match a.workers.unwrap_or(WorkerMode::Process) {
                WorkerMode::Thread => {
                    ensure!(gpus.len() == 1, "--workers thread runs one GPU engine in this process; use process workers for {}", gpus.len());
                    ensure!(
                        std::env::var_os("SYSONE_GPU_LOCKED").is_some(),
                        "--workers thread with the GPU engine: run this under scripts/gpu.sh, which takes the GPU's lock"
                    );
                    workers.push(Worker { name: format!("gpu{}", gpus[0]), start: WorkerStart::Thread(base.clone()) });
                }
                WorkerMode::Process => {
                    let sh = a.gpu_sh.clone().or_else(default_gpu_sh).context("scripts/gpu.sh not found: pass --gpu-sh")?;
                    for (k, g) in gpus.iter().enumerate() {
                        let cmd = super::worker::command(&exe, Some(&sh), Some(g), &base)?;
                        let name = gpu_name("gpu", &gpus, k);
                        workers.push(Worker { name, start: WorkerStart::Process(cmd) });
                    }
                }
            }
        }
    }
    let cpu = a.cpu_workers.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(2).max(1));
    let opts = RunOptions {
        input: a.input.clone(),
        output: a.output.clone(),
        workers,
        window: a.window,
        depth: a.depth,
        policy: PackPolicy { min_fill: a.min_fill },
        query: a.query.clone(),
        info,
        cpu_workers: cpu,
        progress: (a.progress_s > 0.0).then(|| Duration::from_secs_f64(a.progress_s)),
        limit: a.limit,
        trace: a.trace.clone(),
    };
    let summary = super::run::run(opts, frontend)?;
    let json = serde_json::to_string(&summary)?;
    println!("{json}");
    if let Some(p) = &a.summary {
        std::fs::write(p, json + "\n").with_context(|| format!("writing {}", p.display()))?;
    }
    eprintln!(
        "sysone bulk: {} ok, {} errors written, {} failed, {} unrunnable, {} bad lines, {} duplicate ids, {} skipped (done); \
         {:.2} docs/s, {:.1} decisions/s, {:.0} packed tok/s, padding {:.2}% over {:.1} s",
        summary.ok,
        summary.written.iter().filter(|(s, _)| **s != 200).map(|(_, n)| n).sum::<u64>(),
        summary.failed,
        summary.unrunnable,
        summary.bad_lines,
        summary.duplicate_ids,
        summary.skipped_done,
        summary.docs_per_s,
        summary.decisions_per_s,
        summary.packed_tok_per_s,
        summary.padding_frac * 100.0,
        summary.run_s
    );
    for w in &summary.workers {
        eprintln!(
            "  {}: {} ({} steps, {} requests, {:.1}% of tokens, {:.0} tok/s, busy {:.1}%, padding {:.2}%){}",
            w.name,
            w.device,
            w.steps,
            w.requests,
            w.token_share * 100.0,
            w.packed_tok_per_s,
            w.busy_frac * 100.0,
            w.padding_frac * 100.0,
            w.error.as_deref().map(|e| format!(" error: {e}")).unwrap_or_default()
        );
    }
    if let Some(why) = &summary.aborted {
        eprintln!("sysone bulk: stopped early: {why}; rerun the same command to resume");
        return Ok(1);
    }
    Ok(if summary.failed > 0 || summary.unrunnable > 0 { 1 } else { 0 })
}

#[derive(Parser, Debug)]
#[command(name = "sysone bulk gen", about = "Write a synthetic bulk corpus (loadgen's requests with a length mix)")]
pub struct GenArgs {
    #[arg(long, default_value_t = 2000)]
    pub docs: usize,
    /// State length range in words, log-uniform. The vocabulary tokenizes to about 2 tokens per word, so the
    /// default 150..4000 words gives states of about 300..8,000 tokens.
    #[arg(long, default_value_t = 150)]
    pub min_words: usize,
    #[arg(long, default_value_t = 4000)]
    pub max_words: usize,
    #[arg(long, default_value_t = 4)]
    pub min_questions: usize,
    #[arg(long, default_value_t = 16)]
    pub max_questions: usize,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    #[arg(long)]
    pub output: PathBuf,
}

pub fn gen_main(argv: &[String]) -> Result<i32> {
    use std::io::Write;
    let a = GenArgs::try_parse_from(std::iter::once("sysone bulk gen".to_string()).chain(argv.iter().cloned()))
        .unwrap_or_else(|e| e.exit());
    let spec = super::synth::CorpusSpec {
        docs: a.docs,
        min_words: a.min_words,
        max_words: a.max_words,
        min_questions: a.min_questions,
        max_questions: a.max_questions,
        seed: a.seed,
    };
    let mut f = std::io::BufWriter::new(std::fs::File::create(&a.output).with_context(|| format!("creating {}", a.output.display()))?);
    for i in 0..spec.docs {
        writeln!(f, "{}", super::synth::corpus_line(&spec, i))?;
    }
    f.flush()?;
    Ok(0)
}

#[derive(Parser, Debug)]
#[command(
    name = "sysone bulk simulate",
    about = "Plan a job file (CPU only) and compare packings: the server's FIFO batcher vs bulk windows and ladders"
)]
pub struct SimArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub input: PathBuf,
    /// Largest bucket(s) to evaluate.
    #[arg(long, default_value = "16384,32768")]
    pub max_step: String,
    /// Server token budget(s) for the FIFO baseline (default: each --max-step).
    #[arg(long)]
    pub budget: Option<String>,
    /// Bulk window(s).
    #[arg(long, default_value = "64,256,1024")]
    pub window: String,
    #[arg(long, default_value_t = 0.98)]
    pub min_fill: f64,
    /// A platform profile for the throughput curve (default: GB10's bf16 forward).
    #[arg(long)]
    pub profile: Option<PathBuf>,
    #[arg(long)]
    pub query: Option<String>,
}

pub fn simulate_main(argv: &[String]) -> Result<i32> {
    use rayon::prelude::*;
    let a = SimArgs::try_parse_from(std::iter::once("sysone bulk simulate".to_string()).chain(argv.iter().cloned()))
        .unwrap_or_else(|e| e.exit());
    let cfg = ModelConfig::load(&a.model)?;
    let frontend = Frontend::load(&a.model, Limits { max_branch_tokens: 32768, max_request_tokens: 65536 })?;
    let info = ServerInfo::for_cli(&a.model, &cfg.model_type, None, "batched", None)?;
    let text = std::fs::read(&a.input).with_context(|| format!("reading {}", a.input.display()))?;
    let lines: Vec<&[u8]> = text.split(|&b| b == b'\n').collect();
    let sizes: Vec<Option<Size>> = lines
        .par_iter()
        .enumerate()
        .map(|(i, l)| match super::job::parse_line(i as u64 + 1, l) {
            super::job::Line::Job(j) => {
                let p = crate::api::server::params(j.query.as_deref().or(a.query.as_deref()), info.default_debias);
                crate::api::server::prepare(&frontend, &info, j.body, &p)
                    .ok()
                    .map(|r| Size { tokens: r.plan.total, branches: r.plan.branches.len() })
            }
            _ => None,
        })
        .collect();
    let sizes: Vec<Size> = sizes.into_iter().flatten().collect();
    ensure!(!sizes.is_empty(), "no plannable job in {}", a.input.display());
    let tput = match &a.profile {
        Some(p) => {
            let p = crate::engine::platform::PlatformProfile::load(p)?;
            Throughput(p.forward.iter().map(|f| (f.tokens, f.tokens_per_s)).collect())
        }
        None => Throughput::gb10_default(),
    };
    let tokens: usize = sizes.iter().map(|s| s.tokens).sum();
    eprintln!(
        "{} requests, {} packed tokens (mean {:.0}, max {})",
        sizes.len(),
        tokens,
        tokens as f64 / sizes.len() as f64,
        sizes.iter().map(|s| s.tokens).max().unwrap()
    );
    let row = |policy: String, st: &PackStats| {
        serde_json::json!({
            "policy": policy, "steps": st.steps, "tokens": st.tokens, "bucket_tokens": st.bucket_tokens,
            "padding_frac": st.padding(), "est_gpu_s": st.est_seconds,
        })
    };
    let lim = |max: usize, fine: bool| StepLimits { buckets: ladder(max, fine), max_answers: 2048, max_segments: 2048, max_prefixes: 256 };
    let maxes: Vec<usize> = list(&a.max_step)?;
    let windows: Vec<usize> = list(&a.window)?;
    let budgets: Option<Vec<usize>> = a.budget.as_deref().map(list).transpose()?;
    for (k, &max) in maxes.iter().enumerate() {
        let budget = budgets.as_ref().map_or(max, |b| b[k.min(b.len() - 1)]);
        let st = simulate_fifo(&sizes, budget, &lim(max, false), &tput);
        println!("{}", row(format!("server-fifo budget={budget} buckets=pow2..{max}"), &st));
        for &w in &windows {
            for fine in [false, true] {
                let st = simulate_bulk(&sizes, w, &lim(max, fine), &tput, &PackPolicy { min_fill: a.min_fill });
                println!("{}", row(format!("bulk window={w} buckets={}..{max}", if fine { "fine" } else { "pow2" }), &st));
            }
        }
    }
    Ok(0)
}

/// `sysone bulk ...` (argv after `bulk`).
pub fn main(argv: &[String]) -> Result<i32> {
    match argv.first().map(String::as_str) {
        Some("worker") => {
            let spec = match argv.get(1..) {
                Some([flag, json]) if flag == "--spec" => json,
                _ => bail!("usage: sysone bulk worker --spec JSON (started by sysone bulk)"),
            };
            super::worker::process_main(spec)?;
            Ok(0)
        }
        Some("gen") => gen_main(&argv[1..]),
        Some("simulate") => simulate_main(&argv[1..]),
        Some("run") => run_main(&argv[1..]),
        _ => run_main(argv),
    }
}
