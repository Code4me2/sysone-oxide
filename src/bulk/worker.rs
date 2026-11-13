//! Bulk workers: one engine each, driven by the coordinator ([`super::run`]).
//!
//! A worker is either a **process** (`sysone bulk worker --spec JSON`, the default for GPU engines: one process
//! per GPU, started through `scripts/gpu.sh --gpu N`, which takes that GPU's lock and makes it the process's CUDA
//! device 0) or a **thread** of the coordinator (the mock engine, or one GPU when the coordinator itself runs
//! under `scripts/gpu.sh`). Both run the same loop ([`serve`]): build the engine, report a [`Hello`] (device,
//! buckets, limits, throughput curve), then run each step it is sent and report its logits.
//!
//! Why processes per GPU and not one process with an engine per device: the engine and the libraries under it
//! bind to the *default* device in several places (`GpuEngine::load` creates its context on device 0;
//! `TileKernels::new` and every SIMT launch load the module on cuda-async's thread-local default device; cuTile's
//! JIT and launch path and the cuBLAS handles follow the current context). With `CUDA_VISIBLE_DEVICES=N` each of
//! those *is* GPU N, with no code path to audit, and a crash or OOM on one card cannot take the others down.
//! The cost is the pipe: a step's ids and logits are well under a megabyte per several hundred milliseconds of
//! GPU time. docs/BULK.md has the full argument.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::pack::{Throughput, ladder};
use super::wire::{self, Done, Frame, Hello};
use crate::contract::{DecisionBatch, DecisionEngine, MassReadout, MockEngine, SlotLogits, StepLimits};

/// Which engine a worker runs and how it is sized. Passed to a worker process as JSON (`--spec`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EngineSpec {
    /// `bf16` (the GPU engine) or `mock`.
    pub engine: String,
    pub model: PathBuf,
    /// The frontend's slot token ids (the readout rows).
    pub slot_ids: Vec<u32>,
    /// Largest bucket; None: [`DEFAULT_MAX_STEP`] (bf16, then cut to the device's memory) or the mock's.
    pub max_step_tokens: Option<usize>,
    /// Buckets at powers of two and their midpoints ([`ladder`]) instead of powers of two only.
    pub fine_ladder: bool,
    /// bf16: keep the buckets whose arena, scratch and readback fit this fraction of the memory free after the
    /// weights (`EngineOptions::memory_fraction`).
    pub memory_fraction: f64,
    pub max_prefixes: usize,
    pub gdn_backend: String,
    pub cuda_graphs: bool,
    /// bf16: platform profiles (`hwprobe --profile-out`); the one whose device name matches this GPU gives the
    /// throughput curve the packer uses for the last steps of a run (else GB10's measured curve).
    pub profiles: Vec<PathBuf>,
    /// bf16: refuse to start while another process uses the GPU (benchmarks).
    pub require_idle: bool,
    /// mock: simulated step time, fixed part and per bucket token.
    pub mock_step_us: u64,
    pub mock_ns_per_token: u64,
    /// mock: stop after this many steps without answering the next one, as a crashed worker would (tests).
    #[serde(default)]
    pub mock_exit_after: Option<u64>,
}

/// Default largest bucket for bulk on a device with room for it: 32,768 tokens. Larger steps amortise more
/// per-step overhead, pack better, and on GB10 32k is faster per token than 16k (docs/BULK.md); the memory check
/// cuts it on smaller cards.
pub const DEFAULT_MAX_STEP: usize = 32768;

impl EngineSpec {
    pub fn mock(slot_ids: Vec<u32>, max_step_tokens: usize) -> Self {
        EngineSpec {
            engine: "mock".into(),
            model: PathBuf::new(),
            slot_ids,
            max_step_tokens: Some(max_step_tokens),
            fine_ladder: false,
            memory_fraction: 0.9,
            max_prefixes: 256,
            gdn_backend: "auto".into(),
            cuda_graphs: true,
            profiles: Vec::new(),
            require_idle: false,
            mock_step_us: 0,
            mock_ns_per_token: 0,
            mock_exit_after: None,
        }
    }
}

/// `MockEngine` with buckets and a simulated step time (`step_us + ns_per_token x bucket tokens`), so the
/// coordinator's packing, padding accounting and load balancing can be exercised without a GPU.
pub struct SimEngine {
    pub inner: MockEngine,
    pub buckets: Vec<usize>,
    pub step: Duration,
    pub per_token: Duration,
    pub last_bucket: Option<usize>,
}

impl SimEngine {
    fn bucket(&self, tokens: usize) -> Result<usize> {
        self.buckets.iter().copied().find(|&b| b >= tokens).with_context(|| format!("{tokens} tokens over the largest bucket"))
    }

    fn sleep(&self, bucket: usize) {
        let t = self.step + self.per_token * bucket as u32;
        if !t.is_zero() {
            std::thread::sleep(t);
        }
    }
}

impl DecisionEngine for SimEngine {
    fn max_tokens_per_step(&self) -> usize {
        *self.buckets.last().unwrap()
    }

    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
        let b = self.bucket(batch.num_tokens())?;
        self.sleep(b);
        self.last_bucket = Some(b);
        self.inner.run(batch)
    }

    fn run_with_mass(&mut self, batch: &DecisionBatch) -> Result<MassReadout> {
        let b = self.bucket(batch.num_tokens())?;
        self.sleep(b);
        self.last_bucket = Some(b);
        self.inner.run_with_mass(batch)
    }

    fn info(&self) -> Vec<(&'static str, String)> {
        vec![("engine", "mock".into())]
    }

    fn step_limits(&self) -> StepLimits {
        StepLimits { buckets: self.buckets.clone(), max_answers: 2048, max_segments: 2048, max_prefixes: 256 }
    }

    fn last_bucket_tokens(&self) -> Option<usize> {
        self.last_bucket
    }
}

/// Build the engine of `spec` on the calling thread (which then owns it) and describe it.
pub fn build(spec: &EngineSpec) -> Result<(Box<dyn DecisionEngine>, Hello)> {
    let t0 = Instant::now();
    match spec.engine.as_str() {
        "mock" => {
            let max = spec.max_step_tokens.unwrap_or(DEFAULT_MAX_STEP);
            let e = SimEngine {
                inner: MockEngine { max_tokens: max },
                buckets: ladder(max, spec.fine_ladder),
                step: Duration::from_micros(spec.mock_step_us),
                per_token: Duration::from_nanos(spec.mock_ns_per_token),
                last_bucket: None,
            };
            let mut hello = hello(&e, "mock".into(), "none".into(), 0, Throughput::gb10_default(), t0);
            // Where gpu.sh put this worker (the lock tests check it).
            for k in ["CUDA_VISIBLE_DEVICES", "SYSONE_GPU_HELD"] {
                hello.info.push((k.to_lowercase(), std::env::var(k).unwrap_or_default()));
            }
            Ok((Box::new(e), hello))
        }
        "bf16" => build_gpu(spec, t0),
        other => bail!("unknown engine {other:?} (bf16 or mock)"),
    }
}

fn hello(e: &dyn DecisionEngine, device: String, arch: String, mem: u64, tput: Throughput, t0: Instant) -> Hello {
    let l = e.step_limits();
    Hello {
        device,
        arch,
        total_mem_bytes: mem,
        buckets: l.buckets,
        max_answers: l.max_answers,
        max_segments: l.max_segments,
        max_prefixes: l.max_prefixes,
        throughput: tput.0,
        info: e.info().into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        load_seconds: t0.elapsed().as_secs_f64(),
    }
}

fn build_gpu(spec: &EngineSpec, t0: Instant) -> Result<(Box<dyn DecisionEngine>, Hello)> {
    use crate::engine::gpu::{EngineOptions, GpuEngine, parse_gdn_backend};
    use crate::engine::platform::{PlatformProfile, require_idle_gpu};
    if spec.require_idle {
        eprintln!("sysone bulk worker: {}", require_idle_gpu(false)?);
    }
    let profiles = spec.profiles.iter().map(|p| PlatformProfile::load(p)).collect::<Result<Vec<_>>>()?;
    let max = spec.max_step_tokens.unwrap_or(DEFAULT_MAX_STEP);
    let opts = EngineOptions {
        buckets: ladder(max, spec.fine_ladder),
        cuda_graphs: spec.cuda_graphs,
        max_prefixes: spec.max_prefixes,
        gdn_backend: parse_gdn_backend(&spec.gdn_backend)?,
        memory_fraction: Some(spec.memory_fraction),
        ..Default::default()
    };
    let e = GpuEngine::load(&spec.model, &spec.slot_ids, opts)?;
    let (_, total) = e.memory()?;
    let device = e.device_name()?;
    let tput = match profiles.iter().find(|p| p.device_name == device && !p.forward.is_empty()) {
        Some(p) => Throughput(p.forward.iter().map(|f| (f.tokens, f.tokens_per_s)).collect()),
        None => {
            if !profiles.is_empty() {
                eprintln!("sysone bulk worker: no profile for {device:?}; using the default throughput curve");
            }
            Throughput::gb10_default()
        }
    };
    let arch = e.arch.clone();
    let h = hello(&e, device, arch, total as u64, tput, t0);
    Ok((Box::new(e), h))
}

/// The result of one step on a worker.
fn run_step(engine: &mut dyn DecisionEngine, id: u64, wants_mass: bool, batch: &DecisionBatch) -> Result<Done> {
    let t = Instant::now();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if wants_mass {
            engine.run_with_mass(batch)
        } else {
            engine.run(batch).map(|logits| MassReadout { logits, allowed_mass: None })
        }
    }))
    .unwrap_or_else(|_| Err(anyhow::anyhow!("engine panicked")))?;
    let step_seconds = t.elapsed().as_secs_f64();
    anyhow::ensure!(r.logits.rows == batch.num_branches(), "engine returned {} rows for {}", r.logits.rows, batch.num_branches());
    Ok(Done { id, step_seconds, bucket_tokens: engine.last_bucket_tokens(), logits: r.logits, masses: r.allowed_mass })
}

/// A step sent to a worker.
pub struct StepMsg {
    pub id: u64,
    pub wants_mass: bool,
    pub batch: DecisionBatch,
}

/// What workers tell the coordinator; the `usize` is the worker's index.
#[derive(Debug)]
pub enum Event {
    Ready(usize, Hello),
    Done(usize, Done),
    Failed(usize, u64, String),
    /// The worker is gone (load failure, crash, or the end of the run).
    Exited(usize, String),
}

/// The `sysone bulk worker` process: build the engine, send HELLO on stdout, then answer STEP frames from stdin
/// until it closes.
pub fn process_main(spec_json: &str) -> Result<()> {
    let spec: EngineSpec = serde_json::from_str(spec_json).context("--spec")?;
    let (mut engine, hello) = build(&spec)?;
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    wire::write_frame(&mut out, &wire::encode(&Frame::Hello(hello), &[])?)?;
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut steps = 0u64;
    while let Some(body) = wire::read_body(&mut input)? {
        if spec.mock_exit_after.is_some_and(|n| steps >= n) {
            eprintln!("sysone bulk worker: simulated crash after {steps} steps");
            std::process::exit(3);
        }
        steps += 1;
        let (id, wants_mass, batch) = match wire::decode(&body, &|_| None)? {
            Frame::Step { id, wants_mass, batch } => (id, wants_mass, batch),
            other => bail!("worker expected STEP, got {other:?}"),
        };
        let frame = match run_step(engine.as_mut(), id, wants_mass, &batch) {
            Ok(done) => wire::encode(&Frame::Done(done), &batch.answer_slots)?,
            Err(e) => wire::encode(&Frame::Failed { id, msg: format!("{e:#}") }, &[])?,
        };
        wire::write_frame(&mut out, &frame)?;
    }
    Ok(())
}

/// The coordinator's side of one worker.
pub struct Handle {
    pub name: String,
    tx: Option<Sender<StepMsg>>,
    child: Option<Child>,
    threads: Vec<JoinHandle<()>>,
}

impl Handle {
    /// Queue a step (the worker runs its steps in order).
    pub fn send(&self, msg: StepMsg) -> bool {
        self.tx.as_ref().is_some_and(|tx| tx.send(msg).is_ok())
    }

    /// Close the worker's input and wait for it to finish.
    pub fn close(&mut self) {
        self.tx = None;
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        if let Some(mut c) = self.child.take() {
            let _ = c.wait();
        }
    }

    /// Kill a worker process (used when the coordinator aborts).
    pub fn kill(&mut self) {
        if let Some(c) = &mut self.child {
            let _ = c.kill();
        }
    }

    /// A worker thread in this process.
    pub fn thread(index: usize, name: String, spec: EngineSpec, events: Sender<Event>) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel::<StepMsg>();
        let t = std::thread::Builder::new().name(format!("bulk-{name}")).spawn(move || thread_main(index, spec, rx, events))?;
        Ok(Handle { name, tx: Some(tx), child: None, threads: vec![t] })
    }

    /// A worker process running `cmd` (e.g. `scripts/gpu.sh --gpu 1 sysone bulk worker --spec ...`).
    pub fn process(index: usize, name: String, mut cmd: Command, events: Sender<Event>) -> Result<Self> {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting worker {name}: {cmd:?}"))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        // Answer slots of the steps in flight, to widen their compact logits.
        let slots: Arc<Mutex<std::collections::HashMap<u64, Vec<u16>>>> = Arc::default();
        let (tx, rx) = std::sync::mpsc::channel::<StepMsg>();
        let s2 = slots.clone();
        let ev = events.clone();
        let n = name.clone();
        let writer = std::thread::Builder::new().name(format!("bulk-{name}-in")).spawn(move || {
            let mut w = BufWriter::new(stdin);
            for m in rx {
                s2.lock().unwrap().insert(m.id, m.batch.answer_slots.clone());
                let frame = match wire::encode(&Frame::Step { id: m.id, wants_mass: m.wants_mass, batch: m.batch }, &[]) {
                    Ok(f) => f,
                    Err(e) => {
                        let _ = ev.send(Event::Failed(index, m.id, format!("encoding step: {e:#}")));
                        continue;
                    }
                };
                if let Err(e) = wire::write_frame(&mut w, &frame) {
                    let _ = ev.send(Event::Exited(index, format!("worker {n}: writing to it: {e}")));
                    return;
                }
            }
        })?;
        let n = name.clone();
        let reader = std::thread::Builder::new().name(format!("bulk-{name}-out")).spawn(move || {
            let mut r = std::io::BufReader::new(stdout);
            let why = loop {
                let body = match wire::read_body(&mut r) {
                    Ok(Some(b)) => b,
                    Ok(None) => break "exited".to_string(),
                    Err(e) => break format!("reading from it: {e:#}"),
                };
                let slots_of = |id: u64| slots.lock().unwrap().remove(&id);
                let ev = match wire::decode(&body, &slots_of) {
                    Ok(Frame::Hello(h)) => Event::Ready(index, h),
                    Ok(Frame::Done(d)) => Event::Done(index, d),
                    Ok(Frame::Failed { id, msg }) => {
                        slots.lock().unwrap().remove(&id);
                        Event::Failed(index, id, msg)
                    }
                    Ok(other) => break format!("unexpected frame {other:?}"),
                    Err(e) => break format!("bad frame: {e:#}"),
                };
                if events.send(ev).is_err() {
                    return;
                }
            };
            let _ = events.send(Event::Exited(index, format!("worker {n}: {why}")));
        })?;
        let n = name.clone();
        let err = std::thread::Builder::new().name(format!("bulk-{name}-err")).spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                eprintln!("[{n}] {line}");
            }
        })?;
        Ok(Handle { name, tx: Some(tx), child: Some(child), threads: vec![writer, reader, err] })
    }
}

fn thread_main(index: usize, spec: EngineSpec, rx: Receiver<StepMsg>, events: Sender<Event>) {
    let (mut engine, hello) = match build(&spec) {
        Ok(x) => x,
        Err(e) => {
            let _ = events.send(Event::Exited(index, format!("building the engine: {e:#}")));
            return;
        }
    };
    if events.send(Event::Ready(index, hello)).is_err() {
        return;
    }
    for (steps, m) in rx.into_iter().enumerate() {
        if spec.mock_exit_after.is_some_and(|n| steps as u64 >= n) {
            let _ = events.send(Event::Exited(index, format!("simulated crash after {steps} steps")));
            return;
        }
        let ev = match run_step(engine.as_mut(), m.id, m.wants_mass, &m.batch) {
            Ok(d) => Event::Done(index, d),
            Err(e) => Event::Failed(index, m.id, format!("{e:#}")),
        };
        if events.send(ev).is_err() {
            return;
        }
    }
    let _ = events.send(Event::Exited(index, "exited".into()));
}

/// The command that starts a worker process for GPU `gpu` (nvidia-smi index) through `gpu_sh`, or directly when
/// `gpu` is None (the mock engine needs no GPU and no lock).
pub fn command(exe: &std::path::Path, gpu_sh: Option<&std::path::Path>, gpu: Option<&str>, spec: &EngineSpec) -> Result<Command> {
    let json = serde_json::to_string(spec)?;
    let mut c = match (gpu, gpu_sh) {
        (Some(g), Some(sh)) => {
            let mut c = Command::new(sh);
            c.arg("--gpu").arg(g).arg(exe);
            c
        }
        (Some(_), None) => bail!("a GPU worker needs scripts/gpu.sh (--gpu-sh) to take its GPU's lock"),
        (None, _) => Command::new(exe),
    };
    c.arg("bulk").arg("worker").arg("--spec").arg(json);
    Ok(c)
}
