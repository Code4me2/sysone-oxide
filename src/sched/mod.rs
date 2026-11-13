//! Admission queue, token-budget batcher, and the GPU worker that drives a `DecisionEngine` (DESIGN.md §7).
//! Owner: W5.
//!
//! ```text
//! submit ─► AdmissionQueue (bounded; full → refused) ─► batcher thread ─► one-slot channel ─► GPU worker thread
//!                                                       packs batch n+1      (double buffer)    runs batch n
//! ```
//! Each request gets its slot-logit rows back through a oneshot channel.
//!
//! - [`Scheduler::submit`] admits requests ([`Packable`], e.g. a [`Plan`]) all or none; a full queue returns
//!   [`QueueFull`] and the server answers 503.
//! - [`batcher::run`] packs whole requests into one [`DecisionBatch`] until [`BatchPolicy::token_budget`], or until
//!   the oldest has waited [`BatchPolicy::max_wait`] *and* the engine is idle. A request is never split; one over
//!   the budget runs alone; one over the engine's `max_tokens_per_step` is refused.
//! - [`worker::run`] owns the engine (built on that thread by the [`EngineFactory`]), runs each batch, and sends
//!   every job its rows as a [`Scored`].
//! - Around it: [`pool::CpuPool`] (CPU threads for planning and answer assembly), [`metrics::Metrics`]
//!   (`/metrics`), [`affinity`] (`--pin-cores`).
//!
//! See docs/ARCHITECTURE.md §2 and §4.4.

pub mod affinity;
pub mod batcher;
pub mod metrics;
pub mod pool;
pub mod queue;
pub mod worker;

use anyhow::{Context, Result, ensure};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

use crate::contract::{DecisionBatch, DecisionEngine, MAX_SLOTS};
use crate::frontend::Plan;
pub use batcher::BatchPolicy;
pub use metrics::Metrics;
pub use pool::{CpuPool, PoolError};
pub use queue::AdmissionQueue;

/// A request the batcher can pack: one prefix and its suffixes.
pub trait Packable: Send + Sync {
    /// Tokens it adds to a batch (prefix + every suffix).
    fn num_tokens(&self) -> usize;
    fn num_branches(&self) -> usize;
    /// Append to `batch`; returns the batch row of branch 0.
    fn push_into(&self, batch: &mut DecisionBatch) -> Result<usize>;
    /// Whether this request wants `allowed_token_mass` (a `debug` request). A batch holding one runs through
    /// [`DecisionEngine::run_with_mass`].
    fn wants_mass(&self) -> bool {
        false
    }
}

impl Packable for Plan {
    fn num_tokens(&self) -> usize {
        self.total
    }

    fn num_branches(&self) -> usize {
        self.branches.len()
    }

    fn push_into(&self, batch: &mut DecisionBatch) -> Result<usize> {
        Plan::push_into(self, batch)
    }
}

/// A request's result from the engine.
#[derive(Clone, Debug)]
pub struct Scored {
    /// `num_branches × MAX_SLOTS` slot logits, branch-major.
    pub logits: Vec<f32>,
    /// Per-branch `allowed_token_mass`, when the batch ran with masses and the engine reported them.
    pub allowed_mass: Option<Vec<f64>>,
    /// Admission to the start of its engine step.
    pub queue_wait: Duration,
    /// When its engine step started.
    pub step_started: Instant,
    /// Its engine step's wall time.
    pub step: Duration,
    pub batch_tokens: usize,
    pub batch_requests: usize,
}

impl Scored {
    /// Branch `i`'s logits for its first `slots` slots.
    pub fn row(&self, i: usize, slots: usize) -> &[f32] {
        &self.logits[i * MAX_SLOTS..i * MAX_SLOTS + slots]
    }
}

pub type Reply = std::result::Result<Scored, String>;

pub struct Job {
    pub work: Arc<dyn Packable>,
    pub admitted: Instant,
    pub reply: oneshot::Sender<Reply>,
}

/// A validated batch and the jobs in it, each with the batch row of its branch 0.
pub struct Packed {
    pub batch: DecisionBatch,
    pub jobs: Vec<(Job, usize)>,
    /// Some job wants `allowed_token_mass`: run with [`DecisionEngine::run_with_mass`].
    pub wants_mass: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedConfig {
    pub token_budget: usize,
    pub max_wait: Duration,
    /// Requests the admission queue holds before refusing more.
    pub queue_capacity: usize,
}

impl Default for SchedConfig {
    /// DESIGN.md §7 defaults: 16k tokens, 2 ms.
    fn default() -> Self {
        SchedConfig { token_budget: 16384, max_wait: Duration::from_millis(2), queue_capacity: 1024 }
    }
}

/// Builds the engine on the GPU worker thread, which then owns it (a CUDA context is bound to the thread that
/// creates it).
pub type EngineFactory = Box<dyn FnOnce() -> Result<Box<dyn DecisionEngine>> + Send>;

/// The admission queue was full (or the scheduler is shutting down): the server answers 503.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFull;

pub struct Scheduler {
    queue: Arc<AdmissionQueue<Job>>,
    policy: BatchPolicy,
    metrics: Arc<Metrics>,
    threads: Vec<JoinHandle<()>>,
}

impl Scheduler {
    /// Start the GPU worker (building the engine on it) and the batcher.
    pub fn start(cfg: SchedConfig, factory: EngineFactory, metrics: Arc<Metrics>) -> Result<Self> {
        Self::start_with(cfg, factory, metrics, None)
    }

    /// [`start`](Self::start), with the GPU worker and batcher threads pinned to `pin` (docs/PLATFORM.md P7).
    pub fn start_with(cfg: SchedConfig, factory: EngineFactory, metrics: Arc<Metrics>, pin: Option<Vec<usize>>) -> Result<Self> {
        ensure!(cfg.token_budget > 0, "token budget must be positive");
        let (ready_tx, ready_rx) = mpsc::channel::<Result<usize>>();
        // The one-slot channel is the double buffer: one batch running, one packed and waiting.
        let (tx, rx) = mpsc::sync_channel::<Packed>(1);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let busy = in_flight.clone();
        let m = metrics.clone();
        let gpu_pin = pin.clone();
        let gpu = std::thread::Builder::new().name("sysone-gpu".into()).spawn(move || {
            affinity::maybe_pin(gpu_pin.as_deref(), "the GPU worker");
            let engine = match factory() {
                Ok(e) => e,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            m.set_engine_info(engine.info());
            let _ = ready_tx.send(Ok(engine.max_tokens_per_step()));
            worker::run(engine, rx, m, busy);
        })?;
        let max_tokens = match ready_rx.recv() {
            Ok(r) => r.context("building the engine")?,
            Err(_) => {
                let _ = gpu.join();
                anyhow::bail!("the GPU worker exited before building the engine");
            }
        };
        ensure!(max_tokens > 0, "engine accepts no tokens per step");
        let policy =
            BatchPolicy { token_budget: cfg.token_budget.min(max_tokens), max_wait: cfg.max_wait, max_tokens };
        let queue = Arc::new(AdmissionQueue::new(cfg.queue_capacity));
        let q = queue.clone();
        let batcher = std::thread::Builder::new().name("sysone-batcher".into()).spawn(move || {
            affinity::maybe_pin(pin.as_deref(), "the batcher");
            batcher::run(q, policy, tx, in_flight)
        })?;
        Ok(Scheduler { queue, policy, metrics, threads: vec![batcher, gpu] })
    }

    /// The effective policy (the budget is clamped to the engine's step size).
    pub fn policy(&self) -> BatchPolicy {
        self.policy
    }

    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }

    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    pub fn queue_capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Admit requests, all or none. Each receiver yields that request's logits (or an error message).
    pub fn submit(&self, works: Vec<Arc<dyn Packable>>) -> std::result::Result<Vec<oneshot::Receiver<Reply>>, QueueFull> {
        let admitted = Instant::now();
        let mut jobs = Vec::with_capacity(works.len());
        let mut rxs = Vec::with_capacity(works.len());
        for work in works {
            let (reply, rx) = oneshot::channel();
            jobs.push(Job { work, admitted, reply });
            rxs.push(rx);
        }
        self.queue.try_push_all(jobs).map_err(|_| QueueFull)?;
        Ok(rxs)
    }

    /// Stop admitting, finish what is queued, and join the threads.
    pub fn shutdown(&mut self) {
        self.queue.close();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{MockEngine, SlotLogits};
    use std::sync::Mutex;

    /// A synthetic request: `prefix` tokens, then one suffix per entry of `suffixes` (length, slots). Token
    /// values derive from `seed`, so different requests score differently under the mock engine.
    struct Fake {
        seed: u32,
        prefix: usize,
        suffixes: Vec<(usize, u16)>,
    }

    impl Fake {
        fn ids(&self) -> (Vec<u32>, Vec<(Vec<u32>, u16)>) {
            let p = (0..self.prefix as u32).map(|i| self.seed * 1000 + i).collect();
            let s = self
                .suffixes
                .iter()
                .enumerate()
                .map(|(j, &(n, k))| ((0..n as u32).map(|i| self.seed * 7 + j as u32 * 31 + i).collect(), k))
                .collect();
            (p, s)
        }
    }

    impl Packable for Fake {
        fn num_tokens(&self) -> usize {
            self.prefix + self.suffixes.iter().map(|s| s.0).sum::<usize>()
        }

        fn num_branches(&self) -> usize {
            self.suffixes.len()
        }

        fn push_into(&self, batch: &mut DecisionBatch) -> Result<usize> {
            let (p, s) = self.ids();
            let s: Vec<(&[u32], u16)> = s.iter().map(|(v, k)| (v.as_slice(), *k)).collect();
            batch.push_request(&p, &s)
        }
    }

    /// Records every batch it runs: (tokens, requests).
    struct Recording {
        inner: MockEngine,
        log: Arc<Mutex<Vec<(usize, usize)>>>,
        delay: Duration,
    }

    impl DecisionEngine for Recording {
        fn max_tokens_per_step(&self) -> usize {
            self.inner.max_tokens_per_step()
        }

        fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
            let requests = batch.segments.iter().filter(|s| s.is_prefix()).count();
            self.log.lock().unwrap().push((batch.num_tokens(), requests));
            std::thread::sleep(self.delay);
            self.inner.run(batch)
        }
    }

    fn alone(f: &Fake) -> Vec<f32> {
        let mut b = DecisionBatch::default();
        f.push_into(&mut b).unwrap();
        let l = MockEngine { max_tokens: 1 << 20 }.run(&b).unwrap();
        l.data
    }

    type Log = Arc<Mutex<Vec<(usize, usize)>>>;

    fn start(cfg: SchedConfig, max_tokens: usize, delay: Duration) -> (Scheduler, Log) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let l = log.clone();
        let factory: EngineFactory =
            Box::new(move || Ok(Box::new(Recording { inner: MockEngine { max_tokens }, log: l, delay }) as Box<dyn DecisionEngine>));
        (Scheduler::start(cfg, factory, Arc::new(Metrics::new())).unwrap(), log)
    }

    #[test]
    fn packs_within_budget_and_answers_every_request_exactly() {
        let cfg = SchedConfig { token_budget: 1000, max_wait: Duration::from_millis(20), queue_capacity: 4096 };
        let (sched, log) = start(cfg, 1 << 16, Duration::from_millis(1));
        // Sizes from 3 to ~2100 tokens; some exceed the budget and must run alone.
        let mut rng = 0x2545f4914f6cdd1du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let fakes: Vec<Arc<Fake>> = (0..300)
            .map(|i| {
                let big = next() % 23 == 0;
                let prefix = if big { 1100 + (next() % 1000) as usize } else { 1 + (next() % 200) as usize };
                let suffixes = (0..1 + next() % 4).map(|_| (1 + (next() % 30) as usize, 1 + (next() % 256) as u16)).collect();
                Arc::new(Fake { seed: i, prefix, suffixes })
            })
            .collect();
        // Submit in waves so batches actually fill.
        let mut rxs = Vec::new();
        for chunk in fakes.chunks(37) {
            let works: Vec<Arc<dyn Packable>> = chunk.iter().map(|f| f.clone() as Arc<dyn Packable>).collect();
            rxs.extend(sched.submit(works).unwrap());
        }
        for (f, rx) in fakes.iter().zip(rxs) {
            let got = rx.blocking_recv().unwrap().unwrap();
            let want = alone(f);
            for (i, &(_, k)) in f.suffixes.iter().enumerate() {
                assert_eq!(got.row(i, k as usize), &want[i * MAX_SLOTS..i * MAX_SLOTS + k as usize], "request {}", f.seed);
            }
            assert!(got.batch_tokens <= 1000 || got.batch_requests == 1, "{} tokens in a {}-request batch", got.batch_tokens, got.batch_requests);
        }
        let log = log.lock().unwrap();
        let total: usize = log.iter().map(|b| b.1).sum();
        assert_eq!(total, fakes.len());
        for &(tokens, requests) in log.iter() {
            assert!(tokens <= 1000 || requests == 1, "batch of {tokens} tokens holds {requests} requests");
        }
        let packed = log.iter().filter(|b| b.1 > 1).count();
        assert!(packed > 0, "no batch held more than one request: {log:?}");
        let oversize = fakes.iter().filter(|f| f.num_tokens() > 1000).count();
        assert_eq!(log.iter().filter(|b| b.0 > 1000).count(), oversize);
    }

    /// With a 60 ms engine step and a request every 5 ms, a 1 ms max_wait must not cut batches to one request
    /// while the engine is busy: the next batch keeps filling until the engine is free.
    #[test]
    fn keeps_packing_while_the_engine_is_busy() {
        let cfg = SchedConfig { token_budget: 1 << 15, max_wait: Duration::from_millis(1), queue_capacity: 1024 };
        let (sched, log) = start(cfg, 1 << 16, Duration::from_millis(60));
        let mut rxs = Vec::new();
        for i in 0..40 {
            let w = Arc::new(Fake { seed: i, prefix: 50, suffixes: vec![(5, 4)] }) as Arc<dyn Packable>;
            rxs.extend(sched.submit(vec![w]).unwrap());
            std::thread::sleep(Duration::from_millis(5));
        }
        for rx in rxs {
            rx.blocking_recv().unwrap().unwrap();
        }
        let log = log.lock().unwrap();
        let singles = log.iter().filter(|b| b.1 == 1).count();
        assert!(log.len() <= 8 && singles <= 2, "batches (tokens, requests): {log:?}");
    }

    #[test]
    fn full_queue_refuses_and_recovers() {
        let cfg = SchedConfig { token_budget: 100, max_wait: Duration::ZERO, queue_capacity: 2 };
        let (sched, _log) = start(cfg, 1 << 16, Duration::from_millis(30));
        let one = || vec![Arc::new(Fake { seed: 1, prefix: 90, suffixes: vec![(5, 2)] }) as Arc<dyn Packable>];
        let mut rxs = Vec::new();
        let mut refused = 0;
        for _ in 0..50 {
            match sched.submit(one()) {
                Ok(r) => rxs.extend(r),
                Err(QueueFull) => refused += 1,
            }
        }
        assert!(refused > 0);
        // At most: one running, one in the slot, two held by the batcher (a built batch and the carry), and the
        // queue capacity.
        assert!(rxs.len() <= 4 + 2, "{} admitted", rxs.len());
        for rx in rxs {
            rx.blocking_recv().unwrap().unwrap();
        }
        // Drained: admission works again.
        let rx = sched.submit(one()).unwrap().pop().unwrap();
        rx.blocking_recv().unwrap().unwrap();
    }

    #[test]
    fn request_over_engine_step_is_refused() {
        let cfg = SchedConfig { token_budget: 64, max_wait: Duration::ZERO, queue_capacity: 8 };
        let (sched, _log) = start(cfg, 100, Duration::ZERO);
        assert_eq!(sched.policy().token_budget, 64);
        let big = Arc::new(Fake { seed: 3, prefix: 101, suffixes: vec![(1, 2)] }) as Arc<dyn Packable>;
        let ok = Arc::new(Fake { seed: 4, prefix: 80, suffixes: vec![(1, 2)] }) as Arc<dyn Packable>;
        let rxs = sched.submit(vec![big, ok]).unwrap();
        let mut it = rxs.into_iter();
        assert!(it.next().unwrap().blocking_recv().unwrap().unwrap_err().contains("exceeds"));
        // 81 tokens > the 64 budget: runs alone and succeeds.
        let s = it.next().unwrap().blocking_recv().unwrap().unwrap();
        assert_eq!((s.batch_tokens, s.batch_requests), (81, 1));
    }

    #[test]
    fn engine_failure_reaches_every_request() {
        struct Failing;
        impl DecisionEngine for Failing {
            fn max_tokens_per_step(&self) -> usize {
                1000
            }
            fn run(&mut self, _: &DecisionBatch) -> Result<SlotLogits> {
                anyhow::bail!("device lost")
            }
        }
        let factory: EngineFactory = Box::new(|| Ok(Box::new(Failing) as Box<dyn DecisionEngine>));
        let sched = Scheduler::start(SchedConfig::default(), factory, Arc::new(Metrics::new())).unwrap();
        let w = Arc::new(Fake { seed: 5, prefix: 3, suffixes: vec![(1, 2)] }) as Arc<dyn Packable>;
        let rx = sched.submit(vec![w]).unwrap().pop().unwrap();
        assert!(rx.blocking_recv().unwrap().unwrap_err().contains("device lost"));
    }

    /// A `Fake` that asks for `allowed_token_mass`.
    struct Debug(Fake);

    impl Packable for Debug {
        fn num_tokens(&self) -> usize {
            self.0.num_tokens()
        }
        fn num_branches(&self) -> usize {
            self.0.num_branches()
        }
        fn push_into(&self, batch: &mut DecisionBatch) -> Result<usize> {
            self.0.push_into(batch)
        }
        fn wants_mass(&self) -> bool {
            true
        }
    }

    /// Masses reach exactly the batches that hold a debug request, mapped to each request's own rows, and only
    /// those batches pay for `run_with_mass`.
    #[test]
    fn masses_only_for_batches_with_a_debug_request() {
        struct Counting {
            inner: MockEngine,
            with_mass: Arc<Mutex<Vec<bool>>>,
        }
        impl DecisionEngine for Counting {
            fn max_tokens_per_step(&self) -> usize {
                self.inner.max_tokens_per_step()
            }
            fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
                self.with_mass.lock().unwrap().push(false);
                self.inner.run(batch)
            }
            fn run_with_mass(&mut self, batch: &DecisionBatch) -> Result<crate::contract::MassReadout> {
                self.with_mass.lock().unwrap().push(true);
                self.inner.run_with_mass(batch)
            }
        }
        let calls = Arc::new(Mutex::new(Vec::new()));
        let c = calls.clone();
        let factory: EngineFactory =
            Box::new(move || Ok(Box::new(Counting { inner: MockEngine { max_tokens: 1 << 16 }, with_mass: c }) as Box<dyn DecisionEngine>));
        let cfg = SchedConfig { token_budget: 1 << 15, max_wait: Duration::from_millis(50), queue_capacity: 64 };
        let sched = Scheduler::start(cfg, factory, Arc::new(Metrics::new())).unwrap();
        let plain = || Fake { seed: 1, prefix: 10, suffixes: vec![(3, 4), (2, 7)] };
        let debug = Fake { seed: 2, prefix: 12, suffixes: vec![(4, 5), (1, 2), (2, 9)] };
        // Alone, the debug request's masses: MockEngine's are batch-invariant.
        let mut b = DecisionBatch::default();
        debug.push_into(&mut b).unwrap();
        let want = MockEngine { max_tokens: 1 << 16 }.run_with_mass(&b).unwrap().allowed_mass.unwrap();

        let rxs = sched.submit(vec![Arc::new(plain()) as Arc<dyn Packable>, Arc::new(Debug(debug))]).unwrap();
        let got: Vec<Scored> = rxs.into_iter().map(|rx| rx.blocking_recv().unwrap().unwrap()).collect();
        assert_eq!(got[0].batch_requests, 2, "both requests in one batch");
        assert_eq!(got[0].allowed_mass.as_ref().map(Vec::len), Some(2));
        assert_eq!(got[1].allowed_mass.as_ref().unwrap(), &want);
        let rx = sched.submit(vec![Arc::new(plain()) as Arc<dyn Packable>]).unwrap().pop().unwrap();
        assert_eq!(rx.blocking_recv().unwrap().unwrap().allowed_mass, None);
        assert_eq!(*calls.lock().unwrap(), [true, false]);
    }

    #[test]
    fn engine_build_failure_is_reported() {
        let factory: EngineFactory = Box::new(|| anyhow::bail!("no GPU"));
        let err = Scheduler::start(SchedConfig::default(), factory, Arc::new(Metrics::new())).err().unwrap();
        assert!(format!("{err:#}").contains("no GPU"));
    }
}
