//! A fixed pool of CPU threads for request rendering, tokenization, planning and answer assembly (DESIGN.md §7),
//! so that this work neither blocks the async HTTP runtime nor grows without bound.
//!
//! Two queues feed the threads:
//! - **new work** ([`CpuPool::try_run`]: parsing and planning a fresh request) is bounded. When it is full the
//!   caller is refused ([`PoolError::Full`]) and the server answers 503 before reading more work into memory.
//! - **finishing work** ([`CpuPool::run`]: answer assembly for requests the engine already scored) is never
//!   refused (the admission queue bounds it) and runs first, so finished requests leave before new ones start.
//!
//! A task whose caller has gone (the HTTP client disconnected, so the handler future and with it the result
//! receiver were dropped) is skipped instead of run, and counted in [`CpuPool::skipped`].

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use tokio::sync::oneshot;

use crate::sched::affinity;

/// A queued task. It is called with `true` to run, or `false` when the caller is gone (it then only reports
/// whether it was skipped).
type Task = Box<dyn FnOnce() -> bool + Send>;

struct Queues {
    finish: VecDeque<Task>,
    fresh: VecDeque<Task>,
    closed: bool,
}

struct Shared {
    queues: Mutex<Queues>,
    ready: Condvar,
    fresh_capacity: usize,
    skipped: AtomicU64,
}

pub struct CpuPool {
    shared: Arc<Shared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    size: usize,
}

/// The task panicked (its result is lost); the server answers 500.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskPanicked;

/// Why [`CpuPool::try_run`] produced no result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolError {
    /// The new-work queue was full (or the pool is shutting down): the server answers 503.
    Full,
    /// The task panicked: the server answers 500.
    Panicked,
}

/// Default bound of the new-work queue per pool thread.
pub const DEFAULT_QUEUE_PER_THREAD: usize = 16;

impl CpuPool {
    /// `size` threads, a new-work queue of [`DEFAULT_QUEUE_PER_THREAD`] per thread, no pinning.
    pub fn new(size: usize) -> Self {
        let size = size.max(1);
        Self::with_options(size, size * DEFAULT_QUEUE_PER_THREAD, None)
    }

    /// `size` threads, at most `queue_capacity` queued new-work tasks, each thread pinned to `pin` if given.
    pub fn with_options(size: usize, queue_capacity: usize, pin: Option<Vec<usize>>) -> Self {
        let size = size.max(1);
        let shared = Arc::new(Shared {
            queues: Mutex::new(Queues { finish: VecDeque::new(), fresh: VecDeque::new(), closed: false }),
            ready: Condvar::new(),
            fresh_capacity: queue_capacity.max(1),
            skipped: AtomicU64::new(0),
        });
        let threads = (0..size)
            .map(|i| {
                let shared = shared.clone();
                let pin = pin.clone();
                std::thread::Builder::new()
                    .name(format!("sysone-cpu-{i}"))
                    .spawn(move || {
                        affinity::maybe_pin(pin.as_deref(), "a CPU worker");
                        worker(&shared)
                    })
                    .expect("spawning a CPU worker")
            })
            .collect();
        CpuPool { shared, threads: Mutex::new(threads), size }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// New-work tasks waiting for a thread.
    pub fn queue_depth(&self) -> usize {
        self.shared.queues.lock().unwrap().fresh.len()
    }

    pub fn queue_capacity(&self) -> usize {
        self.shared.fresh_capacity
    }

    /// Tasks skipped because their caller had gone.
    pub fn skipped(&self) -> u64 {
        self.shared.skipped.load(Ordering::Relaxed)
    }

    /// Run `f` on a pool thread and await its result. Never refused (finishing work).
    pub async fn run<R: Send + 'static>(&self, f: impl FnOnce() -> R + Send + 'static) -> Result<R, TaskPanicked> {
        let (task, rx) = wrap(f);
        if !self.push(task, false) {
            return Err(TaskPanicked);
        }
        rx.await.map_err(|_| TaskPanicked)
    }

    /// Queue `f` as new work and await its result, or fail at once when the new-work queue is full.
    pub async fn try_run<R: Send + 'static>(&self, f: impl FnOnce() -> R + Send + 'static) -> Result<R, PoolError> {
        let (task, rx) = wrap(f);
        if !self.push(task, true) {
            return Err(PoolError::Full);
        }
        rx.await.map_err(|_| PoolError::Panicked)
    }

    fn push(&self, task: Task, fresh: bool) -> bool {
        let mut q = self.shared.queues.lock().unwrap();
        if q.closed {
            return false;
        }
        if fresh {
            if q.fresh.len() >= self.shared.fresh_capacity {
                return false;
            }
            q.fresh.push_back(task);
        } else {
            q.finish.push_back(task);
        }
        drop(q);
        self.shared.ready.notify_one();
        true
    }
}

/// The task for `f` and the receiver of its result. The task skips `f` when the receiver is gone.
fn wrap<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> (Task, oneshot::Receiver<R>) {
    let (tx, rx) = oneshot::channel();
    let task: Task = Box::new(move || {
        if tx.is_closed() {
            return false;
        }
        // A panic drops `tx`, which the caller sees as a panic; the thread survives.
        if let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            let _ = tx.send(r);
        }
        true
    });
    (task, rx)
}

fn worker(shared: &Shared) {
    loop {
        let task = {
            let mut q = shared.queues.lock().unwrap();
            loop {
                if let Some(t) = q.finish.pop_front().or_else(|| q.fresh.pop_front()) {
                    break t;
                }
                if q.closed {
                    return;
                }
                q = shared.ready.wait(q).unwrap();
            }
        };
        if !task() {
            shared.skipped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        // Queued tasks still run (their callers may be waiting); the threads exit once both queues are empty.
        self.shared.queues.lock().unwrap().closed = true;
        self.shared.ready.notify_all();
        // The last owner may be a task running on a pool thread; that thread exits on its own once the task ends.
        let me = std::thread::current().id();
        for t in self.threads.lock().unwrap().drain(..) {
            if t.thread().id() != me {
                let _ = t.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[tokio::test]
    async fn runs_tasks_and_survives_panics() {
        let pool = CpuPool::new(2);
        assert_eq!(pool.run(|| 6 * 7).await, Ok(42));
        assert_eq!(pool.run(|| -> i32 { panic!("boom") }).await, Err(TaskPanicked));
        assert_eq!(pool.try_run(|| -> i32 { panic!("boom") }).await, Err(PoolError::Panicked));
        let name = pool.run(|| std::thread::current().name().unwrap().to_string()).await.unwrap();
        assert!(name.starts_with("sysone-cpu-"));
        let all = futures_join(&pool).await;
        assert_eq!(all, (0..16).map(|i| i * 2).collect::<Vec<_>>());
    }

    async fn futures_join(pool: &CpuPool) -> Vec<i32> {
        let mut out = Vec::new();
        for i in 0..16 {
            out.push(pool.try_run(move || i * 2).await.unwrap());
        }
        out
    }

    /// Occupy the pool's single thread until the returned sender is used.
    async fn block(pool: &Arc<CpuPool>) -> mpsc::Sender<()> {
        let (go, wait) = mpsc::channel::<()>();
        let (started_tx, started) = oneshot::channel();
        let p = pool.clone();
        tokio::spawn(async move {
            p.run(move || {
                let _ = started_tx.send(());
                let _ = wait.recv();
            })
            .await
        });
        started.await.unwrap();
        go
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn new_work_is_bounded_and_finishing_work_is_not() {
        let pool = Arc::new(CpuPool::with_options(1, 2, None));
        let go = block(&pool).await;
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut queued = Vec::new();
        for i in 0..2 {
            let (o, p) = (order.clone(), pool.clone());
            queued.push(tokio::spawn(async move {
                p.try_run(move || {
                    o.lock().unwrap().push(if i == 0 { "fresh0" } else { "fresh1" });
                    i
                })
                .await
            }));
        }
        while pool.queue_depth() < 2 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(pool.try_run(|| 9).await, Err(PoolError::Full));
        // Finishing work is accepted beyond the bound, and runs before the queued new work.
        let (o, p) = (order.clone(), pool.clone());
        let fin = tokio::spawn(async move { p.run(move || o.lock().unwrap().push("finish")).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        go.send(()).unwrap();
        fin.await.unwrap().unwrap();
        for (i, t) in queued.into_iter().enumerate() {
            assert_eq!(t.await.unwrap(), Ok(i));
        }
        assert_eq!(*order.lock().unwrap(), ["finish", "fresh0", "fresh1"]);
        assert_eq!(pool.try_run(|| 5).await, Ok(5));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tasks_of_departed_callers_are_skipped() {
        let pool = Arc::new(CpuPool::with_options(1, 8, None));
        let go = block(&pool).await;
        let ran = Arc::new(AtomicU64::new(0));
        let (r, p) = (ran.clone(), pool.clone());
        let caller = tokio::spawn(async move { p.try_run(move || r.fetch_add(1, Ordering::SeqCst)).await });
        while pool.queue_depth() < 1 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        // The caller goes away (as a handler future does when its client disconnects).
        caller.abort();
        let _ = caller.await;
        go.send(()).unwrap();
        assert_eq!(pool.try_run(|| 1).await, Ok(1));
        assert_eq!(ran.load(Ordering::SeqCst), 0);
        assert_eq!(pool.skipped(), 1);
    }

    #[test]
    fn pinned_workers_run_on_the_given_cores() {
        let Some(allowed) = affinity::allowed_cores() else { return };
        let one = vec![*allowed.last().unwrap()];
        let pool = CpuPool::with_options(2, 4, Some(one.clone()));
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let got = rt.block_on(pool.run(affinity::allowed_cores)).unwrap();
        assert_eq!(got, Some(one));
    }
}
