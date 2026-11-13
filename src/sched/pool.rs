//! A fixed pool of CPU threads for request rendering, tokenization, planning and answer assembly (DESIGN.md §7),
//! so that this work neither blocks the async HTTP runtime nor grows without bound.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use tokio::sync::oneshot;

type Task = Box<dyn FnOnce() + Send>;

pub struct CpuPool {
    tx: Mutex<Option<Sender<Task>>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    size: usize,
}

/// The task panicked (its result is lost); the server answers 500.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskPanicked;

impl CpuPool {
    pub fn new(size: usize) -> Self {
        let size = size.max(1);
        let (tx, rx) = channel::<Task>();
        let rx = Arc::new(Mutex::new(rx));
        let threads = (0..size)
            .map(|i| {
                let rx: Arc<Mutex<Receiver<Task>>> = rx.clone();
                std::thread::Builder::new()
                    .name(format!("sysone-cpu-{i}"))
                    .spawn(move || {
                        loop {
                            let task = rx.lock().unwrap().recv();
                            match task {
                                Ok(t) => t(),
                                Err(_) => return,
                            }
                        }
                    })
                    .expect("spawning a CPU worker")
            })
            .collect();
        CpuPool { tx: Mutex::new(Some(tx)), threads: Mutex::new(threads), size }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Run `f` on a pool thread and await its result.
    pub async fn run<R: Send + 'static>(&self, f: impl FnOnce() -> R + Send + 'static) -> Result<R, TaskPanicked> {
        let (tx, rx) = oneshot::channel();
        let task: Task = Box::new(move || {
            // A panic drops `tx`, which the caller sees as TaskPanicked; the thread survives.
            if let Ok(r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
                let _ = tx.send(r);
            }
        });
        let sent = self.tx.lock().unwrap().as_ref().map(|s| s.send(task).is_ok()).unwrap_or(false);
        if !sent {
            return Err(TaskPanicked);
        }
        rx.await.map_err(|_| TaskPanicked)
    }
}

impl Drop for CpuPool {
    fn drop(&mut self) {
        self.tx.lock().unwrap().take();
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

    #[tokio::test]
    async fn runs_tasks_and_survives_panics() {
        let pool = CpuPool::new(2);
        assert_eq!(pool.run(|| 6 * 7).await, Ok(42));
        assert_eq!(pool.run(|| -> i32 { panic!("boom") }).await, Err(TaskPanicked));
        let name = pool.run(|| std::thread::current().name().unwrap().to_string()).await.unwrap();
        assert!(name.starts_with("sysone-cpu-"));
        let all = futures_join(&pool).await;
        assert_eq!(all, (0..16).map(|i| i * 2).collect::<Vec<_>>());
    }

    async fn futures_join(pool: &CpuPool) -> Vec<i32> {
        let mut out = Vec::new();
        for i in 0..16 {
            out.push(pool.run(move || i * 2).await.unwrap());
        }
        out
    }
}
