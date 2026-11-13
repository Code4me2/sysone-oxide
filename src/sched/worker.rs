//! The GPU worker: the single thread that owns the [`DecisionEngine`] (and so the CUDA context). It runs one packed
//! batch at a time and sends each request its slot-logit rows (DESIGN.md §7).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::time::Instant;

use crate::contract::{DecisionEngine, MAX_SLOTS};
use crate::sched::metrics::Metrics;
use crate::sched::{Packed, Scored};

/// Run batches until the batcher hangs up. `in_flight` (incremented by the batcher per batch sent) drops when a
/// step ends, which tells the batcher the engine is free.
pub fn run(mut engine: Box<dyn DecisionEngine>, rx: Receiver<Packed>, metrics: Arc<Metrics>, in_flight: Arc<AtomicUsize>) {
    for Packed { batch, jobs } in rx {
        let n_jobs = jobs.len();
        let start = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.run(&batch)))
            .unwrap_or_else(|_| Err(anyhow::anyhow!("engine panicked")));
        let step = start.elapsed();
        in_flight.fetch_sub(1, Ordering::AcqRel);
        metrics.record_batch(batch.num_tokens(), jobs.len(), batch.num_branches(), step);
        let result = result.and_then(|l| {
            let want = batch.num_branches();
            anyhow::ensure!(
                l.rows == want && l.data.len() >= want * MAX_SLOTS,
                "engine returned {} rows ({} values) for {want} branches",
                l.rows,
                l.data.len()
            );
            Ok(l)
        });
        match result {
            Ok(logits) => {
                for (job, first) in jobs {
                    let n = job.work.num_branches();
                    let wait = start.saturating_duration_since(job.admitted);
                    metrics.record_queue_wait(wait);
                    let scored = Scored {
                        logits: logits.data[first * MAX_SLOTS..(first + n) * MAX_SLOTS].to_vec(),
                        queue_wait: wait,
                        step,
                        batch_tokens: batch.num_tokens(),
                        batch_requests: n_jobs,
                    };
                    let _ = job.reply.send(Ok(scored));
                }
            }
            Err(e) => {
                let msg = format!("engine step failed: {e:#}");
                for (job, _) in jobs {
                    let _ = job.reply.send(Err(msg.clone()));
                }
            }
        }
    }
}
