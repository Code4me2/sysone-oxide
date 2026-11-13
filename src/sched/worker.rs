//! The GPU worker: the single thread that owns the [`DecisionEngine`] (and so the CUDA context). It runs one packed
//! batch at a time and sends each request its slot-logit rows (DESIGN.md §7).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::time::Instant;

use crate::contract::{DecisionEngine, MAX_SLOTS, MassReadout};
use crate::sched::metrics::Metrics;
use crate::sched::{Packed, Scored};

/// Run batches until the batcher hangs up. A batch holding a request that wants `allowed_token_mass` runs through
/// [`DecisionEngine::run_with_mass`], the others through [`DecisionEngine::run`]. `in_flight` (incremented by the batcher per batch sent) drops when a
/// step ends, which tells the batcher the engine is free.
pub fn run(mut engine: Box<dyn DecisionEngine>, rx: Receiver<Packed>, metrics: Arc<Metrics>, in_flight: Arc<AtomicUsize>) {
    for Packed { batch, jobs, wants_mass } in rx {
        let n_jobs = jobs.len();
        let start = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if wants_mass {
                engine.run_with_mass(&batch)
            } else {
                engine.run(&batch).map(|logits| MassReadout { logits, allowed_mass: None })
            }
        }))
        .unwrap_or_else(|_| Err(anyhow::anyhow!("engine panicked")));
        let step = start.elapsed();
        in_flight.fetch_sub(1, Ordering::AcqRel);
        metrics.record_batch(batch.num_tokens(), jobs.len(), batch.num_branches(), step);
        if result.is_ok()
            && let Some(bucket) = engine.last_bucket_tokens()
        {
            metrics.record_bucket(batch.num_tokens(), bucket);
        }
        let result = result.and_then(|r| {
            let (l, want) = (&r.logits, batch.num_branches());
            anyhow::ensure!(
                l.rows == want && l.data.len() >= want * MAX_SLOTS,
                "engine returned {} rows ({} values) for {want} branches",
                l.rows,
                l.data.len()
            );
            if let Some(m) = &r.allowed_mass {
                anyhow::ensure!(m.len() == want, "engine returned {} masses for {want} branches", m.len());
            }
            Ok(r)
        });
        match result {
            Ok(MassReadout { logits, allowed_mass }) => {
                for (job, first) in jobs {
                    let n = job.work.num_branches();
                    let wait = start.saturating_duration_since(job.admitted);
                    metrics.record_queue_wait(wait);
                    let scored = Scored {
                        logits: logits.data[first * MAX_SLOTS..(first + n) * MAX_SLOTS].to_vec(),
                        allowed_mass: allowed_mass.as_ref().map(|m| m[first..first + n].to_vec()),
                        queue_wait: wait,
                        step_started: start,
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
