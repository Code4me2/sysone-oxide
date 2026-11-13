//! The batcher: packs whole requests from the admission queue into one [`DecisionBatch`] until the token budget
//! is reached or the oldest request has waited `max_wait` (DESIGN.md §7). A request is never split; one larger
//! than the budget runs alone.
//!
//! It runs on its own thread and hands finished batches to the GPU worker through a one-slot channel, so batch
//! n+1 is packed while the engine runs batch n. While the engine is busy, `max_wait` does not close a batch: it
//! keeps filling until the engine is free or the budget is reached (see [`run`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::SyncSender;
use std::time::{Duration, Instant};

use crate::contract::DecisionBatch;
use crate::sched::queue::{AdmissionQueue, Pop};
use crate::sched::{Job, Packed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchPolicy {
    /// Close a batch once it holds this many tokens; a request that would cross it waits for the next batch.
    pub token_budget: usize,
    /// Close a batch once its oldest request was admitted this long ago (if it is not full before).
    pub max_wait: Duration,
    /// The engine's `max_tokens_per_step`: requests over it are refused.
    pub max_tokens: usize,
}

/// The batch under construction.
pub struct Packer {
    budget: usize,
    batch: DecisionBatch,
    jobs: Vec<(Job, usize)>,
}

impl Packer {
    pub fn new(budget: usize) -> Self {
        Packer { budget, batch: DecisionBatch::default(), jobs: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    pub fn tokens(&self) -> usize {
        self.batch.num_tokens()
    }

    /// Whether `job` may join: always into an empty batch (an oversize request runs alone), otherwise only
    /// within the budget.
    pub fn fits(&self, job: &Job) -> bool {
        self.jobs.is_empty() || self.tokens() + job.work.num_tokens() <= self.budget
    }

    pub fn full(&self) -> bool {
        self.tokens() >= self.budget
    }

    /// Append a job (the caller checked [`fits`](Self::fits)). A job that cannot be packed is answered with the
    /// error and dropped; the batch is left as it was.
    pub fn add(&mut self, job: Job) {
        let before = (self.batch.token_ids.len(), self.batch.segments.len(), self.batch.answer_rows.len());
        match job.work.push_into(&mut self.batch) {
            Ok(first) => self.jobs.push((job, first)),
            Err(e) => {
                self.batch.token_ids.truncate(before.0);
                self.batch.segments.truncate(before.1);
                self.batch.answer_rows.truncate(before.2);
                self.batch.answer_slots.truncate(before.2);
                let _ = job.reply.send(Err(format!("packing request: {e:#}")));
            }
        }
    }

    /// The finished batch, checked against the contract. On a violation every job gets the error.
    pub fn take(&mut self) -> Option<Packed> {
        let batch = std::mem::take(&mut self.batch);
        let jobs = std::mem::take(&mut self.jobs);
        if jobs.is_empty() {
            return None;
        }
        if let Err(e) = batch.validate() {
            for (job, _) in jobs {
                let _ = job.reply.send(Err(format!("invalid batch: {e:#}")));
            }
            return None;
        }
        Some(Packed { batch, jobs })
    }
}

/// Whether a dequeued job should be packed: requests whose client is gone are dropped, and requests over the
/// engine's step size are refused.
fn admit(job: Job, max_tokens: usize) -> Option<Job> {
    if job.reply.is_closed() {
        return None;
    }
    let n = job.work.num_tokens();
    if n > max_tokens {
        let _ = job.reply.send(Err(format!("request of {n} tokens exceeds the engine's {max_tokens} per step")));
        return None;
    }
    Some(job)
}

/// While the engine is busy past a batch's deadline, the batcher re-checks it this often.
const BUSY_POLL: Duration = Duration::from_micros(200);

/// The batcher loop. Returns when the queue is closed and drained, or the worker is gone.
///
/// `in_flight` counts batches handed to the worker and not yet finished (the worker decrements it after each
/// step). A batch is closed when it is full, or when its deadline has passed *and* the engine is idle: while the
/// engine is still running batch n there is nothing to gain by closing batch n+1, so it keeps filling. Requests
/// already queued are taken even past the deadline, up to the budget.
pub fn run(queue: Arc<AdmissionQueue<Job>>, policy: BatchPolicy, tx: SyncSender<Packed>, in_flight: Arc<AtomicUsize>) {
    let mut carry: Option<Job> = None;
    let mut packer = Packer::new(policy.token_budget);
    loop {
        let first = match carry.take() {
            Some(j) => j,
            None => match queue.pop(None) {
                Pop::Item(j) => j,
                Pop::Closed => return,
                Pop::Timeout => continue,
            },
        };
        let Some(first) = admit(first, policy.max_tokens) else { continue };
        let deadline = first.admitted + policy.max_wait;
        packer.add(first);
        while !packer.full() {
            let now = Instant::now();
            let wait_until = if now < deadline {
                deadline
            } else if in_flight.load(Ordering::Acquire) > 0 {
                now + BUSY_POLL
            } else {
                now
            };
            match queue.pop(Some(wait_until)) {
                Pop::Item(j) => {
                    let Some(j) = admit(j, policy.max_tokens) else { continue };
                    if packer.fits(&j) {
                        packer.add(j);
                    } else {
                        carry = Some(j);
                        break;
                    }
                }
                Pop::Timeout => {
                    if Instant::now() >= deadline && in_flight.load(Ordering::Acquire) == 0 {
                        break;
                    }
                }
                Pop::Closed => break,
            }
        }
        let Some(packed) = packer.take() else { continue };
        in_flight.fetch_add(1, Ordering::AcqRel);
        if let Err(e) = tx.send(packed) {
            for (job, _) in e.0.jobs {
                let _ = job.reply.send(Err("GPU worker stopped".into()));
            }
            if let Some(j) = carry.take() {
                let _ = j.reply.send(Err("GPU worker stopped".into()));
            }
            queue.close();
            while let Pop::Item(j) = queue.pop(None) {
                let _ = j.reply.send(Err("GPU worker stopped".into()));
            }
            return;
        }
    }
}
