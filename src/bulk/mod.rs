//! Bulk tagging at full throughput: `sysone bulk` reads a JSONL file of `/v1/systemone` requests and writes one
//! result line per request, keyed by the job's id (docs/BULK.md).
//!
//! The answers are the server's: every job goes through the same validation, planning and answer assembly
//! ([`crate::api::server::prepare`] and [`crate::api::server::finish`]), so a result's `status` and `body` are
//! what `POST /v1/systemone` returns for that body (or its error body). What differs is how work reaches the
//! GPUs: no HTTP, a window of planned requests packed largest-first to fill each step's bucket ([`pack`]), and
//! one engine per GPU fed from one queue, so a slower or smaller GPU simply takes fewer steps ([`run`],
//! [`worker`]).
//!
//! | Module | Role |
//! |---|---|
//! | [`job`] | Input lines and result lines |
//! | [`output`] | The results file: lock, resume, crash-safe appends |
//! | [`pack`] | Choosing each step (window, bucket ladder, throughput curve) and the padding simulations |
//! | [`run`] | The coordinator: reader, planners, dispatcher, writer; the summary |
//! | [`worker`] | Engines per GPU: worker processes (`sysone bulk worker`) or threads; per-device sizing |
//! | [`wire`] | The coordinator ↔ worker-process pipe protocol |
//! | [`synth`] | Synthetic requests (loadgen's) and corpora with a length mix |
//! | [`compare`] | Two result files by id: statuses, fields, probability differences |
//! | [`cli`] | `sysone bulk [run] / gen / simulate / compare / worker` |

pub mod cli;
pub mod compare;
pub mod job;
pub mod output;
pub mod pack;
pub mod run;
pub mod synth;
pub mod wire;
pub mod worker;
