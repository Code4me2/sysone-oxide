//! sysone: an HTTP inference server for System One decision models (elephant-1) on cuda-rust, a drop-in for qev's
//! `/v1/systemone`.
//!
//! A request (a state plus questions) becomes a [`frontend::Plan`]: one shared prefix and one suffix per branch.
//! The scheduler packs many plans into one [`contract::DecisionBatch`] (a packed tree of segments), the GPU engine
//! scores every branch in one forward, and the frontend turns the slot logits back into answers.
//!
//! | Module | Role |
//! |---|---|
//! | [`api`] | HTTP routes, request/response schema, qev-identical errors |
//! | [`frontend`] | Rendering, chat template, tokenization, prefix split, branch planning, answer assembly (CPU) |
//! | [`sched`] | Admission queue, token-budget batcher, GPU worker thread, CPU pool, metrics |
//! | [`contract`] | `DecisionBatch` / `Segment` / `SlotLogits` and the `DecisionEngine` trait between the others |
//! | [`engine`] | The GPU engine: loader, weights, StepGraph forward, T-bucket CUDA graphs, readout; CPU reference |
//! | [`kernels`] | cuTile GPU kernels: norms, pointwise, attention, GatedDeltaNet, readout, NVFP4 |
//! | [`tolerances`] | Test thresholds from `tests/tolerances.toml` |
//!
//! Start with docs/ARCHITECTURE.md (request lifecycle, core concepts, module map); docs/DESIGN.md has the
//! rationale.

pub mod api;
pub mod contract;
pub mod engine;
pub mod frontend;
pub mod kernels;
pub mod sched;
pub mod tolerances;
