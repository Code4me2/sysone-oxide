//! The GPU engine: the packed tree forward over a [`DecisionBatch`](crate::contract::DecisionBatch)
//! (DESIGN.md §6). Owner: W4 (Phase 1), I1 (Phase 2a).
//!
//! - Loading: [`config`] (refuses what v1 cannot serve), [`loader`] (safetensors without a page-cache copy),
//!   [`weights`] (device weights, fused projections, bf16 or NVFP4 per projection, slot head).
//! - The forward: [`forward::build`] writes the 32 layers as a [`graph::StepGraph`] of [`graph::GraphOp`]s;
//!   [`graph`] plans one activation arena per T bucket, runs it eagerly or captures and replays it as a CUDA
//!   graph; [`kernels`] maps the ops onto the cuTile kernels of [`crate::kernels`]; [`gemm`] runs the bf16 GEMMs on
//!   cuBLAS; [`meta`] uploads the per-step batch metadata.
//! - [`gpu::GpuEngine`] ties these together as a [`DecisionEngine`](crate::contract::DecisionEngine).
//! - Support: [`goldens`] (reference outputs for tests and benchmarks), [`platform`] (platform profile, idle-GPU
//!   guard), [`cpu_ref`] (slow f32 CPU reference over packed trees, for tests).
//!
//! See docs/ARCHITECTURE.md §3 (packed tree, buckets, graphs, readout), §4.5 (files and gotchas) and §5 (which
//! kernel or library runs each op).

pub mod config;
pub mod forward;
pub mod gemm;
pub mod goldens;
pub mod gpu;
pub mod graph;
pub mod kernels;
pub mod loader;
pub mod meta;
pub mod platform;
pub mod weights;
pub mod cpu_ref;
