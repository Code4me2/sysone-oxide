//! The packed tree forward over a `DecisionBatch` (DESIGN.md §6): config, weights, StepGraph, T-bucket CUDA
//! graphs, readout. Owner: W4.

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
