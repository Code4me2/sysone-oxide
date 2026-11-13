//! The packed tree forward over a `DecisionBatch` (DESIGN.md §6): config, weights, StepGraph, T-bucket CUDA
//! graphs, readout. Owner: W4.

pub mod config;
pub mod gemm;
pub mod graph;
pub mod loader;
pub mod meta;
pub mod platform;
pub mod weights;
