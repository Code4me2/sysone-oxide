//! GPU kernels (DESIGN.md §6.4): `tile` (cutile-rs), `simt` (cuda-oxide PTX), and the GEMM backends. Owner: W3.
pub mod gdn;

pub mod attention;
pub mod norm;
pub mod pointwise;
pub mod readout;
