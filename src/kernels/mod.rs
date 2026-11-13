//! GPU kernels (DESIGN.md §6.4), written with cuTile (`cutile`) and JIT-compiled on first launch. Owner: W3.
//!
//! | Module | Kernels |
//! |---|---|
//! | [`norm`] | zero-centred RMSNorm, fused residual add + norm |
//! | [`pointwise`] | `silu_mul` (SwiGLU), `gather_rows` |
//! | [`attention`] | `qk_norm_rope`, segment attention `seg_attn` (mask from [`attention::AttnPlan`]) |
//! | [`gdn`] | GatedDeltaNet mixer: tree conv1d, gates, chunked delta-rule scan over segment trees, gated norm |
//! | [`readout`] | slot-logit GEMM (tested; the engine runs the readout on cuBLAS) |
//! | [`nvfp4`] | NVFP4 activation quantization and block-scaled FP4 GEMMs (not yet used by the forward) |
//!
//! The bf16 projection GEMMs are not here: the engine runs them on cuBLAS (`crate::engine::gemm`). There is no
//! SIMT (cuda-oxide) module yet. Launchers that take raw device pointers are `unsafe`; each module states its
//! safety contract and the reference rounding points it reproduces. See docs/ARCHITECTURE.md §4.6 and §5.
pub mod gdn;
pub mod simt;

pub mod attention;
pub mod norm;
pub mod pointwise;
pub mod readout;

pub mod nvfp4;
