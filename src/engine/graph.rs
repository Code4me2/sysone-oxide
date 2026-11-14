//! The StepGraph IR (ported from grout's `model.rs`, DESIGN §6.2, §6.3): one forward as a list of ops over
//! SSA values, planned once into a single activation arena per T bucket, executed eagerly or captured as a
//! CUDA graph per bucket.
//!
//! Differences from grout:
//! - values carry a [`GraphDType`] (bf16 activations, f32 gates/state/logits, u8 + f8e4m3 for NVFP4);
//! - shapes are symbolic in the packed token count ([`Dim::Tokens`]), the answer count and the segment count,
//!   and are made concrete per [`Bucket`];
//! - the tensor pool is an offset plan into one arena (liveness-based first fit), so every value has a fixed
//!   device address per bucket and a captured graph replays without any allocation;
//! - no decode path, KV cache or sampling.
//!
//! Ops call into a [`Kernels`] implementation (W3's cuTile kernels). The executor implements the ops that
//! exist now itself: the embedding gather (cutile-kernels) and bf16 GEMMs (cuBLAS, [`Blas`]).

pub mod tensor;

use crate::engine::config::ModelConfig;
use crate::engine::gemm::Blas;
use crate::engine::meta::DeviceMeta;
use crate::engine::weights::{Layer, Linear, ModelWeights};
use anyhow::{Context as _, Result, anyhow, bail, ensure};
use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext, GraphNode};
use cuda_async::error::DeviceError;
use cuda_core::Stream;
use std::collections::HashMap;
use std::sync::Arc;
use tensor::{DevBuf, DevTensor, GraphDType};

// ---------------------------------------------------------------------------------------------------------
// Shapes and buckets.

/// One dimension of a value's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Fixed(usize),
    /// Packed tokens of the bucket (including padding).
    Tokens,
    /// Answer rows of the bucket.
    Answers,
    /// Segment slots of the bucket (including the padding segment).
    Segments,
}

/// The capacities one captured graph is planned for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bucket {
    pub tokens: usize,
    pub answers: usize,
    pub segments: usize,
}

impl Bucket {
    pub fn dim(&self, d: Dim) -> usize {
        match d {
            Dim::Fixed(n) => n,
            Dim::Tokens => self.tokens,
            Dim::Answers => self.answers,
            Dim::Segments => self.segments,
        }
    }
}

/// The T buckets of an engine (DESIGN §6.2). Bucket sizes come from configuration or the platform profile.
#[derive(Clone, Debug)]
pub struct BucketPlan {
    pub buckets: Vec<Bucket>,
}

impl BucketPlan {
    /// The default token buckets: 256, 512, …, 32768.
    pub const DEFAULT_TOKENS: [usize; 8] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

    /// Buckets for the given token sizes. A bucket of T tokens holds up to `min(T, max_answers)` answers and
    /// `min(T, max_segments) + 1` segments (the extra one is the padding segment).
    pub fn new(tokens: &[usize], max_answers: usize, max_segments: usize) -> Result<Self> {
        let mut t = tokens.to_vec();
        t.sort_unstable();
        t.dedup();
        ensure!(!t.is_empty() && t[0] > 0, "no token buckets");
        let buckets = t
            .into_iter()
            .map(|n| Bucket { tokens: n, answers: n.min(max_answers).max(1), segments: n.min(max_segments) + 1 })
            .collect();
        Ok(Self { buckets })
    }

    /// Smallest bucket that fits a batch, if any.
    pub fn pick(&self, tokens: usize, answers: usize, segments: usize) -> Option<Bucket> {
        self.buckets.iter().copied().find(|b| {
            let pad = (b.tokens > tokens) as usize;
            tokens <= b.tokens && answers <= b.answers && segments + pad <= b.segments
        })
    }

    pub fn largest(&self) -> Bucket {
        *self.buckets.last().unwrap()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorSpec {
    pub dtype: GraphDType,
    pub shape: Vec<Dim>,
}

impl TensorSpec {
    pub fn new(dtype: GraphDType, shape: Vec<Dim>) -> Self {
        Self { dtype, shape }
    }

    pub fn concrete(&self, b: &Bucket) -> Vec<usize> {
        self.shape.iter().map(|d| b.dim(*d)).collect()
    }

    pub fn bytes(&self, b: &Bucket) -> usize {
        self.concrete(b).iter().product::<usize>() * self.dtype.size_in_bytes()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub usize);

// ---------------------------------------------------------------------------------------------------------
// Weights.

/// Per-layer weight slots (both layer kinds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerWeight {
    InputNorm,
    PostNorm,
    /// GDN fused `[qkv; z; a; b]`.
    GdnInProj,
    GdnConv,
    GdnALog,
    GdnDtBias,
    GdnNorm,
    GdnOutProj,
    AttnQ,
    /// Fused `[k; v]`.
    AttnKv,
    AttnO,
    QNorm,
    KNorm,
    /// Fused `[gate; up]`.
    MlpGateUp,
    MlpDown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightRef {
    Embed,
    FinalNorm,
    SlotHead,
    LmHead,
    Layer(usize, LayerWeight),
}

/// Resolves [`WeightRef`]s to device weights. [`ModelWeights`] implements it; tests use small stores.
pub trait WeightStore: Sync {
    fn tensor(&self, w: WeightRef) -> Result<&DevTensor>;
    fn linear(&self, w: WeightRef) -> Result<&Linear>;
}

impl WeightStore for ModelWeights {
    fn tensor(&self, w: WeightRef) -> Result<&DevTensor> {
        use LayerWeight as L;
        Ok(match w {
            WeightRef::Embed => &self.embed,
            WeightRef::FinalNorm => &self.final_norm,
            WeightRef::SlotHead => &self.slot_head.as_ref().context("slot head not loaded")?.w,
            WeightRef::LmHead => self.lm_head.as_ref().context("full lm_head not loaded")?,
            WeightRef::Layer(i, s) => {
                let l = self.layers.get(i).with_context(|| format!("layer {i}"))?;
                match (l, s) {
                    (_, L::InputNorm) => l.input_norm(),
                    (_, L::PostNorm) => l.post_norm(),
                    (Layer::Gdn(g), L::GdnConv) => &g.conv_w,
                    (Layer::Gdn(g), L::GdnALog) => &g.a_log,
                    (Layer::Gdn(g), L::GdnDtBias) => &g.dt_bias,
                    (Layer::Gdn(g), L::GdnNorm) => &g.norm_w,
                    (Layer::Attn(a), L::QNorm) => &a.q_norm,
                    (Layer::Attn(a), L::KNorm) => &a.k_norm,
                    _ => bail!("{w:?} is not a plain tensor of layer {i}"),
                }
            }
        })
    }

    fn linear(&self, w: WeightRef) -> Result<&Linear> {
        use LayerWeight as L;
        let WeightRef::Layer(i, s) = w else { bail!("{w:?} is not a projection") };
        let l = self.layers.get(i).with_context(|| format!("layer {i}"))?;
        Ok(match (l, s) {
            (_, L::MlpGateUp) => &l.mlp().gate_up,
            (_, L::MlpDown) => &l.mlp().down,
            (Layer::Gdn(g), L::GdnInProj) => &g.in_proj,
            (Layer::Gdn(g), L::GdnOutProj) => &g.out_proj,
            (Layer::Attn(a), L::AttnQ) => &a.q_proj,
            (Layer::Attn(a), L::AttnKv) => &a.kv_proj,
            (Layer::Attn(a), L::AttnO) => &a.o_proj,
            _ => bail!("{w:?} is not a projection of layer {i}"),
        })
    }
}

// ---------------------------------------------------------------------------------------------------------
// Ops.

/// A value, or a column range of a rank-2 value (fused GEMM outputs are consumed by column slice).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRef {
    Value(ValueId),
    Cols { v: ValueId, start: usize, len: usize },
}

impl TensorRef {
    pub fn value(self) -> ValueId {
        match self {
            Self::Value(v) | Self::Cols { v, .. } => v,
        }
    }
}

impl From<ValueId> for TensorRef {
    fn from(v: ValueId) -> Self {
        Self::Value(v)
    }
}

/// The per-layer op list of DESIGN §6.3 and the NVFP4 MLP path of §6.6. Shapes are documented with
/// T = bucket tokens, Q = bucket answers.
///
/// The ops follow the W3 kernels' real interfaces (Phase 2a, I1):
/// - the residual stream is updated **in place** by the fused add + norm (`residual_out` aliases `residual`'s
///   storage; the plan enforces that nothing reads `residual` afterwards);
/// - a GDN layer's q/k/g/beta and scan intermediates live in the kernels' own scratch ([`Kernels`] owns it), so
///   [`GraphOp::GdnPrep`] has no graph outputs and [`GraphOp::GdnChunkScan`] reads that scratch;
/// - the attention gate and v are read in place from the `q_proj` and `[k; v]` outputs.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphOp {
    /// `out [T, hidden] bf16 = embed[token_ids]` (executor, cutile-kernels).
    Embedding { out: ValueId },
    /// Zero-centred RMSNorm `(x · rsqrt(mean x² + eps)) · (1 + w)`, fp32 statistics, bf16 out.
    ZcRmsNorm { x: TensorRef, w: WeightRef, out: ValueId },
    /// `residual_out = bf16(residual + x); out = zc_rmsnorm(residual_out, w)`, in place: `residual_out` is
    /// `residual`'s storage.
    AddZcRmsNorm { residual: ValueId, x: TensorRef, w: WeightRef, out: ValueId, residual_out: ValueId },
    /// `out [T, N] = x [T, K] · Wᵀ` for a bf16 [`Linear`] (executor, cuBLAS). `out` may be bf16 or f32.
    Gemm { x: TensorRef, w: WeightRef, out: ValueId },
    /// NVFP4 activation quantization of `x` under `w`'s `input_global_scale`: `q [T, K/2] u8`, `s [T, K/16]`.
    QuantizeNvfp4 { x: TensorRef, w: WeightRef, q: ValueId, s: ValueId },
    /// `zc_rmsnorm` with the NVFP4 quantization fused into the epilogue (`w` names the consuming projection).
    ZcRmsNormNvfp4 { x: TensorRef, norm: WeightRef, w: WeightRef, q: ValueId, s: ValueId },
    /// `AddZcRmsNorm` with fused NVFP4 quantization of the normed output (in place like `AddZcRmsNorm`).
    AddZcRmsNormNvfp4 { residual: ValueId, x: TensorRef, norm: WeightRef, w: WeightRef, q: ValueId, s: ValueId,
                        residual_out: ValueId },
    /// Block-scaled FP4 GEMM, per-part epilogue `1 / (a_gs · w_gs)`, bf16 out.
    GemmNvfp4 { q: TensorRef, s: TensorRef, w: WeightRef, out: ValueId },
    /// F1: the gate/up FP4 GEMM (`w`, rows interleaved for it at load) with SiLU·mul and the NVFP4 quantization
    /// of the product under `w_down`'s input scale in its epilogue: `q_out [T, I/2]`, `s_out [T, I/16]`. Equal
    /// byte for byte to `GemmNvfp4` (bf16 `[T, 2I]`) followed by `SiluMulNvfp4`; the bf16 gate/up output never exists.
    GemmNvfp4SwiGlu { q: TensorRef, s: TensorRef, w: WeightRef, w_down: WeightRef, q_out: ValueId, s_out: ValueId },
    /// `out [T, I] = silu(gu[:, :I]) · gu[:, I:]`.
    SiluMul { gu: TensorRef, out: ValueId },
    /// `SiluMul` with fused NVFP4 quantization under `w`'s input scale.
    SiluMulNvfp4 { gu: TensorRef, w: WeightRef, q: ValueId, s: ValueId },
    /// Causal depthwise conv1d (kernel 4) over segments, taps from the parent prefix, then SiLU.
    /// `x`: the `[T, 8192]` qkv columns of the fused GDN projection; `out [T, 8192]` bf16.
    TreeConv1dSilu { layer: usize, x: TensorRef, out: ValueId },
    /// From the conv output (q | k columns) and the `a`, `b` columns of the fused projection: l2-normalised q, k
    /// and the gates `g = -exp(A_log)·softplus(a + dt_bias)`, `beta = sigmoid(b)`, all f32, into the GDN scratch.
    GdnPrep { layer: usize, conv: TensorRef, a: TensorRef, b: TensorRef },
    /// Chunked gated delta rule over segments: chunk inverse, WY transform, prefix scan (writing each prefix's
    /// final state), suffix scan (from the parent's state). Reads the gates (and, on the cuTile backend, the f32
    /// q/k) from the scratch and `conv` (the whole `[T, 8192]` conv output: the SIMT backend reads q, k and v from
    /// it, the cuTile one only its v columns); `out [T, 4096]` bf16.
    GdnChunkScan { layer: usize, conv: TensorRef, out: ValueId },
    /// Gated RMSNorm per v-head: `bf16(norm(o)) · w · silu(z)`.
    GatedRmsNorm { layer: usize, o: TensorRef, z: TensorRef, out: ValueId },
    /// q/k zero-centred norm and partial RoPE: `q [T, 16·256]` from the query halves of `qg [T, 16·512]`,
    /// `k [T, 4·256]` from the k heads of `kv [T, 8·256]`.
    QkNormRope { layer: usize, qg: TensorRef, kv: TensorRef, q: ValueId, k: ValueId },
    /// Segment attention (non-causal, ≤ 2 KV ranges, LSE merge), GQA, `· sigmoid(gate)` epilogue. v is read from
    /// `kv`, the gate from `qg`; `out [T, 4096]`.
    SegAttn { layer: usize, q: TensorRef, k: TensorRef, kv: TensorRef, qg: TensorRef, out: ValueId },
    /// `out [Q, C] = src[answer_rows]`.
    GatherRows { src: TensorRef, out: ValueId },
    /// `out = src` (device copy; debug taps of a value that is later updated in place).
    Copy { src: TensorRef, out: ValueId },
    /// Reinterpret a contiguous value with another shape (no copy; `out` aliases `input`'s storage).
    View { input: ValueId, out: ValueId },
}

impl GraphOp {
    /// Values this op writes.
    pub fn outputs(&self) -> Vec<ValueId> {
        use GraphOp::*;
        match self {
            Embedding { out } | ZcRmsNorm { out, .. } | Gemm { out, .. } | GemmNvfp4 { out, .. }
            | SiluMul { out, .. } | TreeConv1dSilu { out, .. } | GatedRmsNorm { out, .. } | SegAttn { out, .. }
            | GatherRows { out, .. } | GdnChunkScan { out, .. } | Copy { out, .. } => vec![*out],
            AddZcRmsNorm { out, residual_out, .. } => vec![*out, *residual_out],
            QuantizeNvfp4 { q, s, .. } | ZcRmsNormNvfp4 { q, s, .. } | SiluMulNvfp4 { q, s, .. } => vec![*q, *s],
            GemmNvfp4SwiGlu { q_out, s_out, .. } => vec![*q_out, *s_out],
            AddZcRmsNormNvfp4 { q, s, residual_out, .. } => vec![*q, *s, *residual_out],
            QkNormRope { q, k, .. } => vec![*q, *k],
            GdnPrep { .. } | View { .. } => vec![],
        }
    }

    /// Values this op reads.
    pub fn inputs(&self) -> Vec<ValueId> {
        use GraphOp::*;
        let v = |r: &TensorRef| r.value();
        match self {
            Embedding { .. } => vec![],
            ZcRmsNorm { x, .. } | Gemm { x, .. } | QuantizeNvfp4 { x, .. } | ZcRmsNormNvfp4 { x, .. }
            | TreeConv1dSilu { x, .. } => vec![v(x)],
            AddZcRmsNorm { residual, x, .. } | AddZcRmsNormNvfp4 { residual, x, .. } => vec![*residual, v(x)],
            GemmNvfp4 { q, s, .. } | GemmNvfp4SwiGlu { q, s, .. } => vec![v(q), v(s)],
            SiluMul { gu, .. } | SiluMulNvfp4 { gu, .. } => vec![v(gu)],
            GdnPrep { conv, a, b, .. } => vec![v(conv), v(a), v(b)],
            GdnChunkScan { conv, .. } => vec![v(conv)],
            GatedRmsNorm { o, z, .. } => vec![v(o), v(z)],
            QkNormRope { qg, kv, .. } => vec![v(qg), v(kv)],
            SegAttn { q, k, kv, qg, .. } => vec![v(q), v(k), v(kv), v(qg)],
            GatherRows { src, .. } | Copy { src, .. } => vec![v(src)],
            View { input, .. } => vec![*input],
        }
    }

    /// `(residual, residual_out)` of an op that updates the residual stream in place.
    pub fn in_place(&self) -> Option<(ValueId, ValueId)> {
        match self {
            GraphOp::AddZcRmsNorm { residual, residual_out, .. }
            | GraphOp::AddZcRmsNormNvfp4 { residual, residual_out, .. } => Some((*residual, *residual_out)),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        use GraphOp::*;
        match self {
            Embedding { .. } => "embedding",
            ZcRmsNorm { .. } => "zc_rmsnorm",
            AddZcRmsNorm { .. } => "add_zc_rmsnorm",
            Gemm { .. } => "gemm",
            QuantizeNvfp4 { .. } => "quantize_nvfp4",
            ZcRmsNormNvfp4 { .. } => "zc_rmsnorm_nvfp4",
            AddZcRmsNormNvfp4 { .. } => "add_zc_rmsnorm_nvfp4",
            GemmNvfp4 { .. } => "gemm_nvfp4",
            GemmNvfp4SwiGlu { .. } => "gemm_nvfp4_swiglu",
            SiluMul { .. } => "silu_mul",
            SiluMulNvfp4 { .. } => "silu_mul_nvfp4",
            TreeConv1dSilu { .. } => "tree_conv1d_silu",
            GdnPrep { .. } => "gdn_prep",
            GdnChunkScan { .. } => "gdn_chunk_scan",
            GatedRmsNorm { .. } => "gated_rmsnorm",
            QkNormRope { .. } => "qk_norm_rope",
            SegAttn { .. } => "seg_attn_gqa",
            GatherRows { .. } => "gather_rows",
            Copy { .. } => "copy",
            View { .. } => "view",
        }
    }

    /// Profile class of the op (DESIGN §11 Phase 2a: per-op forward profile): GEMMs by projection, GDN stages,
    /// attention, norms, pointwise, readout.
    pub fn profile_label(&self) -> String {
        use GraphOp::*;
        let proj = |w: &WeightRef| match w {
            WeightRef::Layer(_, s) => format!("{s:?}"),
            other => format!("{other:?}"),
        };
        match self {
            Gemm { w, .. } => format!("gemm.{}", proj(w)),
            GemmNvfp4 { w, .. } => format!("gemm_nvfp4.{}", proj(w)),
            GemmNvfp4SwiGlu { w, .. } => format!("gemm_nvfp4.{}SiluMul", proj(w)),
            Embedding { .. } => "embedding".into(),
            ZcRmsNorm { .. } | AddZcRmsNorm { .. } => "norm.zc_rmsnorm".into(),
            QuantizeNvfp4 { .. } | ZcRmsNormNvfp4 { .. } | AddZcRmsNormNvfp4 { .. } | SiluMulNvfp4 { .. } => {
                format!("nvfp4.{}", self.name())
            }
            SiluMul { .. } => "pointwise.silu_mul".into(),
            TreeConv1dSilu { .. } => "gdn.conv1d_silu".into(),
            GdnPrep { .. } => "gdn.prep".into(),
            GdnChunkScan { .. } => "gdn.scan".into(),
            GatedRmsNorm { .. } => "gdn.gated_rmsnorm".into(),
            QkNormRope { .. } => "attn.qk_norm_rope".into(),
            SegAttn { .. } => "attn.seg_attn".into(),
            GatherRows { .. } => "readout.gather".into(),
            Copy { .. } => "debug.copy".into(),
            View { .. } => "view".into(),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The graph and its pool plan.

/// A forward as SSA values and ops. Built once (shapes symbolic), planned per bucket.
#[derive(Clone, Debug, Default)]
pub struct StepGraph {
    pub specs: Vec<TensorSpec>,
    pub ops: Vec<GraphOp>,
    /// Values read after the graph runs (slot logits, debug taps); they stay live to the end.
    pub outputs: Vec<ValueId>,
}

/// Arena offsets of every value for one bucket.
#[derive(Clone, Debug)]
pub struct PoolPlan {
    pub bucket: Bucket,
    /// Byte offset of each value's storage in the arena.
    pub offsets: Vec<usize>,
    pub arena_bytes: usize,
    /// Sum of all value sizes (what allocating every value separately would cost).
    pub naive_bytes: usize,
}

pub const ARENA_ALIGN: usize = 256;

impl StepGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value(&mut self, dtype: GraphDType, shape: Vec<Dim>) -> ValueId {
        self.specs.push(TensorSpec::new(dtype, shape));
        ValueId(self.specs.len() - 1)
    }

    pub fn push(&mut self, op: GraphOp) {
        self.ops.push(op);
    }

    pub fn spec(&self, v: ValueId) -> &TensorSpec {
        &self.specs[v.0]
    }

    /// Check that every value is defined exactly once before use (a suffix scan's `out` must have been
    /// produced by the matching prefix scan), and that views keep their size.
    pub fn validate(&self) -> Result<()> {
        let n = self.specs.len();
        let mut defined = vec![false; n];
        for (i, op) in self.ops.iter().enumerate() {
            for v in op.inputs() {
                ensure!(v.0 < n, "op {i} ({}) reads unknown value {}", op.name(), v.0);
                ensure!(defined[v.0], "op {i} ({}) reads value {} before it is written", op.name(), v.0);
            }
            if let GraphOp::View { input, out } = op {
                ensure!(!defined[out.0], "op {i}: view target {} already defined", out.0);
                let (a, b) = (self.spec(*input), self.spec(*out));
                ensure!(a.dtype == b.dtype, "op {i}: view changes dtype");
                let big = Bucket { tokens: 1 << 16, answers: 1 << 12, segments: 1 << 10 };
                let small = Bucket { tokens: 3, answers: 2, segments: 5 };
                ensure!(a.bytes(&big) == b.bytes(&big) && a.bytes(&small) == b.bytes(&small),
                        "op {i}: view {:?} -> {:?} changes size", a.shape, b.shape);
                defined[out.0] = true;
            }
            for v in op.outputs() {
                ensure!(v.0 < n, "op {i} ({}) writes unknown value {}", op.name(), v.0);
                ensure!(!defined[v.0], "op {i} ({}) redefines value {}", op.name(), v.0);
                defined[v.0] = true;
            }
        }
        for o in &self.outputs {
            ensure!(defined[o.0], "graph output {} is never written", o.0);
        }
        // An in-place op overwrites `residual`, so nothing may read it (or a view of it) afterwards, and it
        // cannot be a graph output.
        for (i, op) in self.ops.iter().enumerate() {
            let Some((res, res_out)) = op.in_place() else { continue };
            ensure!(self.spec(res) == self.spec(res_out), "op {i}: in-place residual changes shape");
            ensure!(!self.outputs.contains(&res), "op {i}: overwritten residual {} is a graph output", res.0);
            for (j, later) in self.ops.iter().enumerate().skip(i + 1) {
                ensure!(!later.inputs().contains(&res), "op {j} ({}) reads residual {} after op {i} overwrote it",
                        later.name(), res.0);
            }
            for later in &self.ops {
                if let GraphOp::View { input, out } = later {
                    ensure!(*input != res && *out != res, "op {i}: residual {} has a view", res.0);
                }
            }
        }
        Ok(())
    }

    /// Liveness-based arena plan for `bucket` (grout's pool plan, as offsets): a value's storage is taken
    /// when the op writing it runs and returned after its last reader; graph outputs stay live to the end.
    /// Views share their input's storage, and so does the `residual_out` of an in-place op. First fit over a sorted free list with coalescing.
    pub fn plan(&self, bucket: Bucket) -> Result<PoolPlan> {
        self.validate()?;
        let n = self.specs.len();
        // Storage root of each value (views alias their input).
        let mut root: Vec<usize> = (0..n).collect();
        for op in &self.ops {
            if let GraphOp::View { input, out } = op {
                root[out.0] = root[input.0];
            }
            if let Some((res, res_out)) = op.in_place() {
                root[res_out.0] = root[res.0];
            }
        }
        let end = self.ops.len();
        let mut last_use = vec![None::<usize>; n];
        for (i, op) in self.ops.iter().enumerate() {
            for v in op.inputs().into_iter().chain(op.outputs()) {
                last_use[root[v.0]] = Some(i);
            }
        }
        for o in &self.outputs {
            last_use[root[o.0]] = Some(end);
        }
        let size = |v: usize| self.specs[v].bytes(&bucket).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
        let mut offsets = vec![usize::MAX; n];
        let mut free: Vec<(usize, usize)> = Vec::new(); // (offset, len), sorted by offset
        let mut top = 0usize;
        let mut naive = 0usize;
        for (i, op) in self.ops.iter().enumerate() {
            for v in op.outputs() {
                let r = root[v.0];
                if offsets[r] != usize::MAX {
                    continue;
                }
                let sz = size(r).max(ARENA_ALIGN);
                naive += sz;
                offsets[r] = match free.iter().position(|&(_, len)| len >= sz) {
                    Some(k) => {
                        let (off, len) = free[k];
                        if len == sz {
                            free.remove(k);
                        } else {
                            free[k] = (off + sz, len - sz);
                        }
                        off
                    }
                    None => {
                        // Grow the arena, absorbing a free block that ends at the top.
                        match free.last() {
                            Some(&(off, len)) if off + len == top => {
                                free.pop();
                                top = off + sz;
                                off
                            }
                            _ => {
                                top += sz;
                                top - sz
                            }
                        }
                    }
                };
            }
            // Release storages whose last use was this op.
            let mut done: Vec<usize> = op.inputs().into_iter().chain(op.outputs()).map(|v| root[v.0]).collect();
            done.sort_unstable();
            done.dedup();
            for r in done {
                if last_use[r] == Some(i) && offsets[r] != usize::MAX {
                    release(&mut free, offsets[r], size(r).max(ARENA_ALIGN));
                }
            }
        }
        let offsets = (0..n).map(|v| offsets[root[v]]).collect::<Vec<_>>();
        Ok(PoolPlan { bucket, offsets, arena_bytes: top, naive_bytes: naive })
    }
}

fn release(free: &mut Vec<(usize, usize)>, off: usize, len: usize) {
    let k = free.partition_point(|&(o, _)| o < off);
    free.insert(k, (off, len));
    if k + 1 < free.len() && free[k].0 + free[k].1 == free[k + 1].0 {
        free[k].1 += free[k + 1].1;
        free.remove(k + 1);
    }
    if k > 0 && free[k - 1].0 + free[k - 1].1 == free[k].0 {
        free[k - 1].1 += free[k].1;
        free.remove(k);
    }
}

// ---------------------------------------------------------------------------------------------------------
// Kernels (W3) and execution.

/// CUDA-event timeline of one eager forward, for the per-op profile. The executor marks each op with its
/// [`GraphOp::profile_label`]; kernels may add finer marks for their sub-launches ([`Launch::mark`]). The time
/// between a mark and the next one is charged to the first mark's label.
pub struct Profiler {
    ctx: Arc<cuda_core::Context>,
    marks: std::sync::Mutex<Vec<(String, cuda_core::Event)>>,
}

impl Profiler {
    pub fn new(ctx: &Arc<cuda_core::Context>) -> Self {
        Self { ctx: ctx.clone(), marks: Default::default() }
    }

    /// Record a timing event named `label` on `stream`.
    pub fn mark(&self, label: &str, stream: &Stream) -> Result<()> {
        let e = self.ctx.new_event()?;
        e.record(stream)?;
        self.marks.lock().unwrap_or_else(|e| e.into_inner()).push((label.to_string(), e));
        Ok(())
    }

    /// `(label, total ms, launches)` per label in first-seen order, plus the total; waits for the last mark.
    pub fn summary(&self) -> Result<(Vec<(String, f64, usize)>, f64)> {
        let marks = self.marks.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<(String, f64, usize)> = Vec::new();
        let mut total = 0.0;
        for w in marks.windows(2) {
            let ms = w[0].1.elapsed_ms(&w[1].1)? as f64;
            total += ms;
            match out.iter_mut().find(|(l, ..)| *l == w[0].0) {
                Some(e) => {
                    e.1 += ms;
                    e.2 += 1;
                }
                None => out.push((w[0].0.clone(), ms, 1)),
            }
        }
        Ok((out, total))
    }
}

/// What a kernel launch gets besides its operands.
pub struct Launch<'a> {
    /// The cuda-async execution context to launch on (`unsafe { op.execute(cx.exec) }` for cuTile ops).
    pub exec: &'a ExecutionContext,
    pub stream: &'a Arc<Stream>,
    pub bucket: Bucket,
    /// Batch metadata (segments, positions, answer rows) for kernels that are segment-aware.
    pub meta: &'a DeviceMeta,
    pub cfg: Option<&'a ModelConfig>,
    pub weights: &'a dyn WeightStore,
    /// Set for a profiled eager run.
    pub profiler: Option<&'a Profiler>,
}

impl Launch<'_> {
    /// Add a profile mark (no-op unless profiling).
    pub fn mark(&self, label: &str) -> Result<()> {
        match self.profiler {
            Some(p) => p.mark(label, self.stream),
            None => Ok(()),
        }
    }
}

fn missing(name: &str) -> Result<()> {
    Err(anyhow!("kernel {name} is not implemented"))
}

/// The kernels the forward calls (DESIGN §6.4), shaped after the W3 kernels' real interfaces (see
/// [`GraphOp`]). Every method has a failing default, so an implementation can be filled in kernel by kernel.
/// Operands are arena views ([`DevTensor`]); use [`DevTensor::tile`] to pass them to cuTile kernels. All
/// launches go on `cx.exec`'s stream and must not allocate or synchronize (they are recorded into CUDA graphs);
/// batch-dependent state (segment tables, work lists) lives in fixed device buffers the implementation
/// rewrites before each step.
#[allow(unused_variables)]
pub trait Kernels: Sync {
    fn zc_rmsnorm(&self, cx: &Launch, x: &DevTensor, w: &DevTensor, out: &DevTensor) -> Result<()> {
        missing("zc_rmsnorm")
    }
    /// `residual = bf16(residual + x)` in place, then `out = zc_rmsnorm(residual, w)`.
    fn add_zc_rmsnorm(&self, cx: &Launch, residual: &DevTensor, x: &DevTensor, w: &DevTensor, out: &DevTensor)
        -> Result<()> {
        missing("add_zc_rmsnorm")
    }
    fn quantize_nvfp4(&self, cx: &Launch, x: &DevTensor, a_gs: f32, q: &DevTensor, s: &DevTensor) -> Result<()> {
        missing("quantize_nvfp4")
    }
    fn zc_rmsnorm_nvfp4(&self, cx: &Launch, x: &DevTensor, norm: &DevTensor, a_gs: f32, q: &DevTensor,
                        s: &DevTensor) -> Result<()> {
        missing("zc_rmsnorm_nvfp4")
    }
    #[allow(clippy::too_many_arguments)]
    fn add_zc_rmsnorm_nvfp4(&self, cx: &Launch, residual: &DevTensor, x: &DevTensor, norm: &DevTensor, a_gs: f32,
                            q: &DevTensor, s: &DevTensor) -> Result<()> {
        missing("add_zc_rmsnorm_nvfp4")
    }
    fn gemm_nvfp4(&self, cx: &Launch, q: &DevTensor, s: &DevTensor, w: &Linear, out: &DevTensor) -> Result<()> {
        missing("gemm_nvfp4")
    }
    /// The fused gate/up GEMM + SiLU·mul + quantize (`GraphOp::GemmNvfp4SwiGlu`); `a_gs_down` is the down
    /// projection's `input_global_scale`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_nvfp4_swiglu(&self, cx: &Launch, q: &DevTensor, s: &DevTensor, w: &Linear, a_gs_down: f32,
                         q_out: &DevTensor, s_out: &DevTensor) -> Result<()> {
        missing("gemm_nvfp4_swiglu")
    }
    fn silu_mul(&self, cx: &Launch, gu: &DevTensor, out: &DevTensor) -> Result<()> {
        missing("silu_mul")
    }
    fn silu_mul_nvfp4(&self, cx: &Launch, gu: &DevTensor, a_gs: f32, q: &DevTensor, s: &DevTensor) -> Result<()> {
        missing("silu_mul_nvfp4")
    }
    fn tree_conv1d_silu(&self, cx: &Launch, layer: usize, x: &DevTensor, conv_w: &DevTensor, out: &DevTensor)
        -> Result<()> {
        missing("tree_conv1d_silu")
    }
    #[allow(clippy::too_many_arguments)]
    fn gdn_prep(&self, cx: &Launch, layer: usize, conv: &DevTensor, a: &DevTensor, b: &DevTensor, a_log: &DevTensor,
                dt_bias: &DevTensor) -> Result<()> {
        missing("gdn_prep")
    }
    /// `conv`: the whole `[T, 8192]` conv output (q | k | v).
    fn gdn_chunk_scan(&self, cx: &Launch, layer: usize, conv: &DevTensor, out: &DevTensor) -> Result<()> {
        missing("gdn_chunk_scan")
    }
    fn gated_rmsnorm(&self, cx: &Launch, layer: usize, o: &DevTensor, z: &DevTensor, w: &DevTensor, out: &DevTensor)
        -> Result<()> {
        missing("gated_rmsnorm")
    }
    #[allow(clippy::too_many_arguments)]
    fn qk_norm_rope(&self, cx: &Launch, layer: usize, qg: &DevTensor, kv: &DevTensor, q_norm: &DevTensor,
                    k_norm: &DevTensor, q: &DevTensor, k: &DevTensor) -> Result<()> {
        missing("qk_norm_rope")
    }
    #[allow(clippy::too_many_arguments)]
    fn seg_attn(&self, cx: &Launch, layer: usize, q: &DevTensor, k: &DevTensor, kv: &DevTensor, qg: &DevTensor,
                out: &DevTensor) -> Result<()> {
        missing("seg_attn_gqa")
    }
    /// `out[i] = src[answer_rows[i]]` (answer rows from `cx.meta`).
    fn gather_rows(&self, cx: &Launch, src: &DevTensor, out: &DevTensor) -> Result<()> {
        missing("gather_rows")
    }
    /// Profile class of `op` as these kernels run it (default [`GraphOp::profile_label`]); lets an implementation
    /// name the kernels it really launches (e.g. the SIMT GDN gates in place of `gdn.prep`).
    fn profile_label(&self, op: &GraphOp) -> String {
        op.profile_label()
    }
}

/// A [`Kernels`] with nothing implemented (graphs that use only executor-native ops).
pub struct NoKernels;
impl Kernels for NoKernels {}

/// Embedding gather block (elements per CTA along hidden); grout's tuned value.
pub const EMBED_BLOCK: usize = 1024;

/// Runs a [`StepGraph`] over a planned arena.
pub struct Executor<'a> {
    pub graph: &'a StepGraph,
    pub plan: &'a PoolPlan,
    /// The arena (at least `plan.arena_bytes`).
    pub arena: &'a Arc<DevBuf>,
    pub weights: &'a dyn WeightStore,
    pub kernels: &'a dyn Kernels,
    pub blas: &'a Blas,
    pub meta: &'a DeviceMeta,
    pub cfg: Option<&'a ModelConfig>,
    /// Profile marks per op (eager runs only; never set while capturing).
    pub profiler: Option<&'a Profiler>,
}

impl<'a> Executor<'a> {
    /// The arena view of value `v` for this bucket.
    pub fn value(&self, v: ValueId) -> Result<DevTensor> {
        let spec = self.graph.spec(v);
        let off = self.plan.offsets[v.0];
        ensure!(off != usize::MAX, "value {} has no storage", v.0);
        DevTensor::new(self.arena.clone(), off, spec.dtype, spec.concrete(&self.plan.bucket))
    }

    fn resolve(&self, r: TensorRef) -> Result<DevTensor> {
        match r {
            TensorRef::Value(v) => self.value(v),
            TensorRef::Cols { v, start, len } => self.value(v)?.cols_view(start, len),
        }
    }

    /// Enqueue every op on `exec`'s stream.
    pub fn run(&self, exec: &ExecutionContext) -> Result<()> {
        ensure!(self.arena.bytes() >= self.plan.arena_bytes, "arena of {} bytes for a {}-byte plan", self.arena.bytes(),
                self.plan.arena_bytes);
        let cx = Launch {
            exec,
            stream: exec.get_cuda_stream(),
            bucket: self.plan.bucket,
            meta: self.meta,
            cfg: self.cfg,
            weights: self.weights,
            profiler: self.profiler,
        };
        for (i, op) in self.graph.ops.iter().enumerate() {
            if matches!(op, GraphOp::View { .. }) {
                continue;
            }
            cx.mark(&self.kernels.profile_label(op))?;
            self.run_op(&cx, op).with_context(|| format!("op {i} ({})", op.name()))?;
        }
        cx.mark("end")
    }

    fn a_gs(&self, w: WeightRef) -> Result<f32> {
        match self.weights.linear(w)? {
            Linear::Nvfp4 { a_gs, .. } => Ok(*a_gs),
            Linear::Bf16 { .. } => bail!("{w:?} is bf16; NVFP4 quantization needs an NVFP4 projection"),
        }
    }

    fn run_op(&self, cx: &Launch, op: &GraphOp) -> Result<()> {
        use GraphOp::*;
        let k = self.kernels;
        let w = |r: WeightRef| self.weights.tensor(r);
        let lw = |l: usize, s: LayerWeight| self.weights.tensor(WeightRef::Layer(l, s));
        let v = |x: ValueId| self.value(x);
        let r = |x: TensorRef| self.resolve(x);
        match *op {
            Embedding { out } => self.embedding(cx, &v(out)?),
            ZcRmsNorm { x, w: n, out } => k.zc_rmsnorm(cx, &r(x)?, w(n)?, &v(out)?),
            // `residual_out` is `residual`'s storage (planned in place).
            AddZcRmsNorm { x, w: n, out, residual_out, .. } => {
                k.add_zc_rmsnorm(cx, &v(residual_out)?, &r(x)?, w(n)?, &v(out)?)
            }
            Gemm { x, w: wr, out } => {
                let wt = match wr {
                    WeightRef::SlotHead | WeightRef::LmHead | WeightRef::Embed => self.weights.tensor(wr)?,
                    _ => match self.weights.linear(wr)? {
                        Linear::Bf16 { w, .. } => w,
                        Linear::Nvfp4 { .. } => bail!("{wr:?} is NVFP4: use QuantizeNvfp4 + GemmNvfp4"),
                    },
                };
                self.blas.gemm_bf16(cx.stream, &r(x)?, wt, &v(out)?)
            }
            QuantizeNvfp4 { x, w: wr, q, s } => k.quantize_nvfp4(cx, &r(x)?, self.a_gs(wr)?, &v(q)?, &v(s)?),
            ZcRmsNormNvfp4 { x, norm, w: wr, q, s } => {
                k.zc_rmsnorm_nvfp4(cx, &r(x)?, w(norm)?, self.a_gs(wr)?, &v(q)?, &v(s)?)
            }
            AddZcRmsNormNvfp4 { x, norm, w: wr, q, s, residual_out, .. } => {
                k.add_zc_rmsnorm_nvfp4(cx, &v(residual_out)?, &r(x)?, w(norm)?, self.a_gs(wr)?, &v(q)?, &v(s)?)
            }
            GemmNvfp4 { q, s, w: wr, out } => k.gemm_nvfp4(cx, &r(q)?, &r(s)?, self.weights.linear(wr)?, &v(out)?),
            GemmNvfp4SwiGlu { q, s, w: wr, w_down, q_out, s_out } => k.gemm_nvfp4_swiglu(
                cx, &r(q)?, &r(s)?, self.weights.linear(wr)?, self.a_gs(w_down)?, &v(q_out)?, &v(s_out)?),
            SiluMul { gu, out } => k.silu_mul(cx, &r(gu)?, &v(out)?),
            SiluMulNvfp4 { gu, w: wr, q, s } => k.silu_mul_nvfp4(cx, &r(gu)?, self.a_gs(wr)?, &v(q)?, &v(s)?),
            TreeConv1dSilu { layer, x, out } => {
                k.tree_conv1d_silu(cx, layer, &r(x)?, lw(layer, LayerWeight::GdnConv)?, &v(out)?)
            }
            GdnPrep { layer, conv, a, b } => k.gdn_prep(cx, layer, &r(conv)?, &r(a)?, &r(b)?,
                                                       lw(layer, LayerWeight::GdnALog)?,
                                                       lw(layer, LayerWeight::GdnDtBias)?),
            GdnChunkScan { layer, conv, out } => k.gdn_chunk_scan(cx, layer, &r(conv)?, &v(out)?),
            GatedRmsNorm { layer, o, z, out } => {
                k.gated_rmsnorm(cx, layer, &r(o)?, &r(z)?, lw(layer, LayerWeight::GdnNorm)?, &v(out)?)
            }
            QkNormRope { layer, qg, kv, q, k: kk } => k.qk_norm_rope(
                cx, layer, &r(qg)?, &r(kv)?, lw(layer, LayerWeight::QNorm)?, lw(layer, LayerWeight::KNorm)?, &v(q)?,
                &v(kk)?),
            SegAttn { layer, q, k: kk, kv, qg, out } => {
                k.seg_attn(cx, layer, &r(q)?, &r(kk)?, &r(kv)?, &r(qg)?, &v(out)?)
            }
            GatherRows { src, out } => k.gather_rows(cx, &r(src)?, &v(out)?),
            Copy { src, out } => self.copy(cx, &r(src)?, &v(out)?),
            View { .. } => Ok(()),
        }
    }

    /// Device-to-device copy of a contiguous view (row by row for a strided source).
    fn copy(&self, cx: &Launch, src: &DevTensor, out: &DevTensor) -> Result<()> {
        ensure!(src.dtype == out.dtype && src.shape == out.shape && out.is_contiguous(), "copy {src:?} -> {out:?}");
        let es = src.dtype.size_in_bytes();
        let s = cx.stream.cu_stream();
        if src.is_contiguous() {
            unsafe { cuda_core::memory::memcpy_dtod_async(out.ptr(), src.ptr(), src.bytes(), s)? };
        } else {
            let row = src.cols() * es;
            for i in 0..src.rows() {
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(out.ptr() + (i * row) as u64,
                                                         src.ptr() + (i * src.row_stride * es) as u64, row, s)?
                };
            }
        }
        Ok(())
    }

    /// `out[t] = embed[token_ids[t]]` with cutile-kernels' `embedding_batch_f16` (bf16 bits moved as f16:
    /// a pure gather, so bit-exact).
    fn embedding(&self, cx: &Launch, out: &DevTensor) -> Result<()> {
        use cutile::tensor::PartitionMut;
        use cutile::tile_kernel::TileKernel;
        let table = self.weights.tensor(WeightRef::Embed)?;
        ensure!(out.dtype == GraphDType::BF16 && out.shape.len() == 2 && out.shape[1] == table.shape[1],
                "embedding out {out:?} for table {table:?}");
        let d = table.shape[1];
        let block = if d % EMBED_BLOCK == 0 { EMBED_BLOCK } else { d };
        let t = out.shape[0];
        let ids = self.meta.token_ids.rows_view(0, t)?;
        // SAFETY: arena, weight and metadata views are live for the executor's lifetime; within one graph the
        // ops are stream-ordered, so nothing else touches `out` while this kernel writes it.
        unsafe {
            let ids_t = ids.tile::<u32>()?;
            let table_t = table.tile::<half::f16>()?;
            let mut out_t = out.tile::<half::f16>()?;
            let op = cutile_kernels::embedding_batch_f16(&ids_t, &table_t, (&mut out_t).partition([1, block]))
                .generics(vec![d.to_string(), block.to_string()]);
            op.execute(cx.exec).map_err(|e| anyhow!("embedding launch: {e:?}"))?;
        }
        Ok(())
    }
}

/// Wraps host code that enqueues work on the context's stream as a graph-recordable [`DeviceOp`] (grout's
/// `KernelGraphOp`). The closure must only launch kernels and copies between pre-allocated buffers.
pub struct StreamOp<F>(pub F);

impl<F: FnOnce(&ExecutionContext) -> Result<(), DeviceError> + Send> DeviceOp for StreamOp<F> {
    type Output = ();
    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        (self.0)(ctx)
    }
}

impl<F: FnOnce(&ExecutionContext) -> Result<(), DeviceError> + Send> GraphNode for StreamOp<F> {}

impl<F: FnOnce(&ExecutionContext) -> Result<(), DeviceError> + Send> std::future::IntoFuture for StreamOp<F> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        DeviceFuture::failed(DeviceError::Internal("StreamOp runs only with sync_on or graph capture".into()))
    }
}

fn dev_err(e: anyhow::Error) -> DeviceError {
    DeviceError::Anyhow(format!("{e:#}"))
}

impl Executor<'_> {
    /// Run the graph now on `stream` and wait for it.
    pub fn run_eager(&self, stream: &Arc<Stream>) -> Result<()> {
        StreamOp(|cx: &ExecutionContext| self.run(cx).map_err(dev_err))
            .sync_on(stream)
            .map_err(|e| anyhow!("{e}"))
    }

    /// Capture the graph as a CUDA graph on `stream` (run [`run_eager`](Self::run_eager) once first so kernel
    /// JIT and cuBLAS heuristics happen outside the capture).
    pub fn capture(&self, stream: &Arc<Stream>) -> Result<CudaGraph<()>> {
        CudaGraph::scope(stream, |s| s.record(StreamOp(|cx: &ExecutionContext| self.run(cx).map_err(dev_err))))
            .map_err(|e| anyhow!("CUDA graph capture: {e}"))
    }
}

/// One planned arena and one captured CUDA graph per bucket.
pub struct BucketGraphs {
    pub plan: BucketPlan,
    pub pools: HashMap<Bucket, PoolPlan>,
    pub arena: Arc<DevBuf>,
    graphs: HashMap<Bucket, CudaGraph<()>>,
}

impl BucketGraphs {
    /// Plan every bucket and allocate one arena for the largest plan (one step runs at a time).
    pub fn new(graph: &StepGraph, plan: BucketPlan, stream: &Arc<Stream>) -> Result<Self> {
        let mut pools = HashMap::new();
        let mut need = 0;
        for b in &plan.buckets {
            let p = graph.plan(*b)?;
            need = need.max(p.arena_bytes);
            pools.insert(*b, p);
        }
        let arena = DevBuf::zeroed(stream, need)?;
        Ok(Self { plan, pools, arena, graphs: HashMap::new() })
    }

    pub fn pool(&self, b: Bucket) -> Result<&PoolPlan> {
        self.pools.get(&b).with_context(|| format!("no plan for {b:?}"))
    }

    /// Store the captured graph of `b` (from [`Executor::capture`] over this bucket's pool and the arena).
    pub fn insert(&mut self, b: Bucket, graph: CudaGraph<()>) -> Result<()> {
        ensure!(self.pools.contains_key(&b), "no plan for {b:?}");
        self.graphs.insert(b, graph);
        Ok(())
    }

    /// Forget every captured graph (the next step of each bucket runs eagerly and captures again).
    pub fn clear_captured(&mut self) {
        self.graphs.clear();
    }

    pub fn is_captured(&self, b: Bucket) -> bool {
        self.graphs.contains_key(&b)
    }

    /// Replay the captured graph of `b` on `stream` (ordered after earlier work on that stream, e.g. the
    /// metadata upload). Does not wait.
    pub fn launch(&self, b: Bucket, stream: &Arc<Stream>) -> Result<()> {
        let g = self.graphs.get(&b).with_context(|| format!("bucket {b:?} not captured"))?;
        // SAFETY: every buffer the graph touches (arena, weights, metadata) is owned by the engine and outlives
        // the replay; the caller orders host access after a stream synchronize.
        unsafe { g.launch().async_on(stream) }.map_err(|e| anyhow!("graph launch: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain() -> StepGraph {
        // x0 = embed; x1 = gemm(x0); x2 = gemm(x1); x3 = gemm(x2 cols); x3 is the output.
        let mut g = StepGraph::new();
        let x0 = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(64)]);
        let x1 = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(128)]);
        let x2 = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(64)]);
        let x3 = g.value(GraphDType::F32, vec![Dim::Tokens, Dim::Fixed(16)]);
        let wr = WeightRef::Layer(0, LayerWeight::MlpGateUp);
        g.push(GraphOp::Embedding { out: x0 });
        g.push(GraphOp::Gemm { x: x0.into(), w: wr, out: x1 });
        g.push(GraphOp::Gemm { x: TensorRef::Cols { v: x1, start: 0, len: 64 }, w: wr, out: x2 });
        g.push(GraphOp::Gemm { x: x2.into(), w: wr, out: x3 });
        g.outputs.push(x3);
        g
    }

    #[test]
    fn plan_reuses_dead_storage() {
        let g = chain();
        let b = Bucket { tokens: 100, answers: 4, segments: 8 };
        let p = g.plan(b).unwrap();
        // x0 (12.8 KB) is dead after op 1, so x2 (same size) takes its place.
        assert_eq!(p.offsets[2], p.offsets[0]);
        assert!(p.arena_bytes < p.naive_bytes, "{} vs {}", p.arena_bytes, p.naive_bytes);
        // Live ranges never overlap in memory: check every pair of values live at the same op.
        let sz = |v: usize| g.specs[v].bytes(&b);
        for (i, op) in g.ops.iter().enumerate() {
            let live: Vec<usize> = op.inputs().into_iter().chain(op.outputs()).map(|v| v.0).collect();
            for &a in &live {
                for &c in &live {
                    if a != c {
                        let (oa, oc) = (p.offsets[a], p.offsets[c]);
                        assert!(oa + sz(a) <= oc || oc + sz(c) <= oa, "op {i}: values {a} and {c} overlap");
                    }
                }
            }
        }
    }

    #[test]
    fn validate_catches_use_before_def_and_bad_views() {
        let mut g = chain();
        g.ops.swap(1, 2);
        assert!(g.validate().is_err());
        let mut g = StepGraph::new();
        let a = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(8)]);
        let b = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(4)]);
        g.push(GraphOp::Embedding { out: a });
        g.push(GraphOp::View { input: a, out: b });
        assert!(g.validate().is_err());
    }

    #[test]
    fn views_alias_their_input() {
        let mut g = StepGraph::new();
        let a = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(8)]);
        let b = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(2), Dim::Fixed(4)]);
        let c = g.value(GraphDType::BF16, vec![Dim::Tokens, Dim::Fixed(8)]);
        g.push(GraphOp::Embedding { out: a });
        g.push(GraphOp::View { input: a, out: b });
        g.push(GraphOp::Copy { src: b.into(), out: c });
        g.outputs.push(c);
        let p = g.plan(Bucket { tokens: 10, answers: 1, segments: 2 }).unwrap();
        assert_eq!(p.offsets[a.0], p.offsets[b.0]);
        assert_ne!(p.offsets[c.0], p.offsets[a.0]);
    }

    #[test]
    fn in_place_residual_shares_storage_and_is_not_read_after() {
        let w = WeightRef::Layer(0, LayerWeight::InputNorm);
        let mk = |read_old_after: bool| {
            let mut g = StepGraph::new();
            let sh = vec![Dim::Tokens, Dim::Fixed(8)];
            let x = g.value(GraphDType::BF16, sh.clone());
            let d = g.value(GraphDType::BF16, sh.clone());
            let h = g.value(GraphDType::BF16, sh.clone());
            let x2 = g.value(GraphDType::BF16, sh.clone());
            let o = g.value(GraphDType::BF16, sh.clone());
            g.push(GraphOp::Embedding { out: x });
            g.push(GraphOp::Copy { src: x.into(), out: d });
            g.push(GraphOp::AddZcRmsNorm { residual: x, x: d.into(), w, out: h, residual_out: x2 });
            g.push(GraphOp::Copy { src: if read_old_after { x } else { x2 }.into(), out: o });
            g.outputs.extend([h, o]);
            (g, x, x2)
        };
        let (g, x, x2) = mk(false);
        let p = g.plan(Bucket { tokens: 10, answers: 1, segments: 2 }).unwrap();
        assert_eq!(p.offsets[x.0], p.offsets[x2.0]);
        assert!(mk(true).0.validate().is_err());
    }

    #[test]
    fn bucket_pick() {
        let p = BucketPlan::new(&[1024, 256, 512], 64, 1 << 20).unwrap();
        assert_eq!(p.pick(256, 3, 4).unwrap().tokens, 256);
        assert_eq!(p.pick(257, 3, 4).unwrap().tokens, 512);
        // 100 answers do not fit 256's cap of 64 — but no bucket allows more than 64.
        assert!(p.pick(200, 100, 4).is_none());
        // A batch exactly filling a bucket needs no pad segment.
        assert_eq!(p.pick(256, 1, 257).unwrap().tokens, 256);
        assert_eq!(p.pick(255, 1, 257).unwrap().tokens, 512);
    }
}
