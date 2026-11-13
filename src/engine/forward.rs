//! The 32-layer packed tree forward as a [`StepGraph`] (DESIGN §6.3), `[linear ×3, full] ×8` for elephant-1.
//!
//! ```text
//! x  = embed[ids]
//! per layer i:
//!   h      = zc_rmsnorm(x, ln1)                      (layer 0; later layers fuse the residual add, in place)
//!   GDN:   proj = h · [Wqkv; Wz; Wa; Wb]ᵀ             one [T, 12352] GEMM
//!          conv = tree_conv1d_silu(proj[:, :8192]); gdn_prep(conv, a, b); core = gdn_chunk_scan(conv)
//!          (cuTile: gdn_prep stores f32 q, k, the scan reads conv[:, 4096:]; SIMT: gates only, the scan reads all of conv)
//!          o = gated_rmsnorm(core, z) · Woutᵀ
//!   full:  qg = h · Wqᵀ; kv = h · [Wk; Wv]ᵀ; q, k = qk_norm_rope(qg, kv); o = seg_attn(q, k, kv, qg) · Woᵀ
//!   h2     = add_zc_rmsnorm(x += o, ln2)
//!   m      = silu_mul(h2 · [Wgate; Wup]ᵀ) · Wdownᵀ     one [T, 24576] GEMM, then down
//!   (next layer: x += m inside its input norm)
//! hn = add_zc_rmsnorm(x += m, final_norm)            over all T rows
//! logits [Q, n_slots] f32 = gather(hn, answer_rows) · W_slotᵀ
//! ```
//! The NVFP4 MLP path (`Linear::Nvfp4`, §6.6) is not built yet: [`build`] refuses NVFP4 projections.

use crate::engine::config::{LayerType, ModelConfig};
use crate::engine::graph::tensor::GraphDType;
use crate::engine::graph::{Dim, GraphOp, LayerWeight, StepGraph, TensorRef, ValueId, WeightRef, WeightStore};
use crate::kernels::{attention as attn, gdn};
use anyhow::{Result, bail, ensure};

/// Debug outputs of a forward: copies of the residual stream after chosen layers, and the final-norm output.
#[derive(Clone, Debug, Default)]
pub struct Taps {
    /// Decoder layers whose output (the residual stream after the layer) is kept, `[T, hidden]` bf16.
    pub layers: Vec<usize>,
    /// Keep the final norm's output over all rows.
    pub norm: bool,
}

/// A built forward and the values a caller reads back.
pub struct Forward {
    pub graph: StepGraph,
    /// `[Q, n_slots]` f32 slot logits.
    pub logits: ValueId,
    /// `(name, value)` per tap: `layer{i}` and `norm`, each `[T, hidden]` bf16.
    pub taps: Vec<(String, ValueId)>,
    pub n_slots: usize,
}

fn cols(v: ValueId, start: usize, len: usize) -> TensorRef {
    TensorRef::Cols { v, start, len }
}

/// Check that the model matches the dimensions the W3 kernels are compiled for.
pub fn check_dims(cfg: &ModelConfig) -> Result<()> {
    let t = &cfg.text_config;
    ensure!(t.hidden_size % 1024 == 0, "hidden {} is not a multiple of 1024", t.hidden_size);
    ensure!(t.intermediate_size % 1024 == 0, "intermediate {} is not a multiple of 1024", t.intermediate_size);
    ensure!(cfg.linear_conv_dim() == gdn::CONV_DIM && t.linear_conv_kernel_dim == gdn::CONV_KERNEL,
            "GDN conv {}x{} differs from the kernels' {}x{}", cfg.linear_conv_dim(), t.linear_conv_kernel_dim,
            gdn::CONV_DIM, gdn::CONV_KERNEL);
    ensure!(t.linear_num_value_heads == gdn::NUM_V_HEADS && t.linear_num_key_heads == gdn::NUM_K_HEADS
            && t.linear_key_head_dim == gdn::HEAD_DIM && t.linear_value_head_dim == gdn::HEAD_DIM,
            "GDN heads differ from the kernels'");
    ensure!(t.num_attention_heads == attn::NUM_Q_HEADS && t.num_key_value_heads == attn::NUM_KV_HEADS
            && t.head_dim == attn::HEAD_DIM && cfg.rotary_dim() == attn::ROTARY_DIM,
            "attention heads differ from the kernels'");
    ensure!(t.rope_parameters.rope_theta == attn::ROPE_THETA, "rope theta {} differs from the kernels' {}",
            t.rope_parameters.rope_theta, attn::ROPE_THETA);
    Ok(())
}

/// Build the forward for `cfg` over `weights` (which decides bf16 vs NVFP4 per projection) with `n_slots`
/// readout rows.
pub fn build(cfg: &ModelConfig, weights: &dyn WeightStore, n_slots: usize, taps: &Taps) -> Result<Forward> {
    check_dims(cfg)?;
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let inter = t.intermediate_size;
    let conv_dim = cfg.linear_conv_dim();
    let v_dim = t.linear_num_value_heads * t.linear_value_head_dim;
    let vh = t.linear_num_value_heads;
    let q_dim = t.num_attention_heads * t.head_dim;
    let kv_dim = t.num_key_value_heads * t.head_dim;
    for i in 0..t.num_hidden_layers {
        for s in [LayerWeight::MlpGateUp, LayerWeight::MlpDown] {
            if weights.linear(WeightRef::Layer(i, s))?.is_nvfp4() {
                bail!("layer {i} {s:?} is NVFP4: the NVFP4 forward is not implemented yet (Phase 2b)");
            }
        }
    }

    let mut g = StepGraph::new();
    let bf = GraphDType::BF16;
    let tok = |g: &mut StepGraph, n: usize| g.value(bf, vec![Dim::Tokens, Dim::Fixed(n)]);
    let mut out_taps = Vec::new();
    let mut tap = |g: &mut StepGraph, name: String, src: ValueId| {
        let v = g.value(bf, vec![Dim::Tokens, Dim::Fixed(h)]);
        g.push(GraphOp::Copy { src: src.into(), out: v });
        g.outputs.push(v);
        out_taps.push((name, v));
    };

    let mut x = tok(&mut g, h);
    g.push(GraphOp::Embedding { out: x });
    let mut pending: Option<ValueId> = None; // the previous layer's MLP output, not yet added to x
    for (i, ty) in t.layer_types.iter().enumerate() {
        let ln1 = WeightRef::Layer(i, LayerWeight::InputNorm);
        let hn = tok(&mut g, h);
        match pending.take() {
            None => g.push(GraphOp::ZcRmsNorm { x: x.into(), w: ln1, out: hn }),
            Some(m) => {
                let x2 = tok(&mut g, h);
                g.push(GraphOp::AddZcRmsNorm { residual: x, x: m.into(), w: ln1, out: hn, residual_out: x2 });
                x = x2;
                if taps.layers.contains(&(i - 1)) {
                    tap(&mut g, format!("layer{}", i - 1), x);
                }
            }
        }
        let lw = |s| WeightRef::Layer(i, s);
        let mixer = tok(&mut g, h);
        match ty {
            LayerType::LinearAttention => {
                let n_proj = conv_dim + v_dim + 2 * vh;
                let proj = tok(&mut g, n_proj);
                g.push(GraphOp::Gemm { x: hn.into(), w: lw(LayerWeight::GdnInProj), out: proj });
                let conv = tok(&mut g, conv_dim);
                g.push(GraphOp::TreeConv1dSilu { layer: i, x: cols(proj, 0, conv_dim), out: conv });
                g.push(GraphOp::GdnPrep {
                    layer: i,
                    conv: conv.into(),
                    a: cols(proj, conv_dim + v_dim, vh),
                    b: cols(proj, conv_dim + v_dim + vh, vh),
                });
                let core = tok(&mut g, v_dim);
                g.push(GraphOp::GdnChunkScan { layer: i, conv: conv.into(), out: core });
                let gated = tok(&mut g, v_dim);
                g.push(GraphOp::GatedRmsNorm { layer: i, o: core.into(), z: cols(proj, conv_dim, v_dim), out: gated });
                g.push(GraphOp::Gemm { x: gated.into(), w: lw(LayerWeight::GdnOutProj), out: mixer });
            }
            LayerType::FullAttention => {
                let qg = tok(&mut g, 2 * q_dim);
                g.push(GraphOp::Gemm { x: hn.into(), w: lw(LayerWeight::AttnQ), out: qg });
                let kv = tok(&mut g, 2 * kv_dim);
                g.push(GraphOp::Gemm { x: hn.into(), w: lw(LayerWeight::AttnKv), out: kv });
                let q = tok(&mut g, q_dim);
                let k = tok(&mut g, kv_dim);
                g.push(GraphOp::QkNormRope { layer: i, qg: qg.into(), kv: kv.into(), q, k });
                let ao = tok(&mut g, q_dim);
                g.push(GraphOp::SegAttn { layer: i, q: q.into(), k: k.into(), kv: kv.into(), qg: qg.into(), out: ao });
                g.push(GraphOp::Gemm { x: ao.into(), w: lw(LayerWeight::AttnO), out: mixer });
            }
        }
        let h2 = tok(&mut g, h);
        let x2 = tok(&mut g, h);
        g.push(GraphOp::AddZcRmsNorm {
            residual: x,
            x: mixer.into(),
            w: lw(LayerWeight::PostNorm),
            out: h2,
            residual_out: x2,
        });
        x = x2;
        let gu = tok(&mut g, 2 * inter);
        g.push(GraphOp::Gemm { x: h2.into(), w: lw(LayerWeight::MlpGateUp), out: gu });
        let m = tok(&mut g, inter);
        g.push(GraphOp::SiluMul { gu: gu.into(), out: m });
        let mo = tok(&mut g, h);
        g.push(GraphOp::Gemm { x: m.into(), w: lw(LayerWeight::MlpDown), out: mo });
        pending = Some(mo);
    }
    let last = t.num_hidden_layers - 1;
    let normed = tok(&mut g, h);
    let x_final = tok(&mut g, h);
    let m = pending.expect("at least one layer");
    g.push(GraphOp::AddZcRmsNorm { residual: x, x: m.into(), w: WeightRef::FinalNorm, out: normed, residual_out: x_final });
    if taps.layers.contains(&last) {
        tap(&mut g, format!("layer{last}"), x_final);
    }
    if taps.norm {
        tap(&mut g, "norm".into(), normed);
    }
    let hs = g.value(bf, vec![Dim::Answers, Dim::Fixed(h)]);
    g.push(GraphOp::GatherRows { src: normed.into(), out: hs });
    let logits = g.value(GraphDType::F32, vec![Dim::Answers, Dim::Fixed(n_slots)]);
    g.push(GraphOp::Gemm { x: hs.into(), w: WeightRef::SlotHead, out: logits });
    g.outputs.push(logits);
    g.validate()?;
    Ok(Forward { graph: g, logits, taps: out_taps, n_slots })
}
