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
//!
//! NVFP4 MLP projections (`Linear::Nvfp4`, DESIGN §3.1, §6.6; N1) replace the MLP part per projection. Fused
//! (the default, [`build`]), the activation quantization runs in the epilogue of the op before each GEMM:
//! ```text
//!   q1, s1 = add_zc_rmsnorm_nvfp4(x += o, ln2)      residual add + norm + quantize (gate/up's input_global_scale)
//!   gu     = gemm_nvfp4(q1, s1, [Wgate; Wup])        block-scaled FP4 GEMM, per-half epilogue 1/(a_gs·w_gs)
//!   q2, s2 = silu_mul_nvfp4(gu)                      SiLU·mul + quantize (down's input_global_scale)
//!   m      = gemm_nvfp4(q2, s2, Wdown)
//! ```
//! Unfused ([`build_with`] with `nvfp4_fused = false`) the bf16 norm and SiLU·mul outputs are materialised and a
//! standalone `quantize_nvfp4` runs before each GEMM; both forms are bit-identical (tests/kernels_nvfp4_fused.rs).
//! Every other projection stays bf16 (cuBLAS); an NVFP4 projection outside the MLP is refused.

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
    /// Also keep, for every layer in `layers`, the mixer output (`mix{i}`, before the residual add), the residual
    /// stream after the mixer add (`mid{i}`) and the MLP output (`mlp{i}`, before its residual add).
    pub sublayers: bool,
}

/// A built forward and the values a caller reads back.
pub struct Forward {
    pub graph: StepGraph,
    /// `[Q, n_slots]` f32 slot logits.
    pub logits: ValueId,
    /// `(name, value)` per tap: `layer{i}` and `norm`, each `[T, hidden]` bf16.
    pub taps: Vec<(String, ValueId)>,
    pub n_slots: usize,
    /// The last layer was trimmed to the answer rows ([`ForwardOptions::trim_last_layer`]).
    pub trimmed: bool,
}

/// How [`build_opts`] writes the forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardOptions {
    /// NVFP4 checkpoints: quantize activations in the epilogue of the preceding norm / SiLU·mul.
    pub nvfp4_fused: bool,
    /// L1: compute the last layer's queries, attention, `o_proj`, post-norm and MLP, and the final norm, only for
    /// the answer rows ([`trimmed_last_layer`]); keys and values stay computed for every token. Applies only when
    /// the last layer is full attention and no debug tap needs the last layer's rows (`Taps::norm` or the last
    /// layer in `Taps::layers`); otherwise the forward is built untrimmed ([`Forward::trimmed`] says which).
    pub trim_last_layer: bool,
}

impl Default for ForwardOptions {
    fn default() -> Self {
        Self { nvfp4_fused: true, trim_last_layer: true }
    }
}

/// Whether the last layer of `cfg` can be trimmed to the answer rows with these taps.
pub fn can_trim_last_layer(cfg: &ModelConfig, taps: &Taps) -> bool {
    let t = &cfg.text_config;
    let last = t.num_hidden_layers - 1;
    t.layer_types.get(last) == Some(&LayerType::FullAttention) && !taps.norm && !taps.layers.contains(&last)
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
/// readout rows; NVFP4 quantization fused into the preceding norm / SiLU·mul, last layer trimmed when possible
/// ([`ForwardOptions::default`]).
pub fn build(cfg: &ModelConfig, weights: &dyn WeightStore, n_slots: usize, taps: &Taps) -> Result<Forward> {
    build_opts(cfg, weights, n_slots, taps, ForwardOptions::default())
}

/// The untrimmed forward, choosing whether NVFP4 activation quantization is fused into the preceding op
/// (`nvfp4_fused`) or runs as a standalone `quantize_nvfp4` on the materialised bf16 activation.
pub fn build_with(cfg: &ModelConfig, weights: &dyn WeightStore, n_slots: usize, taps: &Taps, nvfp4_fused: bool)
    -> Result<Forward> {
    build_opts(cfg, weights, n_slots, taps, ForwardOptions { nvfp4_fused, trim_last_layer: false })
}

/// The forward as `opts` says.
pub fn build_opts(cfg: &ModelConfig, weights: &dyn WeightStore, n_slots: usize, taps: &Taps, opts: ForwardOptions)
    -> Result<Forward> {
    let nvfp4_fused = opts.nvfp4_fused;
    let trim = opts.trim_last_layer && can_trim_last_layer(cfg, taps);
    check_dims(cfg)?;
    let t = &cfg.text_config;
    let h = t.hidden_size;
    let inter = t.intermediate_size;
    let conv_dim = cfg.linear_conv_dim();
    let v_dim = t.linear_num_value_heads * t.linear_value_head_dim;
    let vh = t.linear_num_value_heads;
    let q_dim = t.num_attention_heads * t.head_dim;
    let kv_dim = t.num_key_value_heads * t.head_dim;
    for (i, ty) in t.layer_types.iter().enumerate() {
        let mixer: &[LayerWeight] = match ty {
            LayerType::LinearAttention => &[LayerWeight::GdnInProj, LayerWeight::GdnOutProj],
            LayerType::FullAttention => &[LayerWeight::AttnQ, LayerWeight::AttnKv, LayerWeight::AttnO],
        };
        for &s in mixer {
            if weights.linear(WeightRef::Layer(i, s))?.is_nvfp4() {
                bail!("layer {i} {s:?} is NVFP4: only the MLP projections can run on NVFP4");
            }
        }
    }
    let u8t = GraphDType::U8;
    let f8 = GraphDType::F8E4M3;
    // NVFP4 activation of a [T, k] bf16 input: packed e2m1 pairs and e4m3 group scales.
    let act = |g: &mut StepGraph, k: usize| {
        (g.value(u8t, vec![Dim::Tokens, Dim::Fixed(k / 2)]), g.value(f8, vec![Dim::Tokens, Dim::Fixed(k / 16)]))
    };

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
        if trim && i == t.num_hidden_layers - 1 {
            let dims = TrimDims { h, inter, q_dim, kv_dim };
            let logits = trimmed_last_layer(&mut g, weights, i, dims, x, hn, n_slots, nvfp4_fused)?;
            g.validate()?;
            return Ok(Forward { graph: g, logits, taps: out_taps, n_slots, trimmed: true });
        }
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
        let (wgu, wdn) = (lw(LayerWeight::MlpGateUp), lw(LayerWeight::MlpDown));
        let (gu_q, dn_q) = (weights.linear(wgu)?.is_nvfp4(), weights.linear(wdn)?.is_nvfp4());
        let x2 = tok(&mut g, h);
        let gu = tok(&mut g, 2 * inter);
        if gu_q && nvfp4_fused {
            let (q, s) = act(&mut g, h);
            g.push(GraphOp::AddZcRmsNormNvfp4 {
                residual: x,
                x: mixer.into(),
                norm: lw(LayerWeight::PostNorm),
                w: wgu,
                q,
                s,
                residual_out: x2,
            });
            g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wgu, out: gu });
        } else {
            let h2 = tok(&mut g, h);
            g.push(GraphOp::AddZcRmsNorm { residual: x, x: mixer.into(), w: lw(LayerWeight::PostNorm), out: h2,
                                           residual_out: x2 });
            if gu_q {
                let (q, s) = act(&mut g, h);
                g.push(GraphOp::QuantizeNvfp4 { x: h2.into(), w: wgu, q, s });
                g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wgu, out: gu });
            } else {
                g.push(GraphOp::Gemm { x: h2.into(), w: wgu, out: gu });
            }
        }
        x = x2;
        let sub = taps.sublayers && taps.layers.contains(&i);
        if sub {
            tap(&mut g, format!("mix{i}"), mixer);
            tap(&mut g, format!("mid{i}"), x);
        }
        let mo = tok(&mut g, h);
        if dn_q && nvfp4_fused {
            let (q, s) = act(&mut g, inter);
            g.push(GraphOp::SiluMulNvfp4 { gu: gu.into(), w: wdn, q, s });
            g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wdn, out: mo });
        } else {
            let m = tok(&mut g, inter);
            g.push(GraphOp::SiluMul { gu: gu.into(), out: m });
            if dn_q {
                let (q, s) = act(&mut g, inter);
                g.push(GraphOp::QuantizeNvfp4 { x: m.into(), w: wdn, q, s });
                g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wdn, out: mo });
            } else {
                g.push(GraphOp::Gemm { x: m.into(), w: wdn, out: mo });
            }
        }
        if sub {
            tap(&mut g, format!("mlp{i}"), mo);
        }
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
    Ok(Forward { graph: g, logits, taps: out_taps, n_slots, trimmed: false })
}

#[derive(Clone, Copy)]
struct TrimDims {
    h: usize,
    inter: usize,
    q_dim: usize,
    kv_dim: usize,
}

/// L1: the last layer (full attention, index `i`) and the readout over the answer rows only. `x` is the residual
/// stream after the layer's input add (every token), `hn` its input norm. Every token's keys and values are still
/// computed (the answer rows attend to them); everything else (queries, attention output, `o_proj`, the residual
/// add, post-norm, MLP and final norm) is computed on the `[Q, ..]` gathered answer rows. Each op is row-local and
/// the same op as in the untrimmed forward, so the answer rows get the same values as there, provided the GEMMs
/// do not depend on the row count (bf16: [`crate::engine::gemm::GemmMode::Pinned`]; the NVFP4 GEMM's tiles are
/// fixed at every T).
///
/// ```text
/// kv    = hn · [Wk; Wv]ᵀ                 [T, 2048]   k = k_norm_rope(kv)        [T, 1024]
/// hn_a  = hn[answer_rows]                [Q, 4096]   qg_a = hn_a · Wqᵀ          [Q, 8192]
/// q_a   = qk_norm_rope(qg_a, kv[answer_rows]) at the answer positions
/// o_a   = seg_attn(q_a | k, kv | gate qg_a) · Woᵀ      one work item per answer row
/// x_a   = x[answer_rows]; h2_a = add_zc_rmsnorm(x_a += o_a, ln2); m_a = MLP(h2_a)   (bf16 or NVFP4, as per layer)
/// hs    = add_zc_rmsnorm(x_a += m_a, final_norm); logits = hs · W_slotᵀ
/// ```
///
/// The MLP ops repeat the per-layer MLP of [`build_opts`] on `Dim::Answers` values (kept separate from the loop so
/// the per-layer code is untouched).
#[allow(clippy::too_many_arguments)]
fn trimmed_last_layer(g: &mut StepGraph, weights: &dyn WeightStore, i: usize, d: TrimDims, x: ValueId, hn: ValueId,
                      n_slots: usize, nvfp4_fused: bool) -> Result<ValueId> {
    let bf = GraphDType::BF16;
    let TrimDims { h, inter, q_dim, kv_dim } = d;
    let tok = |g: &mut StepGraph, n: usize| g.value(bf, vec![Dim::Tokens, Dim::Fixed(n)]);
    let ans = |g: &mut StepGraph, n: usize| g.value(bf, vec![Dim::Answers, Dim::Fixed(n)]);
    let act = |g: &mut StepGraph, k: usize| {
        (g.value(GraphDType::U8, vec![Dim::Answers, Dim::Fixed(k / 2)]),
         g.value(GraphDType::F8E4M3, vec![Dim::Answers, Dim::Fixed(k / 16)]))
    };
    let lw = |s| WeightRef::Layer(i, s);

    // Keys and values for every token.
    let kv = tok(g, 2 * kv_dim);
    g.push(GraphOp::Gemm { x: hn.into(), w: lw(LayerWeight::AttnKv), out: kv });
    let k = tok(g, kv_dim);
    g.push(GraphOp::KNormRope { layer: i, kv: kv.into(), k });
    // Queries, attention and o_proj for the answer rows.
    let hn_a = ans(g, h);
    g.push(GraphOp::GatherRows { src: hn.into(), out: hn_a });
    let qg_a = ans(g, 2 * q_dim);
    g.push(GraphOp::Gemm { x: hn_a.into(), w: lw(LayerWeight::AttnQ), out: qg_a });
    let kv_a = ans(g, 2 * kv_dim);
    g.push(GraphOp::GatherRows { src: kv.into(), out: kv_a });
    let (q_a, k_a) = (ans(g, q_dim), ans(g, kv_dim));
    g.push(GraphOp::QkNormRopeAnswers { layer: i, qg: qg_a.into(), kv: kv_a.into(), q: q_a, k: k_a });
    let ao_a = ans(g, q_dim);
    g.push(GraphOp::SegAttnAnswers { layer: i, q: q_a.into(), k: k.into(), kv: kv.into(), qg: qg_a.into(), out: ao_a });
    let mixer = ans(g, h);
    g.push(GraphOp::Gemm { x: ao_a.into(), w: lw(LayerWeight::AttnO), out: mixer });
    let x_a = ans(g, h);
    g.push(GraphOp::GatherRows { src: x.into(), out: x_a });

    // The MLP on the answer rows: the per-layer MLP of build_opts on Answers-row values.
    let (wgu, wdn) = (lw(LayerWeight::MlpGateUp), lw(LayerWeight::MlpDown));
    let (gu_q, dn_q) = (weights.linear(wgu)?.is_nvfp4(), weights.linear(wdn)?.is_nvfp4());
    let x2 = ans(g, h);
    let gu = ans(g, 2 * inter);
    if gu_q && nvfp4_fused {
        let (q, s) = act(g, h);
        g.push(GraphOp::AddZcRmsNormNvfp4 { residual: x_a, x: mixer.into(), norm: lw(LayerWeight::PostNorm), w: wgu, q,
                                            s, residual_out: x2 });
        g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wgu, out: gu });
    } else {
        let h2 = ans(g, h);
        g.push(GraphOp::AddZcRmsNorm { residual: x_a, x: mixer.into(), w: lw(LayerWeight::PostNorm), out: h2,
                                       residual_out: x2 });
        if gu_q {
            let (q, s) = act(g, h);
            g.push(GraphOp::QuantizeNvfp4 { x: h2.into(), w: wgu, q, s });
            g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wgu, out: gu });
        } else {
            g.push(GraphOp::Gemm { x: h2.into(), w: wgu, out: gu });
        }
    }
    let mo = ans(g, h);
    if dn_q && nvfp4_fused {
        let (q, s) = act(g, inter);
        g.push(GraphOp::SiluMulNvfp4 { gu: gu.into(), w: wdn, q, s });
        g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wdn, out: mo });
    } else {
        let m = ans(g, inter);
        g.push(GraphOp::SiluMul { gu: gu.into(), out: m });
        if dn_q {
            let (q, s) = act(g, inter);
            g.push(GraphOp::QuantizeNvfp4 { x: m.into(), w: wdn, q, s });
            g.push(GraphOp::GemmNvfp4 { q: q.into(), s: s.into(), w: wdn, out: mo });
        } else {
            g.push(GraphOp::Gemm { x: m.into(), w: wdn, out: mo });
        }
    }

    // Final norm and readout on the answer rows.
    let (hs, x_final) = (ans(g, h), ans(g, h));
    g.push(GraphOp::AddZcRmsNorm { residual: x2, x: mo.into(), w: WeightRef::FinalNorm, out: hs, residual_out: x_final });
    let logits = g.value(GraphDType::F32, vec![Dim::Answers, Dim::Fixed(n_slots)]);
    g.push(GraphOp::Gemm { x: hs.into(), w: WeightRef::SlotHead, out: logits });
    g.outputs.push(logits);
    Ok(logits)
}
