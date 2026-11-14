//! Device-resident weights of a System One checkpoint (DESIGN §3, §3.1, §6.3, §6.5).
//!
//! The weight format is a property of each projection ([`Linear`]), so bf16, NVFP4-MLP and any future
//! mixed checkpoint load through the same code. Projections that run as one GEMM are fused at load by
//! writing each source tensor into its row range of one destination buffer:
//! - GDN `[in_proj_qkv; in_proj_z; in_proj_a; in_proj_b]` → `[12352, 4096]`,
//! - attention `[k_proj; v_proj]` → `[2048, 4096]`,
//! - MLP `[gate_proj; up_proj]` → `[24576, 4096]`.
//!
//! For NVFP4 the packed e2m1 bytes and the e4m3 group scales are row-concatenated the same way; each source
//! keeps its own `weight_global_scale` ([`LinearPart::w_gs`]), and all parts must share one
//! `input_global_scale` (the activation is quantized once for the fused GEMM); the loader refuses the
//! checkpoint otherwise.
//!
//! The LM head is reduced to the slot rows (`lm_head[slot_ids]`, ids from the frontend); the full head is
//! loaded only on request (debug `allowed_token_mass`).

use crate::engine::config::{LayerType, ModelConfig};
use crate::engine::graph::tensor::{DevTensor, GraphDType};
use crate::engine::loader::{Checkpoint, StreamOptions, StreamStats, Streamer};
use anyhow::{Context as _, Result, bail, ensure};
use cuda_core::Stream;
use std::sync::Arc;

/// One source projection inside a (possibly fused) [`Linear`].
#[derive(Clone, Debug)]
pub struct LinearPart {
    /// Checkpoint prefix, e.g. `model.language_model.layers.0.mlp.gate_proj`.
    pub name: String,
    /// First output row of this part in the fused weight.
    pub row0: usize,
    pub rows: usize,
    /// NVFP4 only: this part's `weight_global_scale` (dequantized weight = e2m1 · scale / w_gs).
    pub w_gs: Option<f32>,
}

/// A projection `y = x · Wᵀ` with `W [out_features, in_features]`.
#[derive(Clone, Debug)]
pub enum Linear {
    Bf16 {
        /// `[N, K]` bf16, row-major.
        w: DevTensor,
        parts: Vec<LinearPart>,
    },
    /// compressed-tensors `nvfp4-pack-quantized` (W4A4, group 16 along K).
    Nvfp4 {
        /// `[N, K/2]` u8: two e2m1 values per byte, as stored in `weight_packed`.
        packed: DevTensor,
        /// `[N, K/16]` f8e4m3 group scales (`weight_scale`).
        scales: DevTensor,
        /// Parts with their own `w_gs`; the GEMM epilogue multiplies part `i`'s output columns by
        /// `1 / (a_gs · w_gs_i)`.
        parts: Vec<LinearPart>,
        /// The shared `input_global_scale` of every part.
        a_gs: f32,
        in_features: usize,
        /// `Some(b)`: a gate/up projection whose rows were interleaved in blocks of `b` for the fused SiLU·mul
        /// GEMM ([`ModelWeights::interleave_gate_up_nvfp4`]): output row block `j` of `2b` rows is gate rows
        /// `j·b..` then up rows `j·b..`. Only the fused gate/up GEMM may read such a weight. `None`: checkpoint
        /// row order (`parts` in order).
        gate_up_block: Option<usize>,
    },
}

impl Linear {
    pub fn out_features(&self) -> usize {
        match self {
            Self::Bf16 { w, .. } => w.shape[0],
            Self::Nvfp4 { packed, .. } => packed.shape[0],
        }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Bf16 { w, .. } => w.shape[1],
            Self::Nvfp4 { in_features, .. } => *in_features,
        }
    }

    pub fn parts(&self) -> &[LinearPart] {
        match self {
            Self::Bf16 { parts, .. } | Self::Nvfp4 { parts, .. } => parts,
        }
    }

    pub fn is_nvfp4(&self) -> bool {
        matches!(self, Self::Nvfp4 { .. })
    }

    /// Device bytes of the weight.
    pub fn bytes(&self) -> usize {
        match self {
            Self::Bf16 { w, .. } => w.bytes(),
            Self::Nvfp4 { packed, scales, .. } => packed.bytes() + scales.bytes(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mlp {
    /// `[gate; up]`: `[2 · intermediate, hidden]`.
    pub gate_up: Linear,
    /// `[hidden, intermediate]`.
    pub down: Linear,
}

/// A GatedDeltaNet (linear attention) layer.
#[derive(Clone, Debug)]
pub struct GdnLayer {
    pub input_norm: DevTensor,
    pub post_norm: DevTensor,
    /// `[in_proj_qkv; in_proj_z; in_proj_a; in_proj_b]`: `[conv_dim + v_dim + 2·v_heads, hidden]`.
    pub in_proj: Linear,
    /// Depthwise conv taps `[conv_dim, kernel]` (from `[conv_dim, 1, kernel]`).
    pub conv_w: DevTensor,
    /// `[v_heads]` bf16.
    pub a_log: DevTensor,
    /// `[v_heads]` bf16.
    pub dt_bias: DevTensor,
    /// Gated RMSNorm weight `[v_head_dim]` (plain `w`, not `1 + w`).
    pub norm_w: DevTensor,
    pub out_proj: Linear,
    pub mlp: Mlp,
}

/// A full (block-bidirectional) attention layer.
#[derive(Clone, Debug)]
pub struct AttnLayer {
    pub input_norm: DevTensor,
    pub post_norm: DevTensor,
    /// `[heads · 2 · head_dim, hidden]`: per head, query then gate.
    pub q_proj: Linear,
    /// `[k_proj; v_proj]`: `[2 · kv_heads · head_dim, hidden]`.
    pub kv_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: DevTensor,
    pub k_norm: DevTensor,
    pub mlp: Mlp,
}

#[derive(Clone, Debug)]
pub enum Layer {
    Gdn(GdnLayer),
    Attn(AttnLayer),
}

impl Layer {
    pub fn mlp(&self) -> &Mlp {
        match self {
            Self::Gdn(l) => &l.mlp,
            Self::Attn(l) => &l.mlp,
        }
    }

    pub fn input_norm(&self) -> &DevTensor {
        match self {
            Self::Gdn(l) => &l.input_norm,
            Self::Attn(l) => &l.input_norm,
        }
    }

    pub fn post_norm(&self) -> &DevTensor {
        match self {
            Self::Gdn(l) => &l.post_norm,
            Self::Attn(l) => &l.post_norm,
        }
    }
}

/// `lm_head` rows of the slot tokens, `[n_slots, hidden]` bf16, in slot order.
#[derive(Clone, Debug)]
pub struct SlotHead {
    pub token_ids: Vec<u32>,
    pub w: DevTensor,
}

pub struct ModelWeights {
    /// `[vocab, hidden]` bf16.
    pub embed: DevTensor,
    pub final_norm: DevTensor,
    pub layers: Vec<Layer>,
    pub slot_head: Option<SlotHead>,
    /// Full `[vocab, hidden]` head, only when requested.
    pub lm_head: Option<DevTensor>,
}

impl ModelWeights {
    /// Device bytes held.
    pub fn bytes(&self) -> usize {
        let mut b = self.embed.bytes() + self.final_norm.bytes();
        b += self.slot_head.as_ref().map_or(0, |s| s.w.bytes()) + self.lm_head.as_ref().map_or(0, |t| t.bytes());
        for l in &self.layers {
            b += l.input_norm().bytes() + l.post_norm().bytes() + l.mlp().gate_up.bytes() + l.mlp().down.bytes();
            b += match l {
                Layer::Gdn(g) => {
                    g.in_proj.bytes() + g.conv_w.bytes() + g.a_log.bytes() + g.dt_bias.bytes() + g.norm_w.bytes()
                        + g.out_proj.bytes()
                }
                Layer::Attn(a) => {
                    a.q_proj.bytes() + a.kv_proj.bytes() + a.o_proj.bytes() + a.q_norm.bytes() + a.k_norm.bytes()
                }
            };
        }
        b
    }
}

#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// Slot token ids (from the frontend, `n_slots` of them); the slot head is skipped when `None`.
    pub slot_token_ids: Option<Vec<u32>>,
    /// Also load the full `lm_head` (debug mass).
    pub full_lm_head: bool,
    pub stream: StreamOptions,
}

#[derive(Clone, Debug)]
pub struct LoadReport {
    pub stream: StreamStats,
    pub device_bytes: usize,
    pub nvfp4_projections: usize,
}

const LM: &str = "model.language_model";

/// Queues uploads while building the weight structs; the copies run in one pass at the end.
struct Builder<'a> {
    ckpt: &'a Checkpoint,
    stream: &'a Arc<Stream>,
    up: Streamer<'a>,
    nvfp4: usize,
}

impl<'a> Builder<'a> {
    /// A whole tensor, with an expected shape and dtype.
    fn tensor(&mut self, name: &str, dtype: GraphDType, shape: &[usize]) -> Result<DevTensor> {
        let t = self.ckpt.get(name)?;
        ensure!(t.dtype == dtype, "{name}: dtype {:?}, expected {dtype:?}", t.dtype);
        ensure!(t.shape == shape, "{name}: shape {:?}, expected {shape:?}", t.shape);
        let d = DevTensor::alloc(self.stream.context(), dtype, shape)?;
        self.up.tensor(name, &d)?;
        Ok(d)
    }

    /// A (possibly fused) projection from checkpoint prefixes, each `[rows_i, k]`.
    fn linear(&mut self, prefixes: &[String], rows: &[usize], k: usize) -> Result<Linear> {
        let n: usize = rows.iter().sum();
        let nvfp4: Vec<bool> = prefixes.iter().map(|p| self.ckpt.contains(&format!("{p}.weight_packed"))).collect();
        ensure!(nvfp4.iter().all(|&q| q == nvfp4[0]), "fused projections {prefixes:?} mix weight formats");
        let mut parts = Vec::new();
        let mut row0 = 0;
        if !nvfp4[0] {
            let w = DevTensor::alloc(self.stream.context(), GraphDType::BF16, &[n, k])?;
            for (p, &r) in prefixes.iter().zip(rows) {
                let name = format!("{p}.weight");
                let t = self.ckpt.get(&name)?;
                ensure!(t.dtype == GraphDType::BF16 && t.shape == [r, k], "{name}: {:?} {:?}, expected bf16 [{r}, {k}]",
                        t.dtype, t.shape);
                self.up.tensor(&name, &w.rows_view(row0, r)?)?;
                parts.push(LinearPart { name: p.clone(), row0, rows: r, w_gs: None });
                row0 += r;
            }
            return Ok(Linear::Bf16 { w, parts });
        }
        ensure!(k % 16 == 0, "NVFP4 K={k} not a multiple of 16");
        let packed = DevTensor::alloc(self.stream.context(), GraphDType::U8, &[n, k / 2])?;
        let scales = DevTensor::alloc(self.stream.context(), GraphDType::F8E4M3, &[n, k / 16])?;
        let mut a_gs: Option<f32> = None;
        for (p, &r) in prefixes.iter().zip(rows) {
            let pk = self.ckpt.get(&format!("{p}.weight_packed"))?;
            ensure!(pk.dtype == GraphDType::U8 && pk.shape == [r, k / 2], "{p}.weight_packed: {:?} {:?}", pk.dtype, pk.shape);
            let sc = self.ckpt.get(&format!("{p}.weight_scale"))?;
            ensure!(sc.dtype == GraphDType::F8E4M3 && sc.shape == [r, k / 16], "{p}.weight_scale: {:?} {:?}", sc.dtype,
                    sc.shape);
            let w_gs = self.ckpt.read_f32_scalar(&format!("{p}.weight_global_scale"))?;
            let gs = self.ckpt.read_f32_scalar(&format!("{p}.input_global_scale"))?;
            ensure!(w_gs.is_finite() && w_gs > 0.0 && gs.is_finite() && gs > 0.0, "{p}: global scales {w_gs} {gs}");
            match a_gs {
                None => a_gs = Some(gs),
                // Bit equality: the activation is quantized once for every part of the fused GEMM.
                Some(a) if a.to_bits() == gs.to_bits() => {}
                Some(a) => bail!("{p}.input_global_scale {gs} differs from {a} of the projection it is fused with; \
                                  the fused NVFP4 GEMM needs one activation scale"),
            }
            self.up.tensor(&format!("{p}.weight_packed"), &packed.rows_view(row0, r)?)?;
            self.up.tensor(&format!("{p}.weight_scale"), &scales.rows_view(row0, r)?)?;
            parts.push(LinearPart { name: p.clone(), row0, rows: r, w_gs: Some(w_gs) });
            row0 += r;
            self.nvfp4 += 1;
        }
        Ok(Linear::Nvfp4 { packed, scales, parts, a_gs: a_gs.unwrap(), in_features: k, gate_up_block: None })
    }
}

impl ModelWeights {
    /// Load the text model of `ckpt` onto `stream`'s device.
    pub fn load(cfg: &ModelConfig, ckpt: &Checkpoint, stream: &Arc<Stream>, opts: &LoadOptions)
        -> Result<(Self, LoadReport)> {
        let t = &cfg.text_config;
        let h = t.hidden_size;
        let bf = GraphDType::BF16;
        let mut b = Builder { ckpt, stream, up: Streamer::new(ckpt), nvfp4: 0 };

        let embed = b.tensor(&format!("{LM}.embed_tokens.weight"), bf, &[t.vocab_size, h])?;
        let final_norm = b.tensor(&format!("{LM}.norm.weight"), bf, &[h])?;
        let conv_dim = cfg.linear_conv_dim();
        let v_dim = t.linear_num_value_heads * t.linear_value_head_dim;
        let vh = t.linear_num_value_heads;
        let mut layers = Vec::with_capacity(t.num_hidden_layers);
        for (i, ty) in t.layer_types.iter().enumerate() {
            let p = format!("{LM}.layers.{i}");
            let input_norm = b.tensor(&format!("{p}.input_layernorm.weight"), bf, &[h])?;
            let post_norm = b.tensor(&format!("{p}.post_attention_layernorm.weight"), bf, &[h])?;
            let mlp = Mlp {
                gate_up: b.linear(&[format!("{p}.mlp.gate_proj"), format!("{p}.mlp.up_proj")],
                                  &[t.intermediate_size, t.intermediate_size], h)?,
                down: b.linear(&[format!("{p}.mlp.down_proj")], &[h], t.intermediate_size)?,
            };
            layers.push(match ty {
                LayerType::LinearAttention => {
                    let la = format!("{p}.linear_attn");
                    let in_proj = b.linear(
                        &[format!("{la}.in_proj_qkv"), format!("{la}.in_proj_z"), format!("{la}.in_proj_a"),
                          format!("{la}.in_proj_b")],
                        &[conv_dim, v_dim, vh, vh],
                        h,
                    )?;
                    let k = t.linear_conv_kernel_dim;
                    let conv_w = b.tensor(&format!("{la}.conv1d.weight"), bf, &[conv_dim, 1, k])?.reshape(&[conv_dim, k])?;
                    Layer::Gdn(GdnLayer {
                        input_norm,
                        post_norm,
                        in_proj,
                        conv_w,
                        a_log: b.tensor(&format!("{la}.A_log"), bf, &[vh])?,
                        dt_bias: b.tensor(&format!("{la}.dt_bias"), bf, &[vh])?,
                        norm_w: b.tensor(&format!("{la}.norm.weight"), bf, &[t.linear_value_head_dim])?,
                        out_proj: b.linear(&[format!("{la}.out_proj")], &[h], v_dim)?,
                        mlp,
                    })
                }
                LayerType::FullAttention => {
                    let sa = format!("{p}.self_attn");
                    let hd = t.head_dim;
                    let kv = t.num_key_value_heads * hd;
                    let q_rows = t.num_attention_heads * hd * if t.attn_output_gate { 2 } else { 1 };
                    Layer::Attn(AttnLayer {
                        input_norm,
                        post_norm,
                        q_proj: b.linear(&[format!("{sa}.q_proj")], &[q_rows], h)?,
                        kv_proj: b.linear(&[format!("{sa}.k_proj"), format!("{sa}.v_proj")], &[kv, kv], h)?,
                        o_proj: b.linear(&[format!("{sa}.o_proj")], &[h], t.num_attention_heads * hd)?,
                        q_norm: b.tensor(&format!("{sa}.q_norm.weight"), bf, &[hd])?,
                        k_norm: b.tensor(&format!("{sa}.k_norm.weight"), bf, &[hd])?,
                        mlp,
                    })
                }
            });
        }

        let head = "lm_head.weight";
        let info = ckpt.get(head)?;
        ensure!(info.dtype == bf && info.shape == [t.vocab_size, h], "{head}: {:?} {:?}", info.dtype, info.shape);
        let slot_head = match &opts.slot_token_ids {
            Some(ids) => {
                ensure!(!ids.is_empty(), "empty slot token id list");
                ensure!(ids.len() <= cfg.modeling_overrides.n_slots, "{} slot ids for n_slots {}", ids.len(),
                        cfg.modeling_overrides.n_slots);
                let w = DevTensor::alloc(stream.context(), bf, &[ids.len(), h])?;
                b.up.rows(head, ids, &w).context("gathering slot rows of lm_head")?;
                Some(SlotHead { token_ids: ids.clone(), w })
            }
            None => None,
        };
        let lm_head = if opts.full_lm_head { Some(b.tensor(head, bf, &[t.vocab_size, h])?) } else { None };

        let nvfp4 = b.nvfp4;
        let stats = b.up.run(stream, &opts.stream)?;
        let w = Self { embed, final_norm, layers, slot_head, lm_head };
        let report = LoadReport { stream: stats, device_bytes: w.bytes(), nvfp4_projections: nvfp4 };
        Ok((w, report))
    }

    /// F1: interleave the rows of every NVFP4 MLP gate/up projection (packed values and block scales alike) in
    /// blocks of `block` rows (`crate::kernels::nvfp4::interleave_gate_up`), so one `[2·block, K]` weight tile
    /// of the fused gate/up + SiLU·mul GEMM holds the gate and up rows of the same `block` product columns. A
    /// permutation of rows: no value changes. Marks each projection with `gate_up_block = Some(block)`.
    pub fn interleave_gate_up_nvfp4(&mut self, stream: &Stream, block: usize) -> Result<usize> {
        let mut n = 0;
        for l in &mut self.layers {
            let mlp = match l {
                Layer::Gdn(g) => &mut g.mlp,
                Layer::Attn(a) => &mut a.mlp,
            };
            let Linear::Nvfp4 { packed, scales, parts, gate_up_block, .. } = &mut mlp.gate_up else { continue };
            ensure!(gate_up_block.is_none(), "gate/up already interleaved");
            ensure!(parts.len() == 2 && parts[0].rows == parts[1].rows, "gate/up parts {parts:?}");
            let rows = packed.shape[0];
            for t in [&*packed, &*scales] {
                ensure!(t.is_contiguous() && t.shape.len() == 2 && t.shape[0] == rows, "interleave {t:?}");
                let b = t.to_host_bytes(stream)?;
                let o = crate::kernels::nvfp4::interleave_gate_up(&b, rows, b.len() / rows, block)?;
                t.copy_from_host(stream, &o)?;
            }
            *gate_up_block = Some(block);
            n += 1;
        }
        Ok(n)
    }

    /// Debug (numerics diagnostics only): re-express the model in a permuted hidden basis, `new[.., j] =
    /// old[.., perm[j]]` on the residual stream. The model computes the same function (RMSNorm is permutation
    /// invariant; every projection reading or writing the residual stream is permuted to match), but every
    /// reduction over the hidden dimension sums in another order: an independent bf16 realization of the same
    /// numerics, as `reference/hidden_dense.py` builds for the reference. Hidden states then come out permuted.
    pub fn permute_hidden(&self, stream: &Stream, perm: &[usize]) -> Result<()> {
        let h = self.final_norm.numel();
        ensure!(perm.len() == h, "permutation of {} for hidden {h}", perm.len());
        let cols = |t: &DevTensor| -> Result<()> {
            ensure!(t.shape.len() == 2 && t.shape[1] == h && t.is_contiguous(), "permute cols of {t:?}");
            let b = t.to_host_bytes(stream)?;
            let mut o = vec![0u8; b.len()];
            for r in 0..t.shape[0] {
                let (src, dst) = (&b[r * h * 2..(r + 1) * h * 2], &mut o[r * h * 2..(r + 1) * h * 2]);
                for (j, &p) in perm.iter().enumerate() {
                    dst[2 * j..2 * j + 2].copy_from_slice(&src[2 * p..2 * p + 2]);
                }
            }
            t.copy_from_host(stream, &o)
        };
        let rows = |t: &DevTensor| -> Result<()> {
            ensure!(t.shape[0] == h && t.is_contiguous(), "permute rows of {t:?}");
            let b = t.to_host_bytes(stream)?;
            let row = b.len() / h;
            let mut o = vec![0u8; b.len()];
            for (j, &p) in perm.iter().enumerate() {
                o[j * row..(j + 1) * row].copy_from_slice(&b[p * row..(p + 1) * row]);
            }
            t.copy_from_host(stream, &o)
        };
        let bf = |l: &Linear| -> Result<DevTensor> {
            match l {
                Linear::Bf16 { w, .. } => Ok(w.clone()),
                Linear::Nvfp4 { .. } => bail!("permute_hidden supports bf16 projections only"),
            }
        };
        cols(&self.embed)?;
        rows(&self.final_norm)?;
        if let Some(s) = &self.slot_head {
            cols(&s.w)?;
        }
        if let Some(w) = &self.lm_head {
            cols(w)?;
        }
        for l in &self.layers {
            rows(l.input_norm())?;
            rows(l.post_norm())?;
            cols(&bf(&l.mlp().gate_up)?)?;
            rows(&bf(&l.mlp().down)?)?;
            match l {
                Layer::Gdn(g) => {
                    cols(&bf(&g.in_proj)?)?;
                    rows(&bf(&g.out_proj)?)?;
                }
                Layer::Attn(a) => {
                    cols(&bf(&a.q_proj)?)?;
                    cols(&bf(&a.kv_proj)?)?;
                    rows(&bf(&a.o_proj)?)?;
                }
            }
        }
        Ok(())
    }
}

/// Debug: a seeded permutation of `0..n` (Fisher-Yates over a 64-bit LCG), for [`ModelWeights::permute_hidden`].
pub fn debug_permutation(n: usize, seed: u64) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for i in (1..n).rev() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = ((s >> 33) as usize) % (i + 1);
        p.swap(i, j);
    }
    p
}
