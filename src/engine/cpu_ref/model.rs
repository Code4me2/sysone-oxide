//! The elephant-1 text model on the CPU: weights, and the forward over a token [`Layout`].
//!
//! Written from transformers 5.19 `modeling_qwen3_5.py` (`Qwen3_5TextModel`, `Qwen3_5DecoderLayer`,
//! `Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, `Qwen3_5MLP`) and DESIGN.md §3, as the fp32 reference computes it
//! (every bf16 cast of the model's own math is a no-op in fp32). The GatedDeltaNet uses the *recurrent* delta rule
//! (`torch_recurrent_gated_delta_rule`), which the reference's chunked form equals up to rounding.
//!
//! The forward is generic over a [`Layout`]: positions, the conv taps' source tokens, the attention key set of
//! every query, and the delta-rule runs. A [`DecisionBatch`](crate::contract::DecisionBatch) tree and a single
//! flat sequence with an explicit block mask are two layouts of the same math.

use super::ops::{self, matmul};
use super::weights::{Mat, SafeTensors};
use crate::engine::config::{LayerType, ModelConfig};
use anyhow::{Result, ensure};
use rayon::prelude::*;
use std::path::Path;

/// Shapes of the text model (from `config.json`'s `text_config`).
#[derive(Clone, Debug)]
pub struct Dims {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub eps: f64,
    pub layer_types: Vec<LayerType>,
    // full attention
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    // GatedDeltaNet
    pub conv_kernel: usize,
    pub k_heads: usize,
    pub k_dim: usize,
    pub v_heads: usize,
    pub v_dim: usize,
}

impl Dims {
    pub fn from_config(cfg: &ModelConfig) -> Self {
        let t = &cfg.text_config;
        Dims {
            hidden: t.hidden_size,
            intermediate: t.intermediate_size,
            vocab: t.vocab_size,
            eps: t.rms_norm_eps,
            layer_types: t.layer_types.clone(),
            heads: t.num_attention_heads,
            kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            rotary_dim: cfg.rotary_dim(),
            rope_theta: t.rope_parameters.rope_theta,
            conv_kernel: t.linear_conv_kernel_dim,
            k_heads: t.linear_num_key_heads,
            k_dim: t.linear_key_head_dim,
            v_heads: t.linear_num_value_heads,
            v_dim: t.linear_value_head_dim,
        }
    }

    pub fn key_dim(&self) -> usize {
        self.k_heads * self.k_dim
    }

    pub fn value_dim(&self) -> usize {
        self.v_heads * self.v_dim
    }

    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

pub struct GdnWeights {
    pub in_proj_qkv: Mat,
    pub in_proj_z: Mat,
    pub in_proj_b: Mat,
    pub in_proj_a: Mat,
    /// `[conv_dim, conv_kernel]` depthwise taps.
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// Gated RMSNorm weight, `[v_dim]`.
    pub norm: Vec<f32>,
    pub out_proj: Mat,
}

pub struct AttnWeights {
    pub q_proj: Mat,
    pub k_proj: Mat,
    pub v_proj: Mat,
    pub o_proj: Mat,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

pub enum Mixer {
    Gdn(GdnWeights),
    Attn(AttnWeights),
}

pub struct Layer {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub mixer: Mixer,
    pub gate_proj: Mat,
    pub up_proj: Mat,
    pub down_proj: Mat,
}

pub struct Model {
    pub dims: Dims,
    pub embed: Mat,
    pub layers: Vec<Layer>,
    pub norm: Vec<f32>,
    /// `lm_head` (kept as a view; the readout reads only slot rows).
    pub lm_head: Mat,
}

impl Model {
    /// Load the text model from a bf16 checkpoint directory (`config.json` + safetensors).
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg = ModelConfig::load(dir)?;
        let d = Dims::from_config(&cfg);
        let st = SafeTensors::open_dir(dir)?;
        let p = "model.language_model";
        let mut layers = Vec::with_capacity(d.layer_types.len());
        for (i, ty) in d.layer_types.iter().enumerate() {
            let l = format!("{p}.layers.{i}");
            let mixer = match ty {
                LayerType::LinearAttention => {
                    let m = format!("{l}.linear_attn");
                    Mixer::Gdn(GdnWeights {
                        in_proj_qkv: st.mat(&format!("{m}.in_proj_qkv.weight"))?,
                        in_proj_z: st.mat(&format!("{m}.in_proj_z.weight"))?,
                        in_proj_b: st.mat(&format!("{m}.in_proj_b.weight"))?,
                        in_proj_a: st.mat(&format!("{m}.in_proj_a.weight"))?,
                        conv: st.vec(&format!("{m}.conv1d.weight"), d.conv_dim() * d.conv_kernel)?,
                        a_log: st.vec(&format!("{m}.A_log"), d.v_heads)?,
                        dt_bias: st.vec(&format!("{m}.dt_bias"), d.v_heads)?,
                        norm: st.vec(&format!("{m}.norm.weight"), d.v_dim)?,
                        out_proj: st.mat(&format!("{m}.out_proj.weight"))?,
                    })
                }
                LayerType::FullAttention => {
                    let m = format!("{l}.self_attn");
                    Mixer::Attn(AttnWeights {
                        q_proj: st.mat(&format!("{m}.q_proj.weight"))?,
                        k_proj: st.mat(&format!("{m}.k_proj.weight"))?,
                        v_proj: st.mat(&format!("{m}.v_proj.weight"))?,
                        o_proj: st.mat(&format!("{m}.o_proj.weight"))?,
                        q_norm: st.vec(&format!("{m}.q_norm.weight"), d.head_dim)?,
                        k_norm: st.vec(&format!("{m}.k_norm.weight"), d.head_dim)?,
                    })
                }
            };
            layers.push(Layer {
                input_layernorm: st.vec(&format!("{l}.input_layernorm.weight"), d.hidden)?,
                post_attention_layernorm: st.vec(&format!("{l}.post_attention_layernorm.weight"), d.hidden)?,
                mixer,
                gate_proj: st.mat(&format!("{l}.mlp.gate_proj.weight"))?,
                up_proj: st.mat(&format!("{l}.mlp.up_proj.weight"))?,
                down_proj: st.mat(&format!("{l}.mlp.down_proj.weight"))?,
            });
        }
        let model = Model {
            embed: st.mat(&format!("{p}.embed_tokens.weight"))?,
            norm: st.vec(&format!("{p}.norm.weight"), d.hidden)?,
            lm_head: st.mat("lm_head.weight")?,
            layers,
            dims: d,
        };
        model.check()?;
        Ok(model)
    }

    /// Every matrix has the shape the config implies.
    pub fn check(&self) -> Result<()> {
        let d = &self.dims;
        let shape = |m: &Mat, r: usize, c: usize, what: &str| -> Result<()> {
            ensure!(m.rows == r && m.cols == c, "{what} is [{}, {}], expected [{r}, {c}]", m.rows, m.cols);
            Ok(())
        };
        shape(&self.embed, d.vocab, d.hidden, "embed_tokens")?;
        shape(&self.lm_head, d.vocab, d.hidden, "lm_head")?;
        for (i, l) in self.layers.iter().enumerate() {
            shape(&l.gate_proj, d.intermediate, d.hidden, &format!("layer {i} gate_proj"))?;
            shape(&l.up_proj, d.intermediate, d.hidden, &format!("layer {i} up_proj"))?;
            shape(&l.down_proj, d.hidden, d.intermediate, &format!("layer {i} down_proj"))?;
            match &l.mixer {
                Mixer::Gdn(g) => {
                    shape(&g.in_proj_qkv, d.conv_dim(), d.hidden, &format!("layer {i} in_proj_qkv"))?;
                    shape(&g.in_proj_z, d.value_dim(), d.hidden, &format!("layer {i} in_proj_z"))?;
                    shape(&g.in_proj_b, d.v_heads, d.hidden, &format!("layer {i} in_proj_b"))?;
                    shape(&g.in_proj_a, d.v_heads, d.hidden, &format!("layer {i} in_proj_a"))?;
                    shape(&g.out_proj, d.hidden, d.value_dim(), &format!("layer {i} out_proj"))?;
                }
                Mixer::Attn(a) => {
                    shape(&a.q_proj, 2 * d.heads * d.head_dim, d.hidden, &format!("layer {i} q_proj"))?;
                    shape(&a.k_proj, d.kv_heads * d.head_dim, d.hidden, &format!("layer {i} k_proj"))?;
                    shape(&a.v_proj, d.kv_heads * d.head_dim, d.hidden, &format!("layer {i} v_proj"))?;
                    shape(&a.o_proj, d.hidden, d.heads * d.head_dim, &format!("layer {i} o_proj"))?;
                }
            }
        }
        Ok(())
    }
}

/// One delta-rule run: a contiguous token range scanned in order, from a zero state (`init == None`) or from the
/// final state of an earlier run. Runs with `init == None` go first; runs that read a state never write one.
#[derive(Clone, Debug)]
pub struct Run {
    pub start: usize,
    pub len: usize,
    pub init: Option<usize>,
}

/// How the tokens of one forward relate to each other.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    /// RoPE position per token.
    pub pos: Vec<u32>,
    /// `conv_prev[t][d-1]`: the token at logical distance `d` before `t` for the causal conv1d, `None` for the
    /// zero padding before a sequence start.
    pub conv_prev: Vec<Vec<Option<u32>>>,
    /// Keys/values visible to each query token in the full-attention layers.
    pub keys: Vec<Vec<u32>>,
    pub runs: Vec<Run>,
}

/// The final-norm hidden states of `rows` after a forward of `tokens` under `layout`, `[rows.len(), hidden]`.
pub fn forward_hidden(model: &Model, tokens: &[u32], layout: &Layout, rows: &[u32]) -> Vec<f32> {
    let d = &model.dims;
    let t = tokens.len();
    let h = d.hidden;
    let mut x = vec![0.0f32; t * h];
    x.par_chunks_mut(h).zip(tokens.par_iter()).for_each(|(row, &id)| model.embed.row_into(id as usize, row));

    for layer in &model.layers {
        let hn = ops::zc_rmsnorm(&x, &layer.input_layernorm, d.eps);
        let mixed = match &layer.mixer {
            Mixer::Gdn(g) => gdn(d, g, &hn, layout),
            Mixer::Attn(a) => attention(d, a, &hn, layout),
        };
        add_into(&mut x, &mixed);
        let hn = ops::zc_rmsnorm(&x, &layer.post_attention_layernorm, d.eps);
        let mlp_out = mlp(layer, &hn, t);
        add_into(&mut x, &mlp_out);
    }

    let mut out = vec![0.0f32; rows.len() * h];
    for (i, &r) in rows.iter().enumerate() {
        let r = r as usize;
        ops::zc_rmsnorm_row(&x[r * h..(r + 1) * h], &model.norm, d.eps, &mut out[i * h..(i + 1) * h]);
    }
    out
}

/// Logits of `ids` (rows of `lm_head`) for each hidden row, accumulated in f64. `[hidden.len()/H, ids.len()]`.
pub fn readout(model: &Model, hidden: &[f32], ids: &[u32]) -> Vec<f32> {
    let h = model.dims.hidden;
    let w: Vec<Vec<f32>> = ids.par_iter().map(|&i| model.lm_head.row(i as usize)).collect();
    hidden
        .chunks_exact(h)
        .flat_map(|hr| {
            w.iter().map(|wr| hr.iter().zip(wr).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn add_into(x: &mut [f32], y: &[f32]) {
    x.par_iter_mut().zip(y.par_iter()).for_each(|(a, &b)| *a += b);
}

/// `Qwen3_5MLP`: `down(silu(gate(x)) * up(x))`.
fn mlp(layer: &Layer, hn: &[f32], t: usize) -> Vec<f32> {
    let g = matmul(hn, t, &layer.gate_proj);
    let u = matmul(hn, t, &layer.up_proj);
    let m: Vec<f32> = g.par_iter().zip(u.par_iter()).map(|(&g, &u)| ops::silu(g as f64) as f32 * u).collect();
    matmul(&m, t, &layer.down_proj)
}

/// `Qwen3_5GatedDeltaNet.forward` over the layout.
fn gdn(d: &Dims, w: &GdnWeights, hn: &[f32], layout: &Layout) -> Vec<f32> {
    let t = layout.pos.len();
    let (cd, kd, vd, kc) = (d.conv_dim(), d.key_dim(), d.value_dim(), d.conv_kernel);
    let qkv = matmul(hn, t, &w.in_proj_qkv);
    let z = matmul(hn, t, &w.in_proj_z);
    let b = matmul(hn, t, &w.in_proj_b);
    let a = matmul(hn, t, &w.in_proj_a);

    // Causal depthwise conv1d (kernel kc, no bias) + SiLU. out[t] = sum_j w[j] * x[t + j - (kc-1)].
    let mut conv = vec![0.0f32; t * cd];
    conv.par_chunks_mut(cd).enumerate().for_each(|(ti, out)| {
        let src: Vec<Option<usize>> = (0..kc)
            .map(|j| {
                let back = kc - 1 - j;
                if back == 0 { Some(ti) } else { layout.conv_prev[ti][back - 1].map(|s| s as usize) }
            })
            .collect();
        for (c, o) in out.iter_mut().enumerate() {
            let mut s = 0.0f64;
            for (j, sj) in src.iter().enumerate() {
                if let Some(sj) = sj {
                    s += w.conv[c * kc + j] as f64 * qkv[sj * cd + c] as f64;
                }
            }
            *o = ops::silu(s) as f32;
        }
    });

    // Per token and head: L2-normalised q (scaled by k_dim^-0.5) and k in f64, v, beta and the log decay g.
    let rep = d.v_heads / d.k_heads;
    let scale = (d.k_dim as f64).powf(-0.5);
    let mut qn = vec![0.0f64; t * kd];
    let mut kn = vec![0.0f64; t * kd];
    qn.par_chunks_mut(kd).zip(kn.par_chunks_mut(kd)).enumerate().for_each(|(ti, (q, k))| {
        let row = &conv[ti * cd..(ti + 1) * cd];
        for (o, &v) in q.iter_mut().zip(&row[..kd]) {
            *o = v as f64;
        }
        for (o, &v) in k.iter_mut().zip(&row[kd..2 * kd]) {
            *o = v as f64;
        }
        for hh in 0..d.k_heads {
            ops::l2norm_row(&mut q[hh * d.k_dim..(hh + 1) * d.k_dim], 1e-6);
            ops::l2norm_row(&mut k[hh * d.k_dim..(hh + 1) * d.k_dim], 1e-6);
        }
        q.iter_mut().for_each(|v| *v *= scale);
    });
    let nv = d.v_heads;
    let beta: Vec<f64> = b.iter().map(|&x| ops::sigmoid(x as f64)).collect();
    let g: Vec<f64> = a
        .iter()
        .enumerate()
        .map(|(i, &x)| -(w.a_log[i % nv] as f64).exp() * ops::softplus(x as f64 + w.dt_bias[i % nv] as f64))
        .collect();

    // Recurrent gated delta rule per (run, v-head); runs that start from zero first, then the ones that read
    // their parent's final state. State S[k][v] in f64.
    let (dk, dv) = (d.k_dim, d.v_dim);
    let scan = |run: &Run, hv: usize, init: Option<&Vec<f64>>| -> (Vec<f32>, Vec<f64>) {
        let hk = hv / rep;
        let mut s = init.cloned().unwrap_or_else(|| vec![0.0f64; dk * dv]);
        let mut out = vec![0.0f32; run.len * dv];
        let mut kv_mem = vec![0.0f64; dv];
        for i in 0..run.len {
            let ti = run.start + i;
            let q = &qn[ti * kd + hk * dk..ti * kd + (hk + 1) * dk];
            let k = &kn[ti * kd + hk * dk..ti * kd + (hk + 1) * dk];
            let vrow = &conv[ti * cd + 2 * kd + hv * dv..ti * cd + 2 * kd + (hv + 1) * dv];
            let decay = g[ti * nv + hv].exp();
            let bt = beta[ti * nv + hv];
            s.iter_mut().for_each(|x| *x *= decay);
            kv_mem.iter_mut().for_each(|x| *x = 0.0);
            for kk in 0..dk {
                let kv = k[kk];
                for (m, &sv) in kv_mem.iter_mut().zip(&s[kk * dv..(kk + 1) * dv]) {
                    *m += sv * kv;
                }
            }
            for kk in 0..dk {
                let kv = k[kk];
                for ((sv, &m), &vv) in s[kk * dv..(kk + 1) * dv].iter_mut().zip(&kv_mem).zip(vrow) {
                    *sv += kv * ((vv as f64 - m) * bt);
                }
            }
            let o = &mut out[i * dv..(i + 1) * dv];
            let mut acc = vec![0.0f64; dv];
            for kk in 0..dk {
                let qv = q[kk];
                for (a, &sv) in acc.iter_mut().zip(&s[kk * dv..(kk + 1) * dv]) {
                    *a += sv * qv;
                }
            }
            for (o, a) in o.iter_mut().zip(acc) {
                *o = a as f32;
            }
        }
        (out, s)
    };
    let mut core = vec![0.0f32; t * vd];
    let mut states: Vec<Option<Vec<Vec<f64>>>> = vec![None; layout.runs.len()];
    for phase in [false, true] {
        let tasks: Vec<(usize, usize)> = layout
            .runs
            .iter()
            .enumerate()
            .filter(|(_, r)| r.init.is_some() == phase)
            .flat_map(|(ri, _)| (0..nv).map(move |hv| (ri, hv)))
            .collect();
        let results: Vec<(usize, usize, Vec<f32>, Vec<f64>)> = tasks
            .par_iter()
            .map(|&(ri, hv)| {
                let run = &layout.runs[ri];
                let init = run.init.map(|p| &states[p].as_ref().expect("parent run has no state")[hv]);
                let (o, s) = scan(run, hv, init);
                (ri, hv, o, s)
            })
            .collect();
        for (ri, hv, o, s) in results {
            let run = &layout.runs[ri];
            for i in 0..run.len {
                let ti = run.start + i;
                core[ti * vd + hv * dv..ti * vd + (hv + 1) * dv].copy_from_slice(&o[i * dv..(i + 1) * dv]);
            }
            if !phase {
                states[ri].get_or_insert_with(|| vec![Vec::new(); nv])[hv] = s;
            }
        }
    }

    // Gated RMSNorm per v-head, then out_proj.
    let mut normed = vec![0.0f32; t * vd];
    normed.par_chunks_mut(dv).enumerate().for_each(|(i, o)| {
        ops::gated_rmsnorm_row(&core[i * dv..(i + 1) * dv], &z[i * dv..(i + 1) * dv], &w.norm, d.eps, o)
    });
    matmul(&normed, t, &w.out_proj)
}

/// `Qwen3_5Attention.forward` over the layout's key sets.
fn attention(d: &Dims, w: &AttnWeights, hn: &[f32], layout: &Layout) -> Vec<f32> {
    let t = layout.pos.len();
    let (nh, nkv, hd) = (d.heads, d.kv_heads, d.head_dim);
    let qg = matmul(hn, t, &w.q_proj);
    let mut k = matmul(hn, t, &w.k_proj);
    let v = matmul(hn, t, &w.v_proj);

    // q/gate split per head ([.., 2*hd] -> [:hd], [hd:]), q_norm/k_norm, partial RoPE.
    let mut q = vec![0.0f32; t * nh * hd];
    let mut gate = vec![0.0f32; t * nh * hd];
    q.par_chunks_mut(nh * hd)
        .zip(gate.par_chunks_mut(nh * hd))
        .zip(k.par_chunks_mut(nkv * hd))
        .enumerate()
        .for_each(|(ti, ((qr, gr), kr))| {
            let (cos, sin) = ops::rope_cos_sin(layout.pos[ti], d.rotary_dim, d.rope_theta);
            let src = &qg[ti * 2 * nh * hd..(ti + 1) * 2 * nh * hd];
            for hh in 0..nh {
                let qs = &src[hh * 2 * hd..hh * 2 * hd + hd];
                gr[hh * hd..(hh + 1) * hd].copy_from_slice(&src[hh * 2 * hd + hd..(hh + 1) * 2 * hd]);
                ops::zc_rmsnorm_row(qs, &w.q_norm, d.eps, &mut qr[hh * hd..(hh + 1) * hd]);
                ops::rope_apply(&mut qr[hh * hd..(hh + 1) * hd], &cos, &sin);
            }
            for hh in 0..nkv {
                let raw = kr[hh * hd..(hh + 1) * hd].to_vec();
                ops::zc_rmsnorm_row(&raw, &w.k_norm, d.eps, &mut kr[hh * hd..(hh + 1) * hd]);
                ops::rope_apply(&mut kr[hh * hd..(hh + 1) * hd], &cos, &sin);
            }
        });

    // softmax(q k^T * hd^-0.5) v per query head over its visible keys (GQA: q-head h reads kv-head h / rep),
    // then * sigmoid(gate).
    let rep = nh / nkv;
    let scaling = (hd as f64).powf(-0.5);
    let mut o = vec![0.0f32; t * nh * hd];
    o.par_chunks_mut(nh * hd).enumerate().for_each(|(ti, orow)| {
        let keys = &layout.keys[ti];
        for hh in 0..nh {
            let kvh = hh / rep;
            let qv = &q[(ti * nh + hh) * hd..(ti * nh + hh + 1) * hd];
            let scores: Vec<f64> = keys
                .iter()
                .map(|&j| {
                    let kv = &k[(j as usize * nkv + kvh) * hd..(j as usize * nkv + kvh + 1) * hd];
                    qv.iter().zip(kv).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scaling
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f64 = e.iter().sum();
            let mut acc = vec![0.0f64; hd];
            for (&j, p) in keys.iter().zip(&e) {
                let vv = &v[(j as usize * nkv + kvh) * hd..(j as usize * nkv + kvh + 1) * hd];
                for (a, &x) in acc.iter_mut().zip(vv) {
                    *a += p / sum * x as f64;
                }
            }
            let g = &gate[(ti * nh + hh) * hd..(ti * nh + hh + 1) * hd];
            for ((o, a), &gv) in orow[hh * hd..(hh + 1) * hd].iter_mut().zip(acc).zip(g) {
                *o = ((a as f32) as f64 * ops::sigmoid(gv as f64)) as f32;
            }
        }
    });
    matmul(&o, t, &w.o_proj)
}
