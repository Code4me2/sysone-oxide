//! CPU f32 reference forward over [`DecisionBatch`] trees (DESIGN.md §6.1). Owner: W4b.
//!
//! The oracle for the design's core claim: one packed pass over a tree of segments equals the reference's
//! prefix-then-suffix computation. It is slow (minutes for ~1k tokens) and exists for tests:
//! - [`CpuRefEngine`] implements [`DecisionEngine`] over bf16 weights read through mmap;
//! - [`Layout::from_batch`] is the tree semantics (conv taps and delta-rule state from the parent prefix, block
//!   attention per segment); [`Layout::flat`] is the reference's single-sequence semantics with the explicit
//!   `causal OR same-block` mask, used to check the tree against it;
//! - [`check`] compares slot logits against the qev fp32 run recorded by `reference/hf_check/run_qev.py`.

pub mod check;
pub mod model;
pub mod ops;
pub mod weights;

use crate::contract::{DecisionBatch, DecisionEngine, MAX_SLOTS, SlotLogits};
use anyhow::{Context, Result, ensure};
pub use model::{Dims, Layout, Model, Run};
use std::path::Path;

impl Layout {
    /// The tree semantics of a validated batch:
    /// - RoPE position `pos0 + offset`;
    /// - conv taps before a segment's first token come from its parent prefix's last tokens (zeros for a prefix);
    /// - a prefix token sees its whole prefix; a suffix token sees its parent prefix and its whole own suffix;
    /// - each segment is one delta-rule run; a suffix starts from its parent prefix's final state.
    pub fn from_batch(batch: &DecisionBatch, conv_kernel: usize) -> Self {
        let n = batch.num_tokens();
        let mut l = Layout {
            pos: vec![0; n],
            conv_prev: vec![Vec::new(); n],
            keys: vec![Vec::new(); n],
            runs: Vec::with_capacity(batch.segments.len()),
        };
        for s in &batch.segments {
            let parent = (!s.is_prefix()).then(|| batch.segments[s.parent as usize]);
            let mut keys: Vec<u32> = Vec::new();
            if let Some(p) = parent {
                keys.extend(p.start..p.end());
            }
            keys.extend(s.start..s.end());
            for i in 0..s.len {
                let t = (s.start + i) as usize;
                l.pos[t] = s.pos0 + i;
                l.conv_prev[t] = (1..conv_kernel as i64)
                    .map(|d| {
                        let off = i as i64 - d;
                        if off >= 0 {
                            return Some(s.start + off as u32);
                        }
                        let p = parent?;
                        let po = p.len as i64 + off;
                        (po >= 0).then(|| p.start + po as u32)
                    })
                    .collect();
                l.keys[t] = keys.clone();
            }
            l.runs.push(Run { start: s.start as usize, len: s.len as usize, init: parent.map(|_| s.parent as usize) });
        }
        l
    }

    /// One sequence of `n` tokens, the first `prefix` of them block 0 and the rest block 1, as the reference
    /// computes a full branch prompt in one forward: positions `0..n`, conv over the sequence with zero padding,
    /// one delta-rule run from zero, and the full-attention mask `kv <= q OR (block[q] == block[kv] AND
    /// block[q] >= 0)` (`masking_utils` causal mask OR `blockwise_overlay`).
    pub fn flat(n: usize, prefix: usize, conv_kernel: usize) -> Self {
        let block = |i: usize| if i < prefix { 0 } else { 1 };
        Layout {
            pos: (0..n as u32).collect(),
            conv_prev: (0..n)
                .map(|i| (1..conv_kernel).map(|d| i.checked_sub(d).map(|j| j as u32)).collect())
                .collect(),
            keys: (0..n)
                .map(|q| (0..n).filter(|&kv| kv <= q || block(q) == block(kv)).map(|kv| kv as u32).collect())
                .collect(),
            runs: vec![Run { start: 0, len: n, init: None }],
        }
    }
}

/// `qev.prompt.SLOT_LABELS`: A..Z, then the two-letter labels that are one token, 256 in total.
pub fn slot_labels() -> Vec<String> {
    const NOT_ONE_TOKEN: [&str; 25] = [
        "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ", "HJ",
        "IY", "JF", "JG", "JH", "JL", "JN", "JQ",
    ];
    let letters: Vec<char> = ('A'..='Z').collect();
    let mut out: Vec<String> = letters.iter().map(|c| c.to_string()).collect();
    for a in &letters {
        for b in &letters {
            let s = format!("{a}{b}");
            if out.len() < MAX_SLOTS && !NOT_ONE_TOKEN.contains(&s.as_str()) {
                out.push(s);
            }
        }
    }
    out
}

/// Slot token ids looked up in `tokenizer.json`'s BPE vocabulary (ASCII letters are their own byte-level
/// symbols). The reference verifies each label round-trips as one token; tests compare these ids with the ids
/// the reference recorded.
pub fn slot_ids_from_tokenizer(model_dir: &Path) -> Result<Vec<u32>> {
    let path = model_dir.join("tokenizer.json");
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let vocab = v["model"]["vocab"].as_object().context("tokenizer.json has no model.vocab")?;
    slot_labels()
        .iter()
        .map(|l| vocab.get(l).and_then(|x| x.as_u64()).map(|x| x as u32).with_context(|| format!("slot {l} not in vocab")))
        .collect()
}

/// The reference as a [`DecisionEngine`]: slow, CPU, for tests.
pub struct CpuRefEngine {
    pub model: Model,
    pub slot_ids: Vec<u32>,
    pub max_tokens: usize,
}

impl CpuRefEngine {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model = Model::load(model_dir)?;
        let slot_ids = slot_ids_from_tokenizer(model_dir)?;
        Ok(CpuRefEngine { model, slot_ids, max_tokens: 1 << 15 })
    }

    pub fn from_parts(model: Model, slot_ids: Vec<u32>) -> Result<Self> {
        ensure!(slot_ids.len() == MAX_SLOTS, "{} slot ids, expected {MAX_SLOTS}", slot_ids.len());
        Ok(CpuRefEngine { model, slot_ids, max_tokens: 1 << 15 })
    }

    /// Final-norm hidden states at the answer rows, `[Q, hidden]`.
    pub fn answer_hidden(&self, batch: &DecisionBatch) -> Vec<f32> {
        let layout = Layout::from_batch(batch, self.model.dims.conv_kernel);
        model::forward_hidden(&self.model, &batch.token_ids, &layout, &batch.answer_rows)
    }
}

impl DecisionEngine for CpuRefEngine {
    fn max_tokens_per_step(&self) -> usize {
        self.max_tokens
    }

    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
        ensure!(batch.num_tokens() <= self.max_tokens, "batch of {} tokens over {}", batch.num_tokens(), self.max_tokens);
        let hidden = self.answer_hidden(batch);
        let data = model::readout(&self.model, &hidden, &self.slot_ids);
        Ok(SlotLogits { rows: batch.num_branches(), data })
    }
}

#[cfg(test)]
mod tests;
