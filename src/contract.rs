//! The contract between workstreams (DESIGN.md §6.1, §11).
//!
//! The frontend builds a [`DecisionBatch`] from rendered requests, the scheduler packs several into one, and
//! a [`DecisionEngine`] turns it into [`SlotLogits`]. Every producer must hand over a batch that passes
//! [`DecisionBatch::validate`]; every engine may assume it.

use anyhow::{Result, bail, ensure};

/// Number of answer slots the readout exposes per branch (`qev.prompt.MAX_SLOTS`).
pub const MAX_SLOTS: usize = 256;

/// One contiguous run of tokens in the packed stream: a shared prefix (system turn + state block), or one
/// question suffix whose prefix is `parent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Segment {
    /// First token of the segment in the packed stream.
    pub start: u32,
    pub len: u32,
    /// -1 for a prefix, else the index (into `DecisionBatch::segments`) of its prefix segment.
    pub parent: i32,
    /// RoPE position of the first token: 0 for a prefix, the parent's length for a suffix.
    pub pos0: u32,
}

impl Segment {
    pub fn is_prefix(&self) -> bool {
        self.parent < 0
    }

    pub fn end(&self) -> u32 {
        self.start + self.len
    }
}

/// One GPU step's worth of work: several requests, each a prefix segment followed by its suffix segments,
/// packed into one token stream without padding.
///
/// Invariants (checked by [`validate`](Self::validate)):
/// - segments tile `token_ids` exactly, in order, with no gaps or overlap;
/// - every suffix's parent is an earlier prefix segment, and its `pos0` equals the parent's `len`;
/// - every prefix has `pos0 == 0`;
/// - `answer_rows[i]` is the last token of the i-th suffix segment (in segment order), so there is exactly
///   one answer row per suffix, and `answer_slots[i]` is in `1..=MAX_SLOTS`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecisionBatch {
    pub token_ids: Vec<u32>,
    pub segments: Vec<Segment>,
    pub answer_rows: Vec<u32>,
    pub answer_slots: Vec<u16>,
}

impl DecisionBatch {
    pub fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    pub fn num_branches(&self) -> usize {
        self.answer_rows.len()
    }

    /// Append one request: its prefix ids and one suffix (with its slot count) per branch. Returns the index
    /// of the request's first answer row, so the caller can map logits rows back to its branches.
    pub fn push_request(&mut self, prefix: &[u32], suffixes: &[(&[u32], u16)]) -> Result<usize> {
        ensure!(!prefix.is_empty(), "empty prefix");
        ensure!(!suffixes.is_empty(), "request has no branches");
        let first_row = self.answer_rows.len();
        let parent = self.segments.len() as i32;
        let p_len = prefix.len() as u32;
        self.segments.push(Segment { start: self.token_ids.len() as u32, len: p_len, parent: -1, pos0: 0 });
        self.token_ids.extend_from_slice(prefix);
        for &(suffix, slots) in suffixes {
            ensure!(!suffix.is_empty(), "empty suffix");
            let start = self.token_ids.len() as u32;
            self.segments.push(Segment { start, len: suffix.len() as u32, parent, pos0: p_len });
            self.token_ids.extend_from_slice(suffix);
            self.answer_rows.push(start + suffix.len() as u32 - 1);
            self.answer_slots.push(slots);
        }
        Ok(first_row)
    }

    pub fn validate(&self) -> Result<()> {
        let mut cursor = 0u32;
        let mut row = 0usize;
        for (i, s) in self.segments.iter().enumerate() {
            ensure!(s.start == cursor, "segment {i} starts at {} but the previous one ends at {cursor}", s.start);
            ensure!(s.len > 0, "segment {i} is empty");
            cursor = s.end();
            if s.is_prefix() {
                ensure!(s.pos0 == 0, "prefix segment {i} has pos0 {}", s.pos0);
                continue;
            }
            let p = s.parent as usize;
            ensure!(p < i, "segment {i} has parent {p}, which is not an earlier segment");
            let parent = &self.segments[p];
            ensure!(parent.is_prefix(), "segment {i} has parent {p}, which is not a prefix");
            ensure!(s.pos0 == parent.len, "segment {i} has pos0 {} but its prefix has {} tokens", s.pos0, parent.len);
            let Some(&ans) = self.answer_rows.get(row) else { bail!("suffix segment {i} has no answer row") };
            ensure!(ans == s.end() - 1, "answer row {row} is {ans}, expected the last token of segment {i}");
            let slots = self.answer_slots.get(row).copied().unwrap_or(0) as usize;
            ensure!((1..=MAX_SLOTS).contains(&slots), "answer row {row} has {slots} slots");
            row += 1;
        }
        ensure!(cursor as usize == self.token_ids.len(), "segments cover {cursor} of {} tokens", self.token_ids.len());
        ensure!(row == self.answer_rows.len(), "{} answer rows for {row} suffix segments", self.answer_rows.len());
        ensure!(self.answer_slots.len() == row, "{} slot counts for {row} answer rows", self.answer_slots.len());
        Ok(())
    }
}

/// Slot logits, one row per answer row of the batch, `MAX_SLOTS` wide, row-major. Only the first
/// `answer_slots[i]` entries of row i are meaningful; the rest are unspecified.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotLogits {
    pub rows: usize,
    pub data: Vec<f32>,
}

impl SlotLogits {
    pub fn row(&self, i: usize, slots: usize) -> &[f32] {
        &self.data[i * MAX_SLOTS..i * MAX_SLOTS + slots]
    }
}

/// A model that scores packed decision batches. One engine owns one GPU; the scheduler calls it from a
/// single worker thread.
pub trait DecisionEngine: Send {
    /// Largest `num_tokens()` a single [`run`](Self::run) accepts.
    fn max_tokens_per_step(&self) -> usize;

    /// Score every branch of `batch`. The batch has passed [`DecisionBatch::validate`].
    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits>;

    /// [`run`](Self::run), plus each answer row's `allowed_token_mass` (qev's debug diagnostic, DESIGN.md §4.4):
    /// the share of the full-vocabulary softmax that falls on the row's `answer_slots[i]` slot tokens,
    /// `exp(logsumexp(slot logits) - logsumexp(all vocab logits))`. [`allowed_mass`] computes it the way qev does
    /// from the slot logits and the full-vocabulary logsumexp.
    ///
    /// The scheduler calls this instead of `run` only for batches that hold a `debug` request, because it needs
    /// the full LM head at the answer rows. The default implementation reports no masses (`None`), so an engine
    /// that does not implement it still serves debug requests, without `allowed_token_mass`.
    fn run_with_mass(&mut self, batch: &DecisionBatch) -> Result<MassReadout> {
        Ok(MassReadout { logits: self.run(batch)?, allowed_mass: None })
    }

    /// Static facts about the engine reported as labels of `sysone_engine_info` in `/metrics` (e.g. the GDN
    /// backend and the device architecture). Default: none.
    fn info(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    /// What one step can hold, for callers that pack batches to fit the engine exactly (`sysone bulk`). Default:
    /// one bucket of [`max_tokens_per_step`](Self::max_tokens_per_step) tokens and no other limit.
    fn step_limits(&self) -> StepLimits {
        StepLimits::tokens_only(self.max_tokens_per_step())
    }

    /// Token capacity of the bucket(s) the last [`run`](Self::run) used (summed over the steps a split batch ran
    /// in), so `capacity - batch tokens` is the padding it computed. None when the engine has no buckets.
    fn last_bucket_tokens(&self) -> Option<usize> {
        None
    }
}

/// The capacities of an engine step ([`DecisionEngine::step_limits`]). A batch with at most `buckets.last()`
/// tokens, `max_answers` answer rows, `max_segments` segments and `max_prefixes` prefixes runs as one step in the
/// smallest bucket that holds its tokens and its answers ([`answers_in`](Self::answers_in)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepLimits {
    /// Token buckets, ascending (the engine pads a batch up to the smallest one that fits).
    pub buckets: Vec<usize>,
    pub max_answers: usize,
    pub max_segments: usize,
    pub max_prefixes: usize,
    /// A bucket of T tokens holds [`bucket_answers`]`(T, max_answers, tokens_per_answer)` answer rows (1: up to
    /// `min(T, max_answers)`).
    pub tokens_per_answer: usize,
}

/// Answer rows a bucket of `tokens` tokens holds when it allows one answer per `tokens_per_answer` tokens and at
/// most `max_answers` (the GPU engine's [`BucketPlan`](crate::engine::graph::BucketPlan)): `min(T, max_answers)`
/// for `tokens_per_answer == 1`, else `min(ceil(T / tokens_per_answer), max_answers)`.
pub fn bucket_answers(tokens: usize, max_answers: usize, tokens_per_answer: usize) -> usize {
    match tokens_per_answer {
        0 | 1 => tokens.min(max_answers).max(1),
        d => tokens.div_ceil(d).min(max_answers).max(1),
    }
}

impl StepLimits {
    pub fn tokens_only(max_tokens: usize) -> Self {
        StepLimits { buckets: vec![max_tokens], max_answers: usize::MAX, max_segments: usize::MAX, max_prefixes: usize::MAX,
                     tokens_per_answer: 1 }
    }

    /// Answer rows a step in the bucket of `bucket_tokens` tokens holds.
    pub fn answers_in(&self, bucket_tokens: usize) -> usize {
        bucket_answers(bucket_tokens, self.max_answers, self.tokens_per_answer)
    }

    pub fn max_tokens(&self) -> usize {
        self.buckets.last().copied().unwrap_or(0)
    }

    /// The smallest bucket holding `tokens`.
    pub fn bucket_for(&self, tokens: usize) -> Option<usize> {
        self.buckets.iter().copied().find(|&b| b >= tokens)
    }
}

/// What [`DecisionEngine::run_with_mass`] returns.
#[derive(Clone, Debug, PartialEq)]
pub struct MassReadout {
    pub logits: SlotLogits,
    /// One `allowed_token_mass` per answer row, or None when the engine cannot compute it. f64 because qev
    /// reports a Python float; an engine computing in f32 (as qev does) widens exactly.
    pub allowed_mass: Option<Vec<f64>>,
}

/// qev's `allowed_token_mass` for one answer row: `float((logsumexp(sel) - logsumexp(v)).exp())` in fp32, where
/// `sel` are the row's slot logits (its first `answer_slots[i]` entries) and `vocab_lse` the logsumexp of the
/// row's full-vocabulary logits (`qev.engine.DecisionEngine._readout`).
pub fn allowed_mass(slot_logits: &[f32], vocab_lse: f32) -> f64 {
    (logsumexp_f32(slot_logits) - vocab_lse).exp() as f64
}

/// `torch.logsumexp` in fp32: `max + log(sum(exp(x - max)))` (summed in order; the GPU's reduction order may
/// differ in the last bits).
pub fn logsumexp_f32(x: &[f32]) -> f32 {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return m;
    }
    m + x.iter().map(|&v| (v - m).exp()).sum::<f32>().ln()
}

/// A deterministic stand-in engine for scheduler and server work (W5) before the real one exists.
/// Each logit depends only on its branch's own tokens and slot index, so results are batch-invariant.
pub struct MockEngine {
    pub max_tokens: usize,
}

impl DecisionEngine for MockEngine {
    fn max_tokens_per_step(&self) -> usize {
        self.max_tokens
    }

    fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
        ensure!(batch.num_tokens() <= self.max_tokens, "batch of {} tokens over {}", batch.num_tokens(), self.max_tokens);
        let mut data = vec![0.0f32; batch.num_branches() * MAX_SLOTS];
        let suffixes = batch.segments.iter().filter(|s| !s.is_prefix());
        for (row, s) in suffixes.enumerate() {
            let prefix = &batch.segments[s.parent as usize];
            let toks = batch.token_ids[prefix.start as usize..prefix.end() as usize]
                .iter()
                .chain(&batch.token_ids[s.start as usize..s.end() as usize]);
            let h = toks.fold(0xcbf29ce484222325u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100000001b3));
            for k in 0..MAX_SLOTS {
                let x = (h ^ k as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 40;
                data[row * MAX_SLOTS + k] = (x as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
            }
        }
        Ok(SlotLogits { rows: batch.num_branches(), data })
    }

    /// Masses from a made-up full-vocabulary logsumexp: the slot logsumexp plus a per-row offset in [0.05, 3)
    /// that depends only on the row's logits, so masses are batch-invariant like the logits.
    fn run_with_mass(&mut self, batch: &DecisionBatch) -> Result<MassReadout> {
        let logits = self.run(batch)?;
        let mass = (0..logits.rows)
            .map(|i| {
                let row = logits.row(i, batch.answer_slots[i] as usize);
                let h = row.iter().fold(0x84222325u32, |h, v| (h ^ v.to_bits()).wrapping_mul(0x01000193));
                let offset = 0.05 + (h >> 8) as f32 / (1u32 << 24) as f32 * 2.95;
                allowed_mass(row, logsumexp_f32(row) + offset)
            })
            .collect();
        Ok(MassReadout { logits, allowed_mass: Some(mass) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_requests() -> DecisionBatch {
        let mut b = DecisionBatch::default();
        b.push_request(&[1, 2, 3], &[(&[10, 11], 2), (&[12], 5)]).unwrap();
        b.push_request(&[4, 5], &[(&[20, 21, 22], 3)]).unwrap();
        b
    }

    #[test]
    fn push_builds_a_valid_tree() {
        let b = two_requests();
        b.validate().unwrap();
        assert_eq!(b.token_ids, [1, 2, 3, 10, 11, 12, 4, 5, 20, 21, 22]);
        assert_eq!(b.answer_rows, [4, 5, 10]);
        assert_eq!(b.segments[4], Segment { start: 8, len: 3, parent: 3, pos0: 2 });
    }

    #[test]
    fn validate_rejects_broken_batches() {
        let mut b = two_requests();
        b.segments[1].pos0 = 0;
        assert!(b.validate().is_err());
        let mut b = two_requests();
        b.answer_rows[0] = 3;
        assert!(b.validate().is_err());
        let mut b = two_requests();
        b.segments[4].parent = 1;
        assert!(b.validate().is_err());
        let mut b = two_requests();
        b.token_ids.push(0);
        assert!(b.validate().is_err());
    }

    #[test]
    fn allowed_mass_matches_qev_formula() {
        // All of the vocabulary on the slots: mass 1. Two equal slots among four equal tokens: 1/2.
        assert_eq!(allowed_mass(&[0.0], 0.0), 1.0);
        let lse4 = logsumexp_f32(&[1.0, 1.0, 1.0, 1.0]);
        assert!((allowed_mass(&[1.0, 1.0], lse4) - 0.5).abs() < 1e-6);
        // Large logits do not overflow.
        assert!((logsumexp_f32(&[1000.0, 1000.0]) - (1000.0 + 2f32.ln())).abs() < 1e-3);
        assert_eq!(logsumexp_f32(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn mock_engine_masses_are_batch_invariant_and_in_range() {
        let mut alone = DecisionBatch::default();
        alone.push_request(&[4, 5], &[(&[20, 21, 22], 3)]).unwrap();
        let mut e = MockEngine { max_tokens: 1 << 16 };
        let a = e.run_with_mass(&alone).unwrap();
        let both = e.run_with_mass(&two_requests()).unwrap();
        assert_eq!(both.logits, e.run(&two_requests()).unwrap());
        let (ma, mb) = (a.allowed_mass.unwrap(), both.allowed_mass.unwrap());
        assert_eq!((ma.len(), mb.len()), (1, 3));
        assert_eq!(ma[0], mb[2]);
        assert!(mb.iter().all(|&m| m > 0.04 && m < 0.96), "{mb:?}");
    }

    #[test]
    fn default_run_with_mass_reports_none() {
        struct Plain;
        impl DecisionEngine for Plain {
            fn max_tokens_per_step(&self) -> usize {
                100
            }
            fn run(&mut self, batch: &DecisionBatch) -> Result<SlotLogits> {
                Ok(SlotLogits { rows: batch.num_branches(), data: vec![0.0; batch.num_branches() * MAX_SLOTS] })
            }
        }
        let r = Plain.run_with_mass(&two_requests()).unwrap();
        assert_eq!((r.logits.rows, r.allowed_mass), (3, None));
    }

    #[test]
    fn mock_engine_is_batch_invariant() {
        let mut alone = DecisionBatch::default();
        alone.push_request(&[4, 5], &[(&[20, 21, 22], 3)]).unwrap();
        let mut e = MockEngine { max_tokens: 1 << 16 };
        let a = e.run(&alone).unwrap();
        let both = e.run(&two_requests()).unwrap();
        assert_eq!(a.row(0, 3), both.row(2, 3));
    }
}
