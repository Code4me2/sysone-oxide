//! Request → branches → token ids: qev's `decide()` up to the forward (OneJev@81ce62f + serve.patch).
//!
//! 1. Branches: one per question; a choice with more than `max_slots` options is split into balanced chunks plus a
//!    shared "none of the options listed here" slot (`_plan_chunks`); `debias=k` adds `k-1` cyclic shifts.
//! 2. Prefix split (`_prefix_and_suffix_ids`): the shared prefix is the longest common token prefix of all branch
//!    prompts that ends at or before the state block's end, located with a marker render.
//! 3. Limits: prefix + longest suffix ≤ `max_branch_tokens`, prefix + all suffixes ≤ `max_request_tokens`.
//! 4. serve.patch's `_split`: under noncausal attention the prefix block (tokens ending at or before the suffix)
//!    must end exactly at the shared prefix; it also yields each option line's token span.
//!
//! Every branch prompt is `head + suffix + tail` (see
//! [`ChatTemplate::frame`](crate::frontend::template::ChatTemplate::frame)), and the pre-tokenizer always splits
//! between the head (which ends with `</state>\n\n`) and a suffix (which starts with `Question:` or `Rate the
//! state:`), so a branch's tokens are `encode(head) ++ encode(suffix + tail)`. The head, which holds the state, is
//! therefore rendered and tokenized once per request instead of once per branch. The golden gate checks the result
//! against full-prompt tokenizations of the reference.

use sha2::{Digest, Sha256};

use crate::api::schema::{Entry, Question, QuestionKind, SystemOneRequest};
use crate::contract::DecisionBatch;
use crate::frontend::Frontend;
use crate::frontend::pyjson::Json;
use crate::frontend::render::{QuestionParts, SYSTEM_PROMPT, question_parts, render_question, state_block};

pub const NONE_OPTION: &str = "none of the options listed here";
pub const NONE_DESCRIPTION: &str = "The correct option is not among the options listed in this question.";
/// The marker qev renders in place of a suffix to find the end of the state block.
pub const PREFIX_MARK: &str = "QEV_PREFIX_MARK";

/// qev's `DecisionEngine` defaults (DESIGN §4.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_branch_tokens: usize,
    pub max_request_tokens: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_branch_tokens: 32768, max_request_tokens: 65536 }
    }
}

/// Why a request cannot be planned.
#[derive(Debug)]
pub enum PlanError {
    /// `decide()` raised `ValueError(msg)`: qev answers 422 with `msg` (see `ApiError::value_error`).
    Value(String),
    /// Anything else (tokenizer or template failure): a 500.
    Internal(anyhow::Error),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::Value(m) => write!(f, "ValueError: {m}"),
            PlanError::Internal(e) => write!(f, "internal error: {e:#}"),
        }
    }
}

impl std::error::Error for PlanError {}

impl From<anyhow::Error> for PlanError {
    fn from(e: anyhow::Error) -> Self {
        PlanError::Internal(e)
    }
}

/// One branch: one forward suffix and one answer row.
#[derive(Clone, Debug, PartialEq)]
pub struct Branch {
    /// Index of the question in `SystemOneRequest::questions`.
    pub question: usize,
    pub kind: QuestionKind,
    /// The chunk index of a split choice, None otherwise.
    pub chunk: Option<usize>,
    /// qev's owner order: slot i shows original option `order[i]` (for a chunk, `0..=len(chunk names)`).
    pub order: Vec<usize>,
    pub labels: Vec<String>,
    pub descriptions: Vec<Entry>,
    pub suffix_text: String,
    pub option_char_spans: Vec<(usize, usize)>,
    /// The branch's tokens after the shared prefix.
    pub suffix_ids: Vec<u32>,
    /// serve.patch `_split`: each option line's token span relative to the suffix. Empty until
    /// [`Frontend::check`] has run (and always empty under causal attention).
    pub option_token_spans: Vec<(usize, usize)>,
    /// Code-point offsets of `suffix_ids` within `suffix_text + tail` (None when the prefix split is not at the
    /// head boundary, where `_split` refuses the request anyway).
    offsets: Option<Vec<(usize, usize)>>,
}

impl Branch {
    pub fn n_slots(&self) -> usize {
        self.labels.len()
    }
}

/// A planned request: everything `decide()` computes before the forward.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub branches: Vec<Branch>,
    pub prefix_ids: Vec<u32>,
    /// Tokens of the prompt head (system turn and state block): the `_split` prefix block.
    pub head_tokens: usize,
    /// Prefix + longest suffix, and prefix + every suffix (the two limit quantities; `total` is
    /// `usage.input_tokens`).
    pub longest: usize,
    pub total: usize,
    /// Every branch prompt is `prompt_head + branch.suffix_text + prompt_tail`.
    pub prompt_head: String,
    pub prompt_tail: String,
    /// sha256 of the first branch's full prompt (qev meta `prompt_sha256`).
    pub prompt_sha256: String,
    pub num_questions: usize,
}

impl Plan {
    /// Append this request to a batch; returns the batch row of branch 0 (branch i is row `first + i`).
    pub fn push_into(&self, batch: &mut DecisionBatch) -> anyhow::Result<usize> {
        let suffixes: Vec<(&[u32], u16)> =
            self.branches.iter().map(|b| (b.suffix_ids.as_slice(), b.n_slots() as u16)).collect();
        batch.push_request(&self.prefix_ids, &suffixes)
    }

    /// The branch prompt (template applied) of branch `i`.
    pub fn prompt(&self, i: usize) -> String {
        [self.prompt_head.as_str(), &self.branches[i].suffix_text, &self.prompt_tail].concat()
    }
}

/// A branch's tokens after the head and their code-point offsets in `suffix_text + tail`.
type SuffixTokens = (Vec<u32>, Vec<(usize, usize)>);

/// The part of planning that depends only on the questions and `debias`: the rendered branches and the tokens of
/// each `suffix_text + tail` ([`Frontend::prepare_questions`]).
///
/// A branch's tokens after the head are `encode(suffix_text + tail)` (module docs), and neither string depends on
/// the state: the suffix is rendered from the question alone, and the tail is what the chat template emits after
/// the user turn (`<|im_end|>` and the generation prompt; the turn ends with the suffix's `.`, so `|trim` never
/// reaches it). [`Frontend::plan_prepared`] still renders each request's own frame and compares its tail with
/// this one, so a template whose tail did depend on the state would only lose the reuse, not correctness.
#[derive(Clone, Debug)]
pub struct PreparedQuestions {
    branches: Vec<Branch>,
    num_questions: usize,
    tail: String,
    own: Vec<SuffixTokens>,
}

impl PreparedQuestions {
    pub fn num_branches(&self) -> usize {
        self.branches.len()
    }

    /// The template's text after the suffix that the cached tokens were computed with: a plan whose
    /// `prompt_tail` equals it reused them.
    pub fn tail(&self) -> &str {
        &self.tail
    }

    /// Tokens of the longest and of all question suffixes (tail included).
    pub fn suffix_tokens(&self) -> (usize, usize) {
        let lens = self.own.iter().map(|(ids, _)| ids.len());
        (lens.clone().max().unwrap_or(0), lens.sum())
    }
}

/// `_plan_chunks`: balanced chunks of at most `max_slots - 1` option names.
pub fn plan_chunks(n: usize, max_slots: usize) -> Vec<std::ops::Range<usize>> {
    let n_chunks = n.div_ceil(max_slots - 1);
    let size = n.div_ceil(n_chunks);
    (0..n).step_by(size).map(|s| s..(s + size).min(n)).collect()
}

/// Insert into an ordered map with Python dict semantics (an existing key keeps its position).
fn dict_set(m: &mut Vec<(String, Entry)>, k: &str, v: Entry) {
    match m.iter_mut().find(|(key, _)| key == k) {
        Some(slot) => slot.1 = v,
        None => m.push((k.to_string(), v)),
    }
}

impl Frontend {
    /// Plan a request and run every check `decide()` and `_split` make, in the reference's order.
    pub fn plan(&self, req: &SystemOneRequest, debias: i64) -> Result<Plan, PlanError> {
        let mut plan = self.plan_unchecked(req, debias)?;
        self.check(&mut plan)?;
        Ok(plan)
    }

    /// Branches, rendering, tokenization and the prefix split, without the limit and `_split` checks.
    pub fn plan_unchecked(&self, req: &SystemOneRequest, debias: i64) -> Result<Plan, PlanError> {
        let branches = self.render_branches(&req.questions, debias)?;
        self.plan_branches(&req.state, branches, req.questions.len(), None)
    }

    /// Step 1 of planning, which depends only on the questions and `debias` (not on the state): the branches in
    /// `decide()`'s order with their suffix text and slots, without tokens.
    pub fn render_branches(&self, questions: &[(String, Question)], debias: i64) -> Result<Vec<Branch>, PlanError> {
        let mut branches: Vec<Branch> = Vec::new();
        let mut add = |question: usize, chunk, order: Vec<usize>, parts: &QuestionParts<'_>, permute: bool| {
            let r = render_question(parts, permute.then_some(order.as_slice())).map_err(PlanError::Value)?;
            branches.push(Branch {
                question,
                kind: r.kind,
                chunk,
                order,
                labels: r.labels,
                descriptions: r.descriptions,
                suffix_text: r.suffix,
                option_char_spans: r.option_spans,
                suffix_ids: Vec::new(),
                option_token_spans: Vec::new(),
                offsets: None,
            });
            Ok::<_, PlanError>(())
        };
        for (qi, (_, q)) in questions.iter().enumerate() {
            if let Question::Choice { instructions, criteria } = q
                && criteria.len() > self.max_slots
            {
                for (ci, range) in plan_chunks(criteria.len(), self.max_slots).into_iter().enumerate() {
                    let mut sub = criteria[range.clone()].to_vec();
                    dict_set(&mut sub, NONE_OPTION, Json::str(NONE_DESCRIPTION));
                    let sub = Question::Choice { instructions: instructions.clone(), criteria: sub };
                    add(qi, Some(ci), (0..range.len() + 1).collect(), &question_parts(&sub), false)?;
                }
                continue;
            }
            let parts = question_parts(q);
            let k = parts.labels.len();
            let shifts = if k > 1 { (debias.max(1) as usize).min(k) } else { 1 };
            for s in 0..shifts {
                let order: Vec<usize> = (0..k).map(|i| (i + s) % k).collect();
                add(qi, None, order, &parts, s != 0)?;
            }
        }
        Ok(branches)
    }

    /// A question set's branches rendered and their suffixes tokenized once, for planning the same questions over
    /// many states with [`Frontend::plan_prepared`] (`sysone bulk --questions`, docs/BULK.md).
    pub fn prepare_questions(&self, questions: &[(String, Question)], debias: i64) -> Result<PreparedQuestions, PlanError> {
        let branches = self.render_branches(questions, debias)?;
        if branches.is_empty() {
            return Err(PlanError::Internal(anyhow::anyhow!("a question set without branches")));
        }
        // The tail of an empty state's prompt. A plan uses these tokens only when its own tail is the same text.
        let (_, tail) = self.template.frame(SYSTEM_PROMPT, &state_block(&Json::str("")), &branches[0].suffix_text)?;
        let mut own = Vec::with_capacity(branches.len());
        let mut text = String::new();
        for b in &branches {
            text.clear();
            text.push_str(&b.suffix_text);
            text.push_str(&tail);
            own.push(self.tokenizer.encode_with_offsets(&text)?);
        }
        Ok(PreparedQuestions { branches, num_questions: questions.len(), tail, own })
    }

    /// [`Frontend::plan`] of a request with this `state` and the prepared questions: the same plan (the same
    /// checks and errors), with the suffix tokens reused instead of recomputed.
    pub fn plan_prepared(&self, state: &Json, pq: &PreparedQuestions) -> Result<Plan, PlanError> {
        let mut plan = self.plan_branches(state, pq.branches.clone(), pq.num_questions, Some(pq))?;
        self.check(&mut plan)?;
        Ok(plan)
    }

    /// Planning from step 2 on: the prompt frame for `state`, tokenization and the prefix split. `cache`: the
    /// suffix tokens of these same branches, used when this prompt's tail is the cache's text (the tokens of
    /// `suffix_text + tail` are a function of that text alone).
    fn plan_branches(
        &self,
        state: &Json,
        mut branches: Vec<Branch>,
        num_questions: usize,
        cache: Option<&PreparedQuestions>,
    ) -> Result<Plan, PlanError> {
        // 2. Render the prompt frame once and tokenize the head once.
        let block = state_block(state);
        let (head, tail) = self.template.frame(SYSTEM_PROMPT, &block, &branches[0].suffix_text)?;
        let head_ids = self.tokenizer.encode(&head)?;
        let own: Vec<SuffixTokens> = match cache {
            Some(c) if c.tail == tail => c.own.clone(),
            _ => {
                let mut own = Vec::with_capacity(branches.len());
                let mut text = String::new();
                for b in &branches {
                    text.clear();
                    text.push_str(&b.suffix_text);
                    text.push_str(&tail);
                    own.push(self.tokenizer.encode_with_offsets(&text)?);
                }
                own
            }
        };

        // The marker render cuts the prompt at the first occurrence of the marker, which is the end of the head
        // unless the state itself contains the marker.
        let cut = head.find(PREFIX_MARK).unwrap_or(head.len());
        let mark_ids = if cut == head.len() { None } else { Some(self.tokenizer.encode(&head[..cut])?) };
        let n = match &mark_ids {
            None => head_ids.len(),
            Some(mark) => {
                // Longest common token prefix of every branch (head_ids ++ own ids) with the marker prefix.
                own.iter()
                    .map(|(ids, _)| {
                        let full = head_ids.iter().chain(ids.iter());
                        full.zip(mark.iter()).take_while(|(a, b)| a == b).count()
                    })
                    .min()
                    .unwrap_or(0)
            }
        };
        let prefix_ids = match mark_ids {
            None => head_ids.clone(),
            Some(mut m) => {
                m.truncate(n);
                m
            }
        };
        for (b, (ids, offs)) in branches.iter_mut().zip(own) {
            if n == head_ids.len() {
                b.suffix_ids = ids;
                b.offsets = Some(offs);
            } else {
                let full: Vec<u32> = head_ids.iter().chain(ids.iter()).copied().collect();
                b.suffix_ids = full[n..].to_vec();
            }
        }
        if branches.iter().any(|b| b.suffix_ids.is_empty()) {
            return Err(PlanError::Value("a branch has an empty suffix; state block boundary mismatch".into()));
        }

        let p = prefix_ids.len();
        let longest = p + branches.iter().map(|b| b.suffix_ids.len()).max().unwrap_or(0);
        let total = p + branches.iter().map(|b| b.suffix_ids.len()).sum::<usize>();
        let mut h = Sha256::new();
        h.update(head.as_bytes());
        h.update(branches[0].suffix_text.as_bytes());
        h.update(tail.as_bytes());
        let prompt_sha256 = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        Ok(Plan {
            branches,
            prefix_ids,
            head_tokens: head_ids.len(),
            longest,
            total,
            prompt_head: head,
            prompt_tail: tail,
            prompt_sha256,
            num_questions,
        })
    }

    /// The limit checks of `decide()` (branch limit first), then serve.patch's `_split` checks and option token
    /// spans (when the checkpoint's attention is noncausal).
    pub fn check(&self, plan: &mut Plan) -> Result<(), PlanError> {
        let l = &self.limits;
        if plan.longest > l.max_branch_tokens {
            return Err(PlanError::Value(format!(
                "state plus longest question is {} tokens, limit is {}",
                plan.longest, l.max_branch_tokens
            )));
        }
        if plan.total > l.max_request_tokens {
            return Err(PlanError::Value(format!("request is {} tokens, limit is {}", plan.total, l.max_request_tokens)));
        }
        if !self.noncausal {
            return Ok(());
        }
        let p = plan.prefix_ids.len();
        let head_chars = plan.prompt_head.chars().count();
        let ht = plan.head_tokens;
        for b in &mut plan.branches {
            // token_layout: a = bisect_right(ends, cs), b = bisect_left(starts, ce) over the full prompt. Every
            // head token ends at or before the suffix start, so they all count.
            let mut spans = Vec::with_capacity(b.option_char_spans.len());
            if let Some(offs) = &b.offsets {
                for &(cs, ce) in &b.option_char_spans {
                    let a = ht + offs.partition_point(|&(_, e)| e <= cs);
                    let z = ht + offs.partition_point(|&(s, _)| s < ce);
                    if z <= a {
                        return Err(PlanError::Value(format!(
                            "option span {}:{} covers no token",
                            head_chars + cs,
                            head_chars + ce
                        )));
                    }
                    spans.push((a, z));
                }
            }
            if ht != p {
                return Err(PlanError::Value(format!("prefix block ends at token {ht} but the shared cache at {p}")));
            }
            if spans.iter().any(|&(a, _)| a < p) {
                return Err(PlanError::Value("an option span starts inside the shared prefix".into()));
            }
            b.option_token_spans = spans.into_iter().map(|(a, z)| (a - p, z - p)).collect();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_are_balanced() {
        let sizes = |n| plan_chunks(n, 256).iter().map(|r| r.len()).collect::<Vec<_>>();
        assert_eq!(sizes(300), [150, 150]);
        assert_eq!(sizes(511), [171, 171, 169]);
        assert_eq!(sizes(255), [255]);
        let sizes26 = plan_chunks(60, 26).iter().map(|r| r.len()).collect::<Vec<_>>();
        assert_eq!(sizes26, [20, 20, 20]);
    }

    #[test]
    fn dict_set_keeps_position() {
        let mut m = vec![("a".to_string(), Json::Null), (NONE_OPTION.to_string(), Json::Null), ("b".to_string(), Json::Null)];
        dict_set(&mut m, NONE_OPTION, Json::str(NONE_DESCRIPTION));
        assert_eq!(m.len(), 3);
        assert_eq!(m[1].1, Json::str(NONE_DESCRIPTION));
    }
}
