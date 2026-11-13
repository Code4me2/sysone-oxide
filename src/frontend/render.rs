//! Rendering: a question to its suffix text and slots, a state to its block (`qev/prompt.py` at OneJev@81ce62f
//! plus serve.patch's option spans). Only the production style (`qev-labels-v2`, tagged layout) is implemented.

use crate::api::schema::{Entry, Question, QuestionKind};
use crate::frontend::pyjson::{Json, py_strip};

pub const PROMPT_VERSION: &str = "qev-labels-v2";

pub const SYSTEM_PROMPT: &str = "Apply the question to the state. Choose exactly one of the listed options. \
                                 Respond with only its uppercase letter, with no explanation or reasoning.";

pub const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Two-letter labels that are not one token in the Qwen tokenizers (`qev.prompt._NOT_ONE_TOKEN`).
pub const NOT_ONE_TOKEN: [&str; 25] = [
    "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ", "HJ", "IY",
    "JF", "JG", "JH", "JL", "JN", "JQ",
];

/// `len(SLOT_LABELS)`.
pub const NUM_SLOT_LABELS: usize = 256;

pub const NOUL_DEFAULT_INSTRUCTIONS: &str = "Is the statement true, or is the answer to the question yes?";
pub const NOUL_DEFAULT_TRUE: &str = "the statement is true / the answer is yes";
pub const NOUL_DEFAULT_FALSE: &str = "the statement is false / the answer is no";
const CHOICE_DEFAULT_INSTRUCTIONS: &str = "Which option applies to the state?";
const SCORE_DEFAULT_INSTRUCTIONS: &str = "Which level describes the state?";

/// `qev.prompt.SLOT_LABELS`: A–Z, then two-letter labels in row-major order minus [`NOT_ONE_TOKEN`], 256 in all.
pub fn slot_labels() -> &'static [String] {
    static LABELS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LABELS.get_or_init(|| {
        let mut v: Vec<String> = LETTERS.chars().map(String::from).collect();
        'outer: for a in LETTERS.chars() {
            for b in LETTERS.chars() {
                let l: String = [a, b].iter().collect();
                if !NOT_ONE_TOKEN.contains(&l.as_str()) {
                    v.push(l);
                }
                if v.len() == NUM_SLOT_LABELS {
                    break 'outer;
                }
            }
        }
        v
    })
}

/// `render_entry`: strings are stripped, null is empty, objects and arrays are `json.dumps(indent=2)`.
pub fn render_entry(e: &Entry) -> String {
    match e {
        Json::Null => String::new(),
        Json::Str(s) => py_strip(s).to_string(),
        other => other.dumps_indent2(),
    }
}

/// `render_state`: strings are kept as they are (not stripped), null is empty, anything else is
/// `json.dumps(indent=2)`.
pub fn render_state(state: &Json) -> String {
    match state {
        Json::Null => String::new(),
        Json::Str(s) => s.clone(),
        other => other.dumps_indent2(),
    }
}

/// `state_block`: the end of the shared prefix.
pub fn state_block(state: &Json) -> String {
    let body = render_state(state);
    let mut s = String::with_capacity(body.len() + 20);
    s.push_str("<state>\n");
    s.push_str(&body);
    s.push_str("\n</state>\n\n");
    s
}

/// One branch: the suffix text and the labels its slots map to (`RenderedQuestion`).
#[derive(Clone, Debug, PartialEq)]
pub struct RenderedQuestion {
    pub kind: QuestionKind,
    /// The answer label of each slot (option names, `yes`/`no`, or level indices).
    pub labels: Vec<String>,
    pub descriptions: Vec<Entry>,
    pub suffix: String,
    /// serve.patch: each option line's character span `[start, end)` inside `suffix`, in Unicode code points.
    pub option_spans: Vec<(usize, usize)>,
}

impl RenderedQuestion {
    pub fn n_slots(&self) -> usize {
        self.labels.len()
    }
}

/// The unpermuted (labels, names, descriptions, instructions) of a question.
pub struct QuestionParts<'a> {
    pub kind: QuestionKind,
    pub instructions: String,
    pub labels: Vec<String>,
    pub names: Vec<String>,
    pub descriptions: Vec<std::borrow::Cow<'a, Entry>>,
}

pub fn question_parts(q: &Question) -> QuestionParts<'_> {
    use std::borrow::Cow;
    let instructions = render_entry(q.instructions());
    match q {
        Question::Noul { criteria, .. } => {
            let pick = |e: Option<&Entry>, default: &str| match e {
                Some(e) if !e.is_null() => Json::Str(render_entry(e)),
                _ => Json::str(default),
            };
            let t = pick(criteria.as_ref().map(|c| &c.true_), NOUL_DEFAULT_TRUE);
            let f = pick(criteria.as_ref().map(|c| &c.false_), NOUL_DEFAULT_FALSE);
            QuestionParts {
                kind: QuestionKind::Noul,
                instructions: if instructions.is_empty() { NOUL_DEFAULT_INSTRUCTIONS.into() } else { instructions },
                labels: vec!["yes".into(), "no".into()],
                names: vec!["yes".into(), "no".into()],
                descriptions: vec![Cow::Owned(t), Cow::Owned(f)],
            }
        }
        Question::Choice { criteria, .. } => QuestionParts {
            kind: QuestionKind::Choice,
            instructions: if instructions.is_empty() { CHOICE_DEFAULT_INSTRUCTIONS.into() } else { instructions },
            labels: criteria.iter().map(|(k, _)| k.clone()).collect(),
            names: criteria.iter().map(|(k, _)| k.clone()).collect(),
            descriptions: criteria.iter().map(|(_, v)| Cow::Borrowed(v)).collect(),
        },
        Question::Score { criteria, .. } => QuestionParts {
            kind: QuestionKind::Score,
            instructions: if instructions.is_empty() { SCORE_DEFAULT_INSTRUCTIONS.into() } else { instructions },
            labels: (0..criteria.len()).map(|i| i.to_string()).collect(),
            names: (0..criteria.len()).map(|i| format!("level {i}")).collect(),
            descriptions: criteria.iter().map(Cow::Borrowed).collect(),
        },
    }
}

/// `render_question(qid, q, order)`: `order[i]` is the original index of slot i (None = identity).
pub fn render_question(parts: &QuestionParts<'_>, order: Option<&[usize]>) -> Result<RenderedQuestion, String> {
    let k = parts.labels.len();
    if k > NUM_SLOT_LABELS {
        return Err(format!("{k} options exceed the {NUM_SLOT_LABELS} slots of prompt {PROMPT_VERSION}"));
    }
    let idx = |i: usize| order.map_or(i, |o| o[i]);
    let header = if parts.kind == QuestionKind::Score { "Rate the state:" } else { "Question:" };
    let letters = &slot_labels()[..k];
    let mut suffix = String::with_capacity(64 + 48 * k);
    suffix.push_str(header);
    suffix.push(' ');
    suffix.push_str(&parts.instructions);
    suffix.push_str("\n\nOptions:\n");
    let mut pos = suffix.chars().count();
    let mut option_spans = Vec::with_capacity(k);
    for (i, letter) in letters.iter().enumerate() {
        let j = idx(i);
        let text = render_entry(&parts.descriptions[j]);
        if i > 0 {
            suffix.push('\n');
            pos += 1;
        }
        let start_byte = suffix.len();
        suffix.push_str(letter);
        suffix.push_str(". ");
        suffix.push_str(&parts.names[j]);
        if !text.is_empty() {
            suffix.push_str(": ");
            suffix.push_str(&text);
        }
        let len = suffix[start_byte..].chars().count();
        option_spans.push((pos, pos + len));
        pos += len;
    }
    suffix.push_str("\n\n");
    suffix.push_str(if k <= LETTERS.len() { "Answer with one letter: " } else { "Answer with one label: " });
    suffix.push_str(&letters.join(", "));
    suffix.push('.');
    Ok(RenderedQuestion {
        kind: parts.kind,
        labels: (0..k).map(|i| parts.labels[idx(i)].clone()).collect(),
        descriptions: (0..k).map(|i| parts.descriptions[idx(i)].as_ref().clone()).collect(),
        suffix,
        option_spans,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_labels_match_qev() {
        let l = slot_labels();
        assert_eq!(l.len(), 256);
        assert_eq!(l[25], "Z");
        assert_eq!(l[26], "AA");
        assert!(!l.iter().any(|x| x == "BQ" || x == "JQ"));
        assert_eq!(l[255], "JU");
    }

    #[test]
    fn noul_defaults_and_permutation() {
        let q = Question::Noul { instructions: Json::str("  Is it?  "), criteria: None };
        let parts = question_parts(&q);
        let r = render_question(&parts, Some(&[1, 0])).unwrap();
        assert_eq!(
            r.suffix,
            "Question: Is it?\n\nOptions:\nA. no: the statement is false / the answer is no\n\
             B. yes: the statement is true / the answer is yes\n\nAnswer with one letter: A, B."
        );
        assert_eq!(r.labels, ["no", "yes"]);
        let a = &r.suffix[r.suffix.find("A.").unwrap()..];
        assert_eq!(r.option_spans[0].1 - r.option_spans[0].0, a.find('\n').unwrap());
    }

    #[test]
    fn score_entries_render_as_json() {
        let q = Question::Score {
            instructions: Json::Null,
            criteria: vec![Json::Null, Json::Array(vec![Json::int(1), Json::Float(0.5)])],
        };
        let r = render_question(&question_parts(&q), None).unwrap();
        assert_eq!(
            r.suffix,
            "Rate the state: Which level describes the state?\n\nOptions:\nA. level 0\nB. level 1: [\n  1,\n  0.5\n]\n\n\
             Answer with one letter: A, B."
        );
    }
}
