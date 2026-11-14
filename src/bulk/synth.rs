//! Synthetic `/v1/systemone` requests: the load generator's bodies (`loadgen`), and corpora of them with a length
//! mix for bulk runs (`sysone bulk gen`).
//!
//! A state is `words` pseudo-random support-ticket words (about one token each with elephant-1's tokenizer);
//! questions cycle noul / choice with 5 options / score with 4 levels.

const VOCAB: [&str; 16] = [
    "order", "refund", "late", "billing", "error", "customer", "angry", "shipment", "account", "card", "retry", "ok",
    "package", "warehouse", "address", "invoice",
];

/// `n` pseudo-random words of document `i`, space-separated.
pub fn words(i: usize, n: usize) -> String {
    let mut x = (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    let words: Vec<&str> = (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            VOCAB[(x % VOCAB.len() as u64) as usize]
        })
        .collect();
    words.join(" ")
}

/// The questions object (JSON text) of `n` questions; `case`: the case number their instructions name (None: the
/// same text for every document, as in a question set).
pub fn questions(case: Option<usize>, n: usize) -> String {
    let c = case.map(|i| format!(" {i}")).unwrap_or_default();
    let qs: Vec<String> = (0..n)
        .map(|q| match q % 3 {
            0 => format!(r#""q{q}": {{"type": "noul", "instructions": "Should case{c} be escalated (check {q})?"}}"#),
            1 => format!(
                r#""q{q}": {{"type": "choice", "instructions": "Route case{c}", "criteria": {{"billing": "Billing team", "tech": "Technical support", "ship": "Shipping", "fraud": "Fraud review", "other": null}}}}"#
            ),
            _ => format!(
                r#""q{q}": {{"type": "score", "instructions": "Urgency of case{c}", "criteria": ["low", "medium", "high", "critical"]}}"#
            ),
        })
        .collect();
    format!("{{{}}}", qs.join(", "))
}

/// Request body `i` with a state of `n_words` words and `n_questions` questions (loadgen's body for
/// `--state-tokens n_words --questions n_questions`).
pub fn body(i: usize, n_words: usize, n_questions: usize) -> String {
    format!(r#"{{"state": {{"ticket": "{}", "case": {i}}}, "questions": {}}}"#, words(i, n_words), questions(Some(i), n_questions))
}

/// Parameters of a synthetic bulk corpus.
#[derive(Clone, Debug)]
pub struct CorpusSpec {
    pub docs: usize,
    /// State length range in words (log-uniform: as many short as long documents per factor of length).
    pub min_words: usize,
    pub max_words: usize,
    /// Questions per document, uniform.
    pub min_questions: usize,
    pub max_questions: usize,
    pub seed: u64,
}

/// Document `i` of a corpus: its state words and question count.
pub fn doc_shape(spec: &CorpusSpec, i: usize) -> (usize, usize) {
    let mut x = (spec.seed ^ (i as u64).wrapping_mul(0xd1b54a32d192ed03)).wrapping_add(0x9e3779b97f4a7c15);
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let _ = next();
    let u = (next() >> 11) as f64 / (1u64 << 53) as f64;
    let (lo, hi) = (spec.min_words.max(1) as f64, spec.max_words.max(spec.min_words.max(1)) as f64);
    let words = (lo * (hi / lo).powf(u)).round() as usize;
    let span = spec.max_questions.saturating_sub(spec.min_questions) + 1;
    let q = spec.min_questions + (next() % span as u64) as usize;
    (words, q.max(1))
}

/// The corpus as bulk job lines (`{"id": "doc-000001", "body": {...}}`), one per document.
pub fn corpus_line(spec: &CorpusSpec, i: usize) -> String {
    let (words, q) = doc_shape(spec, i);
    format!("{{\"id\": \"doc-{i:06}\", \"body\": {}}}", body(i, words, q))
}

/// A question set of `n` questions (`sysone bulk gen --question-set`): `{"questions": {...}}`.
pub fn question_set(n: usize) -> String {
    format!(r#"{{"questions": {}}}"#, questions(None, n))
}

/// Document `i` of a question-set corpus: `{"id": "doc-000001", "state": "<words>"}` (the state is text).
pub fn doc_line(spec: &CorpusSpec, i: usize) -> String {
    let (w, _) = doc_shape(spec, i);
    format!(r#"{{"id": "doc-{i:06}", "state": "{}"}}"#, words(i, w))
}

/// The job line equivalent to [`doc_line`] with [`question_set`]`(n)`: `{"state": ..., "questions": ...}`, the
/// order `QuestionSet::body` gives.
pub fn doc_job_line(spec: &CorpusSpec, i: usize, n: usize) -> String {
    let (w, _) = doc_shape(spec, i);
    format!(r#"{{"id": "doc-{i:06}", "body": {{"state": "{}", "questions": {}}}}}"#, words(i, w), questions(None, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_deterministic_and_spans_the_range() {
        let spec = CorpusSpec { docs: 2000, min_words: 300, max_words: 8000, min_questions: 4, max_questions: 16, seed: 1 };
        let shapes: Vec<(usize, usize)> = (0..spec.docs).map(|i| doc_shape(&spec, i)).collect();
        assert_eq!(shapes, (0..spec.docs).map(|i| doc_shape(&spec, i)).collect::<Vec<_>>());
        let w: Vec<usize> = shapes.iter().map(|s| s.0).collect();
        assert!(*w.iter().min().unwrap() >= 300 && *w.iter().max().unwrap() <= 8000);
        // Log-uniform: about a third below ~900 words (e^(ln 26.7 / 3) x 300 = 897).
        let short = w.iter().filter(|&&x| x < 897).count() as f64 / w.len() as f64;
        assert!((0.28..0.39).contains(&short), "{short}");
        let q: Vec<usize> = shapes.iter().map(|s| s.1).collect();
        assert!(q.iter().all(|&x| (4..=16).contains(&x)) && q.contains(&4) && q.contains(&16));
        let line = corpus_line(&spec, 3);
        assert!(line.starts_with("{\"id\": \"doc-000003\", \"body\": {\"state\""));
        crate::frontend::pyjson::parse(line.as_bytes()).unwrap();
    }
}
