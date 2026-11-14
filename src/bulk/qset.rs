//! Question-set runs: one set of questions evaluated over many documents (`sysone bulk --questions Q.json --docs
//! D.jsonl|DIR`, docs/BULK.md).
//!
//! The question file is a `/v1/systemone` request body without `state`: `{"questions": {...}}` plus any other
//! top-level request fields that apply to every document (`model`; keys the server ignores are kept and ignored).
//! Each document supplies `state`, optionally its own `query` and `overrides` (top-level fields that replace the
//! set's). A document's result is, by definition, the server's answer to its **equivalent full body**
//! ([`QuestionSet::body`]): `{"state": <state>}` followed by the question file's fields, then the overrides,
//! where a later key replaces an earlier one in place (Python dict semantics).
//!
//! What is saved is CPU planning only: the question set is validated once, before any worker starts, and its
//! branches are rendered and their suffix tokens computed once per `debias` value
//! ([`Frontend::prepare_questions`]), so a document costs the rendering of its own prompt head and the
//! tokenization of its state. The GPU work per document is unchanged: every question attends to its own
//! document's prefix, so no question computation can be shared across documents.
//!
//! The reuse is exact: validation does not look at `state` (any JSON value is a valid state), so a document that
//! does not override `model` or `questions` validates exactly as the set did; and the suffix tokens are a function
//! of the suffix text and the template's tail alone, which [`Frontend::plan_prepared`] re-checks per document. A
//! document that overrides `model` or `questions` goes through the server's whole path on its equivalent body.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::job::Doc;
use crate::api::schema::{Question, SystemOneRequest, check_model};
use crate::api::server::{self, Outcome, Params, Ready, ServerInfo};
use crate::frontend::pyjson::{self, Json};
use crate::frontend::{Frontend, PlanError, PreparedQuestions};

/// Top-level request fields the server reads besides `state`: a document overriding one of them is planned from its
/// full body.
const READ_FIELDS: [&str; 2] = ["model", "questions"];

pub struct QuestionSet {
    /// The question file's fields, in file order (no `state`).
    fields: Vec<(String, Json)>,
    model: String,
    questions: Vec<(String, Question)>,
    prepared: Mutex<HashMap<i64, Arc<PreparedQuestions>>>,
    /// Documents planned from the prepared set, and from their full body (overrides).
    pub(crate) docs_prepared: AtomicU64,
    pub(crate) docs_full_body: AtomicU64,
    /// For the summary: questions, branches and suffix tokens (longest, all) at the default `debias`.
    pub num_questions: usize,
    pub num_branches: usize,
    pub suffix_tokens: (usize, usize),
}

/// Insert with Python dict semantics: an existing key keeps its position and takes the new value.
fn dict_set(m: &mut Vec<(String, Json)>, k: String, v: Json) {
    match m.iter_mut().find(|(key, _)| *key == k) {
        Some(slot) => slot.1 = v,
        None => m.push((k, v)),
    }
}

impl QuestionSet {
    /// Read and validate a question file (see [`QuestionSet::from_json`]).
    pub fn load(path: &Path, fe: &Frontend, info: &ServerInfo, default_query: Option<&str>) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading the question set {}", path.display()))?;
        let v = pyjson::parse(&bytes).map_err(|e| anyhow::anyhow!("question set {}: not valid JSON ({e:?})", path.display()))?;
        Self::from_json(v, fe, info, default_query).with_context(|| format!("question set {}", path.display()))
    }

    /// Validate a question set as the server would validate every document's body, and prepare it for the
    /// default `debias` (`default_query`, else the server's default). Every error here is one every document
    /// would get (or a refusal of the input), reported before any worker starts.
    pub fn from_json(v: Json, fe: &Frontend, info: &ServerInfo, default_query: Option<&str>) -> Result<Self> {
        let Json::Object(fields) = v else { bail!("must be a JSON object: a request body without \"state\"") };
        if fields.iter().any(|(k, _)| k == "state") {
            bail!("must not hold \"state\": each document supplies its own");
        }
        if !fields.iter().any(|(k, _)| k == "questions") {
            bail!("has no \"questions\" object");
        }
        let mut probe = vec![("state".to_string(), Json::Null)];
        probe.extend(fields.iter().cloned());
        let req = SystemOneRequest::validate(&Json::Object(probe))
            .map_err(|e| anyhow::anyhow!("invalid (the server would answer every document {}): {}", e.status(), e.body().dumps_compact().unwrap_or_default()))?;
        check_model(&req.model, &info.served_name).map_err(|e| anyhow::anyhow!("{e}"))?;
        let p = server::params(default_query, info.default_debias);
        let Some(debias) = p.debias() else { bail!("the default query {default_query:?} has a debias that is not an integer") };
        let pq = fe.prepare_questions(&req.questions, debias).map_err(|e| anyhow::anyhow!("cannot be planned: {e}"))?;
        // Every prompt is head + suffix with a head of at least one token, so a suffix this long fails every
        // document's limits.
        let (longest, all) = pq.suffix_tokens();
        let l = &fe.limits;
        if longest >= l.max_branch_tokens || all >= l.max_request_tokens {
            bail!(
                "its longest question is {longest} tokens and all its questions {all} (with the chat template's tail): \
                 no document fits the limits ({} per branch, {} per request)",
                l.max_branch_tokens,
                l.max_request_tokens
            );
        }
        let qs = QuestionSet {
            fields,
            model: req.model,
            num_questions: req.questions.len(),
            num_branches: pq.num_branches(),
            suffix_tokens: (longest, all),
            questions: req.questions,
            prepared: Mutex::new(HashMap::from([(debias, Arc::new(pq))])),
            docs_prepared: AtomicU64::new(0),
            docs_full_body: AtomicU64::new(0),
        };
        Ok(qs)
    }

    /// The equivalent full request body of a document: `{"state": ...}`, then the question file's fields, then the
    /// document's overrides (a repeated key keeps its first position and takes the last value).
    pub fn body(&self, doc: &Doc) -> Json {
        let mut m = vec![("state".to_string(), doc.state.clone())];
        for (k, v) in self.fields.iter().chain(&doc.overrides) {
            dict_set(&mut m, k.clone(), v.clone());
        }
        Json::Object(m)
    }

    fn prepared(&self, fe: &Frontend, debias: i64) -> Result<Arc<PreparedQuestions>, PlanError> {
        if let Some(p) = self.prepared.lock().unwrap().get(&debias) {
            return Ok(p.clone());
        }
        let p = Arc::new(fe.prepare_questions(&self.questions, debias)?);
        Ok(self.prepared.lock().unwrap().entry(debias).or_insert(p).clone())
    }

    /// [`server::prepare`] of the document's equivalent body, with the question set's validation and rendering
    /// reused.
    pub(crate) fn prepare(&self, fe: &Frontend, info: &ServerInfo, doc: Doc, p: &Params) -> Result<Ready, Outcome> {
        if doc.overrides.iter().any(|(k, _)| READ_FIELDS.contains(&k.as_str())) {
            self.docs_full_body.fetch_add(1, Ordering::Relaxed);
            return server::prepare(fe, info, Ok(self.body(&doc)), p);
        }
        self.docs_prepared.fetch_add(1, Ordering::Relaxed);
        // server::prepare's order: validation and the model (both passed for the set, and independent of the
        // state), the debias, then planning.
        let started = Instant::now();
        let Some(debias) = p.debias() else { return Err(Outcome::Internal) };
        let pq = self.prepared(fe, debias).map_err(server::plan_error)?;
        let plan = fe.plan_prepared(&doc.state, &pq).map_err(server::plan_error)?;
        let req = SystemOneRequest { state: doc.state, model: self.model.clone(), questions: self.questions.clone() };
        Ok(Ready { req, plan: Arc::new(plan), debug: p.debug(), started, planned: Instant::now() })
    }
}
