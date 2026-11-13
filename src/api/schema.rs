//! Request and response types of `/v1/systemone`, equivalent to qev's pydantic schema (`qev/schema.py` at
//! OneJev@81ce62f) and to its server's error bodies (`qev/server.py`).
//!
//! Validation is hand-written rather than derived, because the 422 body must be pydantic's: the same `loc`, `msg`
//! and `type` for every error, in pydantic's order, including the smart-union member tags
//! (`NoulQuestion`, `function-after[_check(), ChoiceQuestion]`, `str`, `dict[str,any]`, ...). Like the pydantic
//! models, everything is lenient: extra keys are ignored everywhere, entries may be strings, objects, arrays or null.
//! Bodies are parsed with Python's `json.loads` semantics ([`pyjson`]), so object key order (which sets option and
//! question order) and number formatting survive.

use crate::frontend::pyjson::{self, Json, py_repr_str};

/// `qev.schema.MAX_CHOICE_OPTIONS`.
pub const MAX_CHOICE_OPTIONS: usize = 255;
/// `qev.schema.MAX_SCORE_LEVELS`.
pub const MAX_SCORE_LEVELS: usize = 10;
/// `qev.server.ALIASES`, sorted.
pub const ALIASES: [&str; 3] = ["jev-latest", "jev-preview", "qev-latest"];
/// The default `model` of a request.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// An instructions or criteria entry: a string, an object, an array, or null (`qev.schema.Entry`). Held as the
/// parsed JSON value; validation guarantees it is one of those four.
pub type Entry = Json;

#[derive(Clone, Debug, PartialEq)]
pub struct NoulCriteria {
    pub true_: Entry,
    pub false_: Entry,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Question {
    Noul { instructions: Entry, criteria: Option<NoulCriteria> },
    /// `criteria` in request order (option names are the keys).
    Choice { instructions: Entry, criteria: Vec<(String, Entry)> },
    Score { instructions: Entry, criteria: Vec<Entry> },
}

impl Question {
    pub fn kind(&self) -> QuestionKind {
        match self {
            Question::Noul { .. } => QuestionKind::Noul,
            Question::Choice { .. } => QuestionKind::Choice,
            Question::Score { .. } => QuestionKind::Score,
        }
    }

    pub fn instructions(&self) -> &Entry {
        match self {
            Question::Noul { instructions, .. }
            | Question::Choice { instructions, .. }
            | Question::Score { instructions, .. } => instructions,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuestionKind {
    Noul,
    Choice,
    Score,
}

impl QuestionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            QuestionKind::Noul => "noul",
            QuestionKind::Choice => "choice",
            QuestionKind::Score => "score",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SystemOneRequest {
    pub state: Json,
    pub model: String,
    /// In request order.
    pub questions: Vec<(String, Question)>,
}

/// One entry of a 422 `detail` list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorItem {
    pub loc: Vec<String>,
    pub msg: String,
    pub kind: String,
}

impl ErrorItem {
    fn new(loc: &[String], msg: impl Into<String>, kind: &str) -> Self {
        ErrorItem { loc: loc.to_vec(), msg: msg.into(), kind: kind.into() }
    }

    fn to_json(&self) -> Json {
        Json::Object(vec![
            ("loc".into(), Json::Array(self.loc.iter().map(|s| Json::str(s.as_str())).collect())),
            ("msg".into(), Json::str(self.msg.as_str())),
            ("type".into(), Json::str(self.kind.as_str())),
        ])
    }
}

/// An error response of qev's server, before the engine runs (or a `ValueError` raised by the engine).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiError {
    /// 422 `{"detail": [...]}`.
    Unprocessable(Vec<ErrorItem>),
    /// 404 `{"detail": "<message>"}`.
    NotFound(String),
}

impl ApiError {
    /// The 422 for a body `json.loads` refuses.
    pub fn invalid_json() -> Self {
        ApiError::Unprocessable(vec![ErrorItem::new(&["body".into()], "invalid JSON", "json_invalid")])
    }

    /// The 422 qev returns when `decide()` raises `ValueError(msg)` (limits, prefix split).
    pub fn value_error(msg: impl Into<String>) -> Self {
        ApiError::Unprocessable(vec![ErrorItem::new(&["body".into()], msg, "value_error")])
    }

    pub fn status(&self) -> u16 {
        match self {
            ApiError::Unprocessable(_) => 422,
            ApiError::NotFound(_) => 404,
        }
    }

    pub fn body(&self) -> Json {
        let detail = match self {
            ApiError::Unprocessable(items) => Json::Array(items.iter().map(ErrorItem::to_json).collect()),
            ApiError::NotFound(msg) => Json::str(msg.as_str()),
        };
        Json::Object(vec![("detail".into(), detail)])
    }

    /// The response body as Starlette renders it.
    pub fn body_bytes(&self) -> Vec<u8> {
        self.body().dumps_compact().expect("error bodies hold no floats").into_bytes()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Unprocessable(items) => {
                let msgs: Vec<_> = items.iter().map(|e| format!("{}: {}", e.loc.join("."), e.msg)).collect();
                write!(f, "422 {}", msgs.join("; "))
            }
            ApiError::NotFound(msg) => write!(f, "404 {msg}"),
        }
    }
}

impl std::error::Error for ApiError {}

/// Model routing (`qev.server`): the request's model must be an alias or the served name. An empty model means
/// the default alias.
pub fn check_model(requested: &str, served: &str) -> Result<(), ApiError> {
    let requested = if requested.is_empty() { DEFAULT_MODEL } else { requested };
    if ALIASES.contains(&requested) || requested == served {
        return Ok(());
    }
    let aliases: Vec<String> = ALIASES.iter().map(|a| py_repr_str(a)).collect();
    Err(ApiError::NotFound(format!(
        "model {} is not served; available: {served} plus aliases [{}]",
        py_repr_str(requested),
        aliases.join(", ")
    )))
}

impl SystemOneRequest {
    /// Parse and validate a POST body: `json.loads` then `SystemOneRequest.model_validate`.
    pub fn from_body(body: &[u8]) -> Result<Self, ApiError> {
        let v = pyjson::parse(body).map_err(|_| ApiError::invalid_json())?;
        Self::validate(&v)
    }

    /// `SystemOneRequest.model_validate(value)`, with pydantic's errors.
    pub fn validate(v: &Json) -> Result<Self, ApiError> {
        let mut errs = Vec::new();
        let root = vec!["body".to_string()];
        let Some(obj) = v.as_object() else {
            errs.push(ErrorItem::new(&root, "Input should be a valid dictionary or instance of SystemOneRequest", "model_type"));
            return Err(ApiError::Unprocessable(errs));
        };
        let field = |k: &str| obj.iter().find(|(key, _)| key == k).map(|(_, v)| v);
        let state = field("state").cloned().unwrap_or(Json::Null);
        let model = match field("model") {
            None => Some(DEFAULT_MODEL.to_string()),
            Some(Json::Str(s)) => Some(s.clone()),
            Some(_) => {
                errs.push(ErrorItem::new(&path(&root, "model"), "Input should be a valid string", "string_type"));
                None
            }
        };
        let qloc = path(&root, "questions");
        let questions = match field("questions") {
            None => {
                errs.push(ErrorItem::new(&qloc, "Field required", "missing"));
                None
            }
            Some(Json::Object(qs)) => {
                let mut out = Vec::with_capacity(qs.len());
                let mut ok = true;
                for (qid, q) in qs {
                    match validate_question(q, &path(&qloc, qid), &mut errs) {
                        Some(q) => out.push((qid.clone(), q)),
                        None => ok = false,
                    }
                }
                ok.then_some(out)
            }
            Some(_) => {
                errs.push(ErrorItem::new(&qloc, "Input should be a valid dictionary", "dict_type"));
                None
            }
        };
        if !errs.is_empty() {
            return Err(ApiError::Unprocessable(errs));
        }
        let (model, questions) = (model.unwrap(), questions.unwrap());
        if questions.is_empty() {
            return Err(ApiError::Unprocessable(vec![ErrorItem::new(
                &root,
                "Value error, at least one question is required",
                "value_error",
            )]));
        }
        Ok(SystemOneRequest { state, model, questions })
    }
}

fn path(loc: &[String], seg: &str) -> Vec<String> {
    let mut p = loc.to_vec();
    p.push(seg.to_string());
    p
}

const NOUL_TAG: &str = "NoulQuestion";
const CHOICE_TAG: &str = "function-after[_check(), ChoiceQuestion]";
const SCORE_TAG: &str = "function-after[_check(), ScoreQuestion]";

/// The `Question` smart union: the member whose `type` matches wins; if none validates, pydantic reports every
/// member's errors in declaration order.
fn validate_question(q: &Json, loc: &[String], errs: &mut Vec<ErrorItem>) -> Option<Question> {
    let mut member_errs = Vec::new();
    for kind in [QuestionKind::Noul, QuestionKind::Choice, QuestionKind::Score] {
        let mut e = Vec::new();
        if let Some(q) = validate_member(q, kind, loc, &mut e) {
            return Some(q);
        }
        member_errs.extend(e);
    }
    errs.extend(member_errs);
    None
}

fn validate_member(q: &Json, kind: QuestionKind, loc: &[String], errs: &mut Vec<ErrorItem>) -> Option<Question> {
    let (tag, model) = match kind {
        QuestionKind::Noul => (NOUL_TAG, "NoulQuestion"),
        QuestionKind::Choice => (CHOICE_TAG, "ChoiceQuestion"),
        QuestionKind::Score => (SCORE_TAG, "ScoreQuestion"),
    };
    let loc = path(loc, tag);
    let Some(obj) = q.as_object() else {
        errs.push(ErrorItem::new(&loc, format!("Input should be a valid dictionary or instance of {model}"), "model_type"));
        return None;
    };
    let field = |k: &str| obj.iter().find(|(key, _)| key == k).map(|(_, v)| v);
    let before = errs.len();
    match field("type") {
        None => errs.push(ErrorItem::new(&path(&loc, "type"), "Field required", "missing")),
        Some(Json::Str(s)) if s == kind.as_str() => {}
        Some(_) => errs.push(ErrorItem::new(&path(&loc, "type"), format!("Input should be '{}'", kind.as_str()), "literal_error")),
    }
    let instructions = match field("instructions") {
        None => Json::Null,
        Some(v) => validate_entry(v, &path(&loc, "instructions"), errs).unwrap_or(Json::Null),
    };
    let cloc = path(&loc, "criteria");
    let question = match kind {
        QuestionKind::Noul => {
            let criteria = match field("criteria") {
                None | Some(Json::Null) => None,
                Some(Json::Object(c)) => {
                    let get = |k: &str| c.iter().find(|(key, _)| key == k).map(|(_, v)| v);
                    let mut entry = |k: &str| match get(k) {
                        None => Json::Null,
                        Some(v) => validate_entry(v, &path(&cloc, k), errs).unwrap_or(Json::Null),
                    };
                    let true_ = entry("true");
                    let false_ = entry("false");
                    Some(NoulCriteria { true_, false_ })
                }
                Some(_) => {
                    errs.push(ErrorItem::new(&cloc, "Input should be a valid dictionary or instance of NoulCriteria", "model_type"));
                    None
                }
            };
            Question::Noul { instructions, criteria }
        }
        QuestionKind::Choice => {
            let criteria = match field("criteria") {
                None => {
                    errs.push(ErrorItem::new(&cloc, "Field required", "missing"));
                    Vec::new()
                }
                Some(Json::Object(c)) => c
                    .iter()
                    .map(|(k, v)| (k.clone(), validate_entry(v, &path(&cloc, k), errs).unwrap_or(Json::Null)))
                    .collect(),
                Some(_) => {
                    errs.push(ErrorItem::new(&cloc, "Input should be a valid dictionary", "dict_type"));
                    Vec::new()
                }
            };
            Question::Choice { instructions, criteria }
        }
        QuestionKind::Score => {
            let criteria = match field("criteria") {
                None => {
                    errs.push(ErrorItem::new(&cloc, "Field required", "missing"));
                    Vec::new()
                }
                Some(Json::Array(c)) => c
                    .iter()
                    .enumerate()
                    .map(|(i, v)| validate_entry(v, &path(&cloc, &i.to_string()), errs).unwrap_or(Json::Null))
                    .collect(),
                Some(_) => {
                    errs.push(ErrorItem::new(&cloc, "Input should be a valid list", "list_type"));
                    Vec::new()
                }
            };
            Question::Score { instructions, criteria }
        }
    };
    if errs.len() > before {
        return None;
    }
    // The `_check` after-validators run once the fields are valid.
    let check = match &question {
        Question::Choice { criteria, .. } if criteria.is_empty() => {
            Some("choice question needs at least one option in criteria".to_string())
        }
        Question::Choice { criteria, .. } if criteria.len() > MAX_CHOICE_OPTIONS => {
            Some(format!("choice question has {} options, maximum is {MAX_CHOICE_OPTIONS}", criteria.len()))
        }
        Question::Score { criteria, .. } if criteria.is_empty() => {
            Some("score question needs at least one level in criteria".to_string())
        }
        Question::Score { criteria, .. } if criteria.len() > MAX_SCORE_LEVELS => {
            Some(format!("score question has {} levels, maximum is {MAX_SCORE_LEVELS}", criteria.len()))
        }
        _ => None,
    };
    if let Some(msg) = check {
        errs.push(ErrorItem::new(&loc, format!("Value error, {msg}"), "value_error"));
        return None;
    }
    Some(question)
}

/// `Entry = Union[str, dict[str, Any], list[Any], None]`.
fn validate_entry(v: &Json, loc: &[String], errs: &mut Vec<ErrorItem>) -> Option<Entry> {
    match v {
        Json::Null | Json::Str(_) | Json::Object(_) | Json::Array(_) => Some(v.clone()),
        _ => {
            errs.push(ErrorItem::new(&path(loc, "str"), "Input should be a valid string", "string_type"));
            errs.push(ErrorItem::new(&path(loc, "dict[str,any]"), "Input should be a valid dictionary", "dict_type"));
            errs.push(ErrorItem::new(&path(loc, "list[any]"), "Input should be a valid list", "list_type"));
            None
        }
    }
}

/// `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`. Maps keep their order.
#[derive(Clone, Debug, PartialEq)]
pub enum Answer {
    Noul { noul: f64 },
    Choice { choice: String, probabilities: Vec<(String, f64)>, confidence: f64 },
    Score { score: f64, legend: Vec<(String, Entry)>, probabilities: Vec<(String, f64)>, confidence: f64 },
}

fn prob_map(p: &[(String, f64)]) -> Json {
    Json::Object(p.iter().map(|(k, v)| (k.clone(), Json::Float(*v))).collect())
}

impl Answer {
    /// The answer as pydantic's `model_dump` lays it out (`type` first).
    pub fn to_json(&self) -> Json {
        let mut m = Vec::with_capacity(5);
        match self {
            Answer::Noul { noul } => {
                m.push(("type".into(), Json::str("noul")));
                m.push(("noul".into(), Json::Float(*noul)));
            }
            Answer::Choice { choice, probabilities, confidence } => {
                m.push(("type".into(), Json::str("choice")));
                m.push(("choice".into(), Json::str(choice.as_str())));
                m.push(("probabilities".into(), prob_map(probabilities)));
                m.push(("confidence".into(), Json::Float(*confidence)));
            }
            Answer::Score { score, legend, probabilities, confidence } => {
                m.push(("type".into(), Json::str("score")));
                m.push(("score".into(), Json::Float(*score)));
                m.push(("legend".into(), Json::Object(legend.clone())));
                m.push(("probabilities".into(), prob_map(probabilities)));
                m.push(("confidence".into(), Json::Float(*confidence)));
            }
        }
        Json::Object(m)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

/// The `qev` block qev's server adds to every 200 response (without `timing`, which the server adds).
#[derive(Clone, Debug, PartialEq)]
pub struct QevMeta {
    pub prompt_version: String,
    pub prompt_sha256: String,
    pub fork_mode: String,
    pub prefix_tokens: u64,
    pub branches: u64,
}

impl QevMeta {
    pub fn to_json(&self) -> Json {
        Json::Object(vec![
            ("prompt_version".into(), Json::str(self.prompt_version.as_str())),
            ("prompt_sha256".into(), Json::str(self.prompt_sha256.as_str())),
            ("fork_mode".into(), Json::str(self.fork_mode.as_str())),
            ("prefix_tokens".into(), Json::Int(self.prefix_tokens.to_string())),
            ("branches".into(), Json::Int(self.branches.to_string())),
        ])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SystemOneResponse {
    pub model: String,
    /// In qev's order: chunked choices first, then the other questions, each in request order.
    pub answers: Vec<(String, Answer)>,
    pub usage: Usage,
}

impl SystemOneResponse {
    /// `response.model_dump()` plus `payload["qev"] = meta` when given (a JSON object the server builds, e.g. from
    /// [`QevMeta::to_json`] with `timing` and `diagnostics` appended).
    pub fn to_json(&self, qev: Option<Json>) -> Json {
        let usage = Json::Object(vec![
            ("input_tokens".into(), Json::Int(self.usage.input_tokens.to_string())),
            ("output_tokens".into(), Json::Int(self.usage.output_tokens.to_string())),
        ]);
        let mut m = vec![
            ("model".into(), Json::str(self.model.as_str())),
            ("answers".into(), Json::Object(self.answers.iter().map(|(k, a)| (k.clone(), a.to_json())).collect())),
            ("usage".into(), usage),
        ];
        if let Some(q) = qev {
            m.push(("qev".into(), q));
        }
        Json::Object(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(body: &str) -> Vec<(String, String, String)> {
        match SystemOneRequest::from_body(body.as_bytes()) {
            Err(ApiError::Unprocessable(items)) => {
                items.into_iter().map(|e| (e.loc.join("/"), e.msg, e.kind)).collect()
            }
            other => panic!("expected a 422, got {other:?}"),
        }
    }

    #[test]
    fn valid_requests_keep_order_and_defaults() {
        let r = SystemOneRequest::from_body(
            br#"{"questions": {"z": {"type": "noul", "x": 1}, "a": {"type": "choice", "criteria": {"b": null, "a": "x"}}}}"#,
        )
        .unwrap();
        assert_eq!(r.model, "jev-latest");
        assert_eq!(r.state, Json::Null);
        assert_eq!(r.questions[0].0, "z");
        assert_eq!(r.questions[0].1, Question::Noul { instructions: Json::Null, criteria: None });
        let Question::Choice { criteria, .. } = &r.questions[1].1 else { panic!() };
        assert_eq!(criteria[0], ("b".to_string(), Json::Null));
    }

    #[test]
    fn errors_match_pydantic() {
        assert_eq!(err("[1]"), [("body".into(), "Input should be a valid dictionary or instance of SystemOneRequest".into(), "model_type".into())]);
        assert_eq!(err("{"), [("body".into(), "invalid JSON".into(), "json_invalid".into())]);
        assert_eq!(err(r#"{"questions": {}}"#)[0].1, "Value error, at least one question is required");
        let e = err(r#"{"model": 3, "questions": {"q": "x"}}"#);
        assert_eq!(e[0], ("body/model".into(), "Input should be a valid string".into(), "string_type".into()));
        assert_eq!(e[1].0, "body/questions/q/NoulQuestion");
        assert_eq!(e[1].1, "Input should be a valid dictionary or instance of NoulQuestion");
        assert_eq!(e[3].0, "body/questions/q/function-after[_check(), ScoreQuestion]");
        let e = err(r#"{"questions": {"q": {"type": "score", "criteria": [1]}}}"#);
        let locs: Vec<_> = e.iter().map(|x| x.0.as_str()).collect();
        assert_eq!(locs, [
            "body/questions/q/NoulQuestion/type",
            "body/questions/q/NoulQuestion/criteria",
            "body/questions/q/function-after[_check(), ChoiceQuestion]/type",
            "body/questions/q/function-after[_check(), ChoiceQuestion]/criteria",
            "body/questions/q/function-after[_check(), ScoreQuestion]/criteria/0/str",
            "body/questions/q/function-after[_check(), ScoreQuestion]/criteria/0/dict[str,any]",
            "body/questions/q/function-after[_check(), ScoreQuestion]/criteria/0/list[any]",
        ]);
    }

    #[test]
    fn model_routing_message() {
        assert!(check_model("", "elephant-1").is_ok());
        assert!(check_model("qev-latest", "elephant-1").is_ok());
        assert!(check_model("elephant-1", "elephant-1").is_ok());
        let e = check_model("it's", "elephant-1").unwrap_err();
        assert_eq!(
            e,
            ApiError::NotFound(
                "model \"it's\" is not served; available: elephant-1 plus aliases ['jev-latest', 'jev-preview', 'qev-latest']".into()
            )
        );
        assert_eq!(e.status(), 404);
    }
}
