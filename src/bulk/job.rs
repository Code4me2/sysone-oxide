//! Input lines of `sysone bulk`: one job per line, `{"id": ..., "body": {...}}`.
//!
//! - `id`: a string or an integer, unique within the input. Results are keyed by it.
//! - `body`: a `/v1/systemone` request body as JSON (any value: an invalid one gets the server's 422).
//! - `body_text` (instead of `body`): the request body as raw text, parsed exactly like an HTTP body (so text that
//!   is not JSON gets the server's "invalid JSON" 422).
//! - `query` (optional): the query string the request would carry (`debias=2&debug=1`); default: `--query`.
//!
//! Other keys are ignored. A line that is not a JSON object, or has no usable `id`, or more than one body, is a
//! *bad line*: it is reported (with its line number) and not processed, because it has no id to key a result by.

use sha2::{Digest, Sha256};

use crate::api::schema::ApiError;
use crate::frontend::pyjson::{self, Json};

/// A 128-bit digest of a job id's compact JSON (`"7"` and `7` are different ids). Used for the done and seen
/// sets, so resuming a run over millions of jobs holds 16 bytes per id rather than the ids themselves.
pub type IdKey = u128;

pub fn id_key(id: &Json) -> IdKey {
    let text = id.dumps_compact().unwrap_or_default();
    let d = Sha256::digest(text.as_bytes());
    u128::from_le_bytes(d[..16].try_into().unwrap())
}

/// One job read from the input.
#[derive(Debug)]
pub struct Job {
    /// 1-based input line number.
    pub line: u64,
    pub id: Json,
    pub key: IdKey,
    /// The parsed request body, or the server's error for a body that is not JSON (`body_text`).
    pub body: Result<Json, ApiError>,
    pub query: Option<String>,
}

/// What one input line holds.
#[derive(Debug)]
pub enum Line {
    Job(Job),
    /// Empty or whitespace only: skipped silently.
    Blank,
    Bad { line: u64, msg: String },
}

/// Parse input line `line` (1-based) of `bytes` (without its newline).
pub fn parse_line(line: u64, bytes: &[u8]) -> Line {
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Line::Blank;
    }
    let bad = |msg: &str| Line::Bad { line, msg: msg.to_string() };
    let Ok(v) = pyjson::parse(bytes) else { return bad("not valid JSON") };
    let Json::Object(fields) = v else { return bad("not a JSON object") };
    let mut id = None;
    let mut body = None;
    let mut body_text = None;
    let mut query = None;
    for (k, v) in fields {
        match k.as_str() {
            "id" => id = Some(v),
            "body" => body = Some(v),
            "body_text" => body_text = Some(v),
            "query" => query = Some(v),
            _ => {}
        }
    }
    let id = match id {
        Some(v @ (Json::Str(_) | Json::Int(_))) => v,
        Some(_) => return bad("\"id\" must be a string or an integer"),
        None => return bad("no \"id\""),
    };
    let body = match (body, body_text) {
        (Some(b), None) => Ok(b),
        (None, Some(Json::Str(t))) => pyjson::parse(t.as_bytes()).map_err(|_| ApiError::invalid_json()),
        (None, Some(_)) => return bad("\"body_text\" must be a string"),
        (Some(_), Some(_)) => return bad("both \"body\" and \"body_text\""),
        (None, None) => return bad("no \"body\" or \"body_text\""),
    };
    let query = match query {
        None | Some(Json::Null) => None,
        Some(Json::Str(q)) => Some(q),
        Some(_) => return bad("\"query\" must be a string"),
    };
    Line::Job(Job { line, key: id_key(&id), id, body, query })
}

/// The id field of a result line (`{"id":...,`), parsed without the rest of the line. Result lines are written by
/// [`result_line`], so the id is always the first key.
pub fn result_id(line: &[u8]) -> Option<Json> {
    let v = pyjson::parse(line).ok()?;
    v.get("id").cloned()
}

/// A result line: `{"id":<id>,"status":<status>,"body":<body>}` + newline, where `<body>` is exactly the bytes the
/// server would send (compact JSON).
pub fn result_line(id: &Json, status: u16, body: &Json) -> Result<Vec<u8>, String> {
    let mut s = String::from("{\"id\":");
    s.push_str(&id.dumps_compact()?);
    s.push_str(&format!(",\"status\":{status},\"body\":"));
    s.push_str(&body.dumps_compact()?);
    s.push_str("}\n");
    Ok(s.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(s: &str) -> Job {
        match parse_line(1, s.as_bytes()) {
            Line::Job(j) => j,
            other => panic!("{s}: {other:?}"),
        }
    }

    fn bad(s: &str) -> String {
        match parse_line(3, s.as_bytes()) {
            Line::Bad { line, msg } => {
                assert_eq!(line, 3);
                msg
            }
            other => panic!("{s}: {other:?}"),
        }
    }

    #[test]
    fn lines_parse_like_the_server_parses_bodies() {
        let j = job(r#"{"id": "a", "body": {"state": "x", "questions": {}}, "query": "debias=2"}"#);
        assert_eq!(j.id, Json::str("a"));
        assert_eq!(j.query.as_deref(), Some("debias=2"));
        assert!(j.body.unwrap().get("state").is_some());
        // A raw body that is not JSON gets the server's invalid-JSON error; any JSON value is passed on.
        let j = job(r#"{"id": 7, "body_text": "{nope"}"#);
        assert_eq!(j.id, Json::Int("7".into()));
        assert_eq!(j.body.unwrap_err().status(), 422);
        assert_eq!(job(r#"{"id": 8, "body": [1, 2]}"#).body.unwrap(), Json::Array(vec![Json::Int("1".into()), Json::Int("2".into())]));
        // Duplicate keys inside the body survive (Python's json.loads keeps the last; pyjson does the same).
        let j = job(r#"{"id": 9, "body_text": "{\"state\": 1, \"state\": \"two\"}"}"#);
        assert_eq!(j.body.unwrap().get("state"), Some(&Json::str("two")));
        assert!(matches!(parse_line(1, b"  "), Line::Blank));
        assert!(bad("[1]").contains("object"));
        assert!(bad("{oops").contains("JSON"));
        assert!(bad(r#"{"body": {}}"#).contains("id"));
        assert!(bad(r#"{"id": 1.5, "body": {}}"#).contains("id"));
        assert!(bad(r#"{"id": null, "body": {}}"#).contains("id"));
        assert!(bad(r#"{"id": "x"}"#).contains("body"));
        assert!(bad(r#"{"id": "x", "body": {}, "body_text": "{}"}"#).contains("both"));
        assert!(bad(r#"{"id": "x", "body": {}, "query": 3}"#).contains("query"));
    }

    #[test]
    fn ids_of_different_types_differ() {
        assert_ne!(id_key(&Json::str("7")), id_key(&Json::Int("7".into())));
        assert_eq!(id_key(&Json::str("a")), id_key(&Json::str("a")));
    }

    #[test]
    fn result_lines_round_trip_their_id() {
        let body = Json::Object(vec![("detail".into(), Json::str("x"))]);
        let l = result_line(&Json::str("d\"1"), 422, &body).unwrap();
        assert_eq!(std::str::from_utf8(&l).unwrap(), "{\"id\":\"d\\\"1\",\"status\":422,\"body\":{\"detail\":\"x\"}}\n");
        assert_eq!(result_id(&l[..l.len() - 1]), Some(Json::str("d\"1")));
    }
}
