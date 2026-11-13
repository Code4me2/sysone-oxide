//! Answer assembly, extra schema cases and Python float built-ins against qev itself
//! (tests/golden/dev/assemble, written by reference/assemble_golden.py):
//! - answers.jsonl: qev's `decide()` on synthetic slot logits (debias, chunks, calibration temperatures, ties,
//!   underflow); the answers and debug diagnostics must be byte-identical to Starlette's JSON of qev's.
//! - schema.jsonl: 422/404 bodies of qev's server for malformed requests.
//! - pyfloat.json: `repr(float)`, `round(x, 6)` and Python 3.12 `sum()`.
//!
//! Needs the elephant-1 tokenizer and chat template (`$SYSONE_MODEL_DIR`, default
//! /home/velvet/Desktop/hf_models/elephant-1).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use sysone::api::schema::{Question, SystemOneRequest, check_model};
use sysone::frontend::assemble::{Calibration, assemble};
use sysone::frontend::pyjson::{self, Json, py_float_repr, py_fsum, py_round6};
use sysone::frontend::{Frontend, Limits, PlanError};

const SERVED: &str = "elephant-1";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn golden(name: &str) -> PathBuf {
    repo().join("tests/golden/dev/assemble").join(name)
}

fn frontend() -> Frontend {
    let model = std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"));
    Frontend::load(&model, Limits::default()).unwrap_or_else(|e| panic!("loading {}: {e:#}", model.display()))
}

fn records(name: &str) -> Vec<Json> {
    fs::read_to_string(golden(name))
        .unwrap()
        .split('\n')
        .filter(|l| !l.is_empty())
        .map(|l| pyjson::parse_str(l).unwrap())
        .collect()
}

fn f64_of(v: &Json) -> f64 {
    match v {
        Json::Float(f) => *f,
        Json::Int(s) => s.parse().unwrap(),
        _ => panic!("not a number: {v:?}"),
    }
}

/// Python's `float.fromhex` for the `float.hex()` strings in pyfloat.json.
fn from_hex(s: &str) -> f64 {
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let s = s.strip_prefix("0x").unwrap();
    let (mant, exp) = s.split_once('p').unwrap();
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    // At most 53 significant bits, so the mantissa is exact as an f64; scale by exact powers of two.
    let m = u64::from_str_radix(&format!("{int}{frac}"), 16).unwrap() as f64;
    let e = exp.parse::<i32>().unwrap() - 4 * frac.len() as i32;
    let pow2 = |e: i32| f64::from_bits(((e + 1023) as u64) << 52);
    let v = if e >= -1022 { m * pow2(e) } else { m * pow2(-1022) * pow2(e + 1022) };
    if neg { -v } else { v }
}

#[test]
fn python_float_builtins() {
    let v = pyjson::parse(&fs::read(golden("pyfloat.json")).unwrap()).unwrap();
    let mut n = 0;
    for p in v.get("repr").unwrap().as_array().unwrap() {
        let p = p.as_array().unwrap();
        let x = from_hex(p[0].as_str().unwrap());
        assert_eq!(py_float_repr(x), p[1].as_str().unwrap(), "repr of {}", p[0].as_str().unwrap());
        n += 1;
    }
    for p in v.get("round6").unwrap().as_array().unwrap() {
        let p = p.as_array().unwrap();
        let x = from_hex(p[0].as_str().unwrap());
        assert_eq!(py_float_repr(py_round6(x)), p[1].as_str().unwrap(), "round of {}", p[0].as_str().unwrap());
        n += 1;
    }
    for p in v.get("sum").unwrap().as_array().unwrap() {
        let p = p.as_array().unwrap();
        let xs: Vec<f64> = p[0].as_array().unwrap().iter().map(|x| from_hex(x.as_str().unwrap())).collect();
        let want = from_hex(p[1].as_str().unwrap());
        assert_eq!(py_fsum(xs.iter().copied()).to_bits(), want.to_bits(), "sum of {xs:?}");
        n += 1;
    }
    println!("{n} float checks");
    assert!(n > 1000);
}

#[test]
fn schema_errors_match_qev_server() {
    let fe = frontend();
    let recs = records("schema.jsonl");
    for r in &recs {
        let id = r.get("id").unwrap().as_str().unwrap();
        let body = r.get("body").unwrap().as_str().unwrap();
        let status = f64_of(r.get("status").unwrap()) as u16;
        let res = SystemOneRequest::from_body(body.as_bytes()).and_then(|q| check_model(&q.model, SERVED).map(|()| q));
        match res {
            Ok(req) => {
                assert_eq!(status, 200, "{id}: accepted, qev answered {status}");
                fe.plan(&req, 1).unwrap_or_else(|e| panic!("{id}: {e}"));
            }
            Err(e) => {
                assert_eq!(e.status(), status, "{id}: status");
                assert_eq!(&e.body(), r.get("response").unwrap(), "{id}: body");
                // Byte-level: Starlette's rendering of the same body.
                let want = r.get("response").unwrap().dumps_compact().unwrap();
                assert_eq!(String::from_utf8(e.body_bytes()).unwrap(), want, "{id}: bytes");
            }
        }
    }
    assert!(recs.len() >= 30);
}

#[test]
fn answers_match_qev_assembly() {
    let mut fe = frontend();
    let corpus: HashMap<String, Json> = fs::read_to_string(repo().join("reference/corpus/requests.jsonl"))
        .unwrap()
        .split('\n')
        .filter(|l| !l.is_empty())
        .map(|l| {
            let v = pyjson::parse_str(l).unwrap();
            (v.get("id").unwrap().as_str().unwrap().to_string(), v)
        })
        .collect();
    let recs = records("answers.jsonl");
    let (mut chunked, mut debiased, mut calibrated, mut errors) = (0, 0, 0, 0);
    for r in &recs {
        let id = r.get("id").unwrap().as_str().unwrap();
        let case = &corpus[r.get("case").unwrap().as_str().unwrap()];
        let body = case.get("body").unwrap();
        let req = if r.get("engine_direct") == Some(&Json::Bool(true)) {
            let questions = body
                .get("questions")
                .unwrap()
                .as_object()
                .unwrap()
                .iter()
                .map(|(qid, q)| {
                    let criteria = q.get("criteria").unwrap().as_object().unwrap().to_vec();
                    let instructions = q.get("instructions").cloned().unwrap_or(Json::Null);
                    (qid.clone(), Question::Choice { instructions, criteria })
                })
                .collect();
            SystemOneRequest { state: body.get("state").cloned().unwrap_or(Json::Null), model: SERVED.into(), questions }
        } else {
            SystemOneRequest::validate(body).unwrap()
        };
        let debias = f64_of(r.get("debias").unwrap()) as i64;
        // Chunking needs choices wider than the slot count; the generator replays some with fewer slots.
        fe.max_slots = f64_of(r.get("max_slots").unwrap()) as usize;
        let plan = match fe.plan(&req, debias) {
            Ok(p) => p,
            Err(PlanError::Value(m)) => {
                assert_eq!(Some(m.as_str()), r.get("error").and_then(Json::as_str), "{id}: error");
                errors += 1;
                continue;
            }
            Err(e) => panic!("{id}: {e}"),
        };
        let mut cal = Calibration::none();
        for (k, v) in r.get("calibration").unwrap().as_object().unwrap() {
            cal.temperatures.insert(k.clone(), f64_of(v));
        }
        calibrated += !cal.temperatures.is_empty() as usize;
        let logits: Vec<Vec<f32>> = r
            .get("logits")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(|x| f64_of(x) as f32).collect())
            .collect();
        assert_eq!(logits.len(), plan.branches.len(), "{id}: branch count");
        let masses: Option<Vec<f64>> = match r.get("masses").unwrap() {
            Json::Null => None,
            m => Some(m.as_array().unwrap().iter().map(f64_of).collect()),
        };
        chunked += plan.branches.iter().any(|b| b.chunk.is_some()) as usize;
        debiased += (plan.branches.len() > req.questions.len()) as usize;
        let out = assemble(&req, &plan, &cal, fe.max_slots, |i| logits[i].as_slice(), masses.as_deref()).unwrap();
        let answers = Json::Object(out.answers.iter().map(|(q, a)| (q.clone(), a.to_json())).collect());
        assert_eq!(answers.dumps_compact().unwrap(), r.get("answers_json").unwrap().as_str().unwrap(), "{id}: answers");
        assert_eq!(
            out.diagnostics.dumps_compact().unwrap(),
            r.get("diagnostics_json").unwrap().as_str().unwrap(),
            "{id}: diagnostics"
        );
        let usage = fe.response(&plan, out.answers, SERVED).to_json(None);
        assert_eq!(usage.get("usage"), r.get("usage"), "{id}: usage");
    }
    println!("{} cases: {chunked} chunked, {debiased} debiased, {calibrated} calibrated, {errors} errors", recs.len());
    assert!(recs.len() > 200 && chunked >= 20 && debiased > 20 && calibrated > 50);
}
