//! Frontend gate (DESIGN §9.2, frontend row): every request of the render goldens (tests/golden/dev/render, made by
//! reference/render_golden.py from qev's own server) through sysone's frontend. Required for 100% of the corpus:
//! byte-identical prompt strings; identical prefix ids, suffix ids, labels, owners (question, chunk, order), slot
//! counts, option spans, limit quantities, usage and qev meta; identical error status and bodies.
//!
//! Needs the elephant-1 tokenizer and chat template: `$SYSONE_MODEL_DIR`, default
//! /home/velvet/Desktop/hf_models/elephant-1. Run with `--release` (the corpus holds ~4 M tokens of states).

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use sysone::api::schema::{ApiError, Question, SystemOneRequest, check_model};
use sysone::contract::DecisionBatch;
use sysone::frontend::pyjson::{self, Json};
use sysone::frontend::{Frontend, Limits, Plan, PlanError};

const SERVED: &str = "elephant-1";

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"))
}

fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn lines(path: &Path) -> Vec<String> {
    // Split on "\n" only: strings may hold U+2028.
    fs::read_to_string(path).unwrap().split('\n').filter(|l| !l.is_empty()).map(str::to_string).collect()
}

fn ids(v: &Json) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| if let Json::Int(s) = x { s.parse().unwrap() } else { panic!("id") }).collect()
}

fn spans(v: &Json) -> Vec<(usize, usize)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = ids(p);
            (p[0] as usize, p[1] as usize)
        })
        .collect()
}

fn int(v: Option<&Json>) -> Option<u64> {
    match v {
        Some(Json::Int(s)) => s.parse().ok(),
        _ => None,
    }
}

/// Mismatches by field, with the first few examples.
#[derive(Default)]
struct Report {
    checked: BTreeMap<&'static str, usize>,
    failed: BTreeMap<&'static str, Vec<String>>,
}

impl Report {
    fn eq<T: PartialEq + std::fmt::Debug>(&mut self, field: &'static str, id: &str, got: T, want: T) {
        *self.checked.entry(field).or_default() += 1;
        if got != want {
            let msg = format!("{id}: got {:.300} want {:.300}", format!("{got:?}"), format!("{want:?}"));
            self.failed.entry(field).or_default().push(msg);
        }
    }
}

/// `SystemOneRequest.model_construct` as the golden generator does for `engine_direct` cases: no validation.
fn construct(body: &Json) -> SystemOneRequest {
    let questions = body
        .get("questions")
        .unwrap()
        .as_object()
        .unwrap()
        .iter()
        .map(|(qid, q)| {
            assert_eq!(q.get("type").and_then(Json::as_str), Some("choice"), "engine_direct cases are choices");
            let criteria = q.get("criteria").unwrap().as_object().unwrap().to_vec();
            let instructions = q.get("instructions").cloned().unwrap_or(Json::Null);
            (qid.clone(), Question::Choice { instructions, criteria })
        })
        .collect();
    SystemOneRequest { state: body.get("state").cloned().unwrap_or(Json::Null), model: SERVED.into(), questions }
}

/// Compare a plan (and the outcome of its checks) with a golden `render` record.
fn compare_render(rep: &mut Report, id: &str, req: &SystemOneRequest, plan: &Plan, checked: &Result<(), PlanError>, r: &Json) {
    rep.eq("prefix_ids", id, plan.prefix_ids.clone(), ids(r.get("prefix_ids").unwrap()));
    rep.eq("prefix_tokens", id, Some(plan.prefix_ids.len() as u64), int(r.get("prefix_tokens")));
    rep.eq("longest", id, Some(plan.longest as u64), int(r.get("longest")));
    rep.eq("total", id, Some(plan.total as u64), int(r.get("total")));
    rep.eq("prompt_head", id, plan.prompt_head.as_str(), r.get("prompt_head").unwrap().as_str().unwrap());
    rep.eq("prompt_tail", id, plan.prompt_tail.as_str(), r.get("prompt_tail").unwrap().as_str().unwrap());
    let err = match checked {
        Ok(()) => None,
        Err(PlanError::Value(m)) => Some(format!("ValueError: {m}")),
        Err(e) => panic!("{id}: {e}"),
    };
    rep.eq("error", id, err.as_deref(), r.get("error").unwrap().as_str());
    // The generator records `_split`'s result only when it returned (no limit error before it, no error in it).
    let split_ran = r.get("split_ran") == Some(&Json::Bool(true));
    rep.eq("split_ran", id, checked.is_ok(), split_ran);
    let gb = r.get("branches").unwrap().as_array().unwrap();
    rep.eq("branches", id, plan.branches.len(), gb.len());
    for (i, (b, g)) in plan.branches.iter().zip(gb).enumerate() {
        let bid = format!("{id}#{i}");
        rep.eq("qid", &bid, req.questions[b.question].0.as_str(), g.get("qid").unwrap().as_str().unwrap());
        rep.eq("kind", &bid, b.kind.as_str(), g.get("kind").unwrap().as_str().unwrap());
        rep.eq("chunk", &bid, b.chunk.map(|c| c as u64), int(g.get("chunk")));
        let order: Vec<usize> = ids(g.get("order").unwrap()).into_iter().map(|x| x as usize).collect();
        rep.eq("order", &bid, b.order.clone(), order);
        let labels: Vec<&str> = g.get("labels").unwrap().as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
        rep.eq("labels", &bid, b.labels.iter().map(String::as_str).collect::<Vec<_>>(), labels);
        rep.eq("n_slots", &bid, Some(b.n_slots() as u64), int(g.get("n_slots")));
        rep.eq("descriptions", &bid, &b.descriptions[..], g.get("descriptions").unwrap().as_array().unwrap());
        rep.eq("suffix_text", &bid, b.suffix_text.as_str(), g.get("suffix_text").unwrap().as_str().unwrap());
        rep.eq("option_char_spans", &bid, b.option_char_spans.clone(), spans(g.get("option_char_spans").unwrap()));
        rep.eq("suffix_ids", &bid, b.suffix_ids.clone(), ids(g.get("suffix_ids").unwrap()));
        if let Some(ts) = g.get("option_token_spans") {
            // Present when _split ran; it computes them for every branch before returning.
            if checked.is_ok() {
                rep.eq("option_token_spans", &bid, b.option_token_spans.clone(), spans(ts));
            }
        }
    }
    if checked.is_ok() {
        let mut batch = DecisionBatch::default();
        plan.push_into(&mut batch).unwrap();
        batch.validate().unwrap();
        rep.eq("batch_tokens", id, batch.num_tokens(), plan.total);
    }
}

#[test]
fn frontend_matches_render_goldens() {
    let dir = repo().join("tests/golden/dev/render");
    let prov = pyjson::parse(&fs::read(dir.join("provenance.json")).unwrap()).unwrap();
    let model = model_dir();
    assert!(model.join("tokenizer.json").exists(), "no checkpoint at {} (set SYSONE_MODEL_DIR)", model.display());
    // The goldens were made with this exact tokenizer and template.
    let shas = prov.get("checkpoint").unwrap().get("sha256").unwrap();
    for f in ["tokenizer.json", "chat_template.jinja", "config.json", "readout.json"] {
        let want = shas.get(f).unwrap().as_str().unwrap();
        assert_eq!(sha256_hex(&fs::read(model.join(f)).unwrap()), want, "{f} differs from the golden checkpoint");
    }
    let lim = prov.get("limits").unwrap();
    let limits = Limits {
        max_branch_tokens: int(lim.get("max_branch_tokens")).unwrap() as usize,
        max_request_tokens: int(lim.get("max_request_tokens")).unwrap() as usize,
    };
    let fe = Frontend::load(&model, limits).unwrap();
    assert!(fe.noncausal, "elephant-1 is noncausal");

    // Slot table: label -> id, all 256 single tokens.
    let slots = prov.get("slots").unwrap().as_array().unwrap();
    let golden_slots: Vec<u32> = slots.iter().map(|s| int(s.get("id")).unwrap() as u32).collect();
    assert_eq!(fe.slot_ids, golden_slots, "slot ids");
    assert_eq!(fe.max_slots, int(prov.get("tokenizer").unwrap().get("max_slots")).unwrap() as usize);

    let corpus: HashMap<String, (String, Json)> = lines(&repo().join("reference/corpus/requests.jsonl"))
        .into_iter()
        .map(|l| {
            let v = pyjson::parse_str(&l).unwrap();
            (v.get("id").unwrap().as_str().unwrap().to_string(), (l, v))
        })
        .collect();

    let mut rep = Report::default();
    let mut cases = 0;
    let t0 = std::time::Instant::now();
    let files = prov.get("files").unwrap().as_object().unwrap();
    for (name, _) in files {
        for line in lines(&dir.join(name)) {
            let rec = pyjson::parse_str(&line).unwrap();
            let id = rec.get("id").unwrap().as_str().unwrap();
            cases += 1;
            let (cline, case) = &corpus[id];
            rep.eq("request_sha256", id, sha256_hex(cline.as_bytes()).as_str(), rec.get("request_sha256").unwrap().as_str().unwrap());
            let body = case.get("body").unwrap();
            let debias: i64 = int(case.get("debias")).unwrap() as i64;
            let http = rec.get("http").unwrap();
            let status = int(http.get("status")).unwrap() as u16;
            let gbody = http.get("body").unwrap();

            // What qev's server sends: the corpus body re-serialized compactly (render_golden.py's `dumps`).
            let payload = body.dumps_compact().unwrap();
            let req = match SystemOneRequest::from_body(payload.as_bytes())
                .and_then(|r| check_model(&r.model, SERVED).map(|()| r))
            {
                Ok(r) => r,
                Err(e) => {
                    rep.eq("status", id, e.status(), status);
                    rep.eq("error_body", id, e.body(), gbody.clone());
                    rep.eq("render_is_null", id, true, rec.get("render").unwrap().is_null());
                    if let Some(ed) = rec.get("engine_direct") {
                        let req = construct(body);
                        let mut plan = fe.plan_unchecked(&req, debias).unwrap();
                        let checked = fe.check(&mut plan);
                        let edr = ed.get("render").unwrap();
                        compare_render(&mut rep, &format!("{id}/engine_direct"), &req, &plan, &checked, edr);
                        let outcome = ed.get("outcome").unwrap();
                        rep.eq("engine_direct_ok", id, checked.is_ok(), outcome.get("ok") == Some(&Json::Bool(true)));
                        if checked.is_ok() {
                            let usage = fe.response(&plan, vec![], SERVED).to_json(None);
                            rep.eq("engine_direct_usage", id, usage.get("usage"), outcome.get("usage"));
                            let meta = fe.meta(&plan, "sequential").to_json();
                            rep.eq("engine_direct_qev", id, Some(&meta), outcome.get("qev"));
                        }
                    }
                    continue;
                }
            };
            let r = rec.get("render").unwrap();
            let mut plan = match fe.plan_unchecked(&req, debias) {
                Ok(p) => p,
                Err(e) => panic!("{id}: planning failed: {e}"),
            };
            let checked = fe.check(&mut plan);
            compare_render(&mut rep, id, &req, &plan, &checked, r);
            match &checked {
                Ok(()) => {
                    rep.eq("status", id, 200, status);
                    let meta = fe.meta(&plan, "sequential").to_json();
                    let mut got = fe.response(&plan, vec![], SERVED).to_json(Some(meta));
                    if let Json::Object(m) = &mut got {
                        m.retain(|(k, _)| k != "answers");
                    }
                    rep.eq("response_body", id, got, gbody.clone());
                }
                Err(PlanError::Value(m)) => {
                    let e = ApiError::value_error(m.clone());
                    rep.eq("status", id, e.status(), status);
                    rep.eq("error_body", id, e.body(), gbody.clone());
                }
                Err(e) => panic!("{id}: {e}"),
            }
        }
    }
    let total_checks: usize = rep.checked.values().sum();
    println!("frontend gate: {cases} cases, {total_checks} field checks in {:.1}s", t0.elapsed().as_secs_f64());
    for (field, n) in &rep.checked {
        let bad = rep.failed.get(field).map_or(0, Vec::len);
        println!("  {field:<22} {:>6} checked {:>4} mismatched", n, bad);
    }
    for (field, msgs) in &rep.failed {
        for m in msgs.iter().take(5) {
            println!("MISMATCH {field}: {m}");
        }
    }
    assert_eq!(cases as u64, int(prov.get("counts").unwrap().get("cases")).unwrap(), "case count");
    assert!(rep.failed.is_empty(), "{} fields mismatched", rep.failed.len());
}
