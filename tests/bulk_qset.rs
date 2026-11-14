//! `sysone bulk --questions Q.json --docs D` without a GPU (mock engine; needs the checkpoint's tokenizer and chat
//! template from `$SYSONE_MODEL_DIR`, like tests/bulk.rs):
//!
//! - `prepared_plans_equal_full_plans`: [`Frontend::plan_prepared`] (suffix tokens reused) gives exactly the
//!   [`Frontend::plan`] of the full request, for states that stress the prompt frame and the prefix split, every
//!   debias, with causal and noncausal attention (`_split`'s option spans), and the same errors.
//! - `question_set_results_equal_full_body_jobs`: a question set over a documents JSONL (per-document queries,
//!   overrides, every error status) and over a directory of text files gives byte for byte the result lines of
//!   the equivalent full-body jobs (apart from `qev.timing`).
//! - `question_set_runs_resume`: SIGKILL mid-run plus a torn last line; the rerun completes and equals a clean run.
//! - `question_sets_are_validated_before_any_work`: invalid question sets and argument combinations are refused
//!   with a clear message before the output file is created or a worker starts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sysone::api::schema::SystemOneRequest;
use sysone::frontend::pyjson::{self, Json};
use sysone::frontend::{Frontend, Limits};

fn model_dir() -> PathBuf {
    let m = std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"));
    assert!(m.join("tokenizer.json").exists(), "no checkpoint at {} (set SYSONE_MODEL_DIR)", m.display());
    m
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_sysone")
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sysone-qset-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn parse(b: &[u8]) -> Json {
    pyjson::parse(b).unwrap_or_else(|e| panic!("not JSON ({e:?}): {}", String::from_utf8_lossy(b)))
}

fn s(text: &str) -> String {
    Json::str(text).dumps_compact().unwrap()
}

/// A body without `qev.timing`, as compact JSON text.
fn untimed(v: &Json) -> String {
    let mut v = v.clone();
    if let Json::Object(m) = &mut v {
        for (k, q) in m.iter_mut() {
            if k == "qev"
                && let Json::Object(q) = q
            {
                q.retain(|(k, _)| k != "timing");
            }
        }
    }
    v.dumps_compact().unwrap()
}

/// Result lines by id: (status, untimed body).
fn results(path: &Path) -> BTreeMap<String, (u16, String)> {
    let mut m = BTreeMap::new();
    for l in std::fs::read(path).unwrap().split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        let v = parse(l);
        let id = v.get("id").unwrap().dumps_compact().unwrap();
        let status = match v.get("status") {
            Some(Json::Int(s)) => s.parse().unwrap(),
            other => panic!("status {other:?}"),
        };
        assert!(m.insert(id.clone(), (status, untimed(v.get("body").unwrap()))).is_none(), "id {id} twice in {}", path.display());
    }
    m
}

/// Drop what an outer scripts/gpu.sh exported (the mock runs need no lock).
fn clear_outer_gpu_sh(c: &mut Command) {
    for k in ["SYSONE_GPU_HELD", "SYSONE_GPU_LOCKED", "SYSONE_GPU", "SYSONE_GPU_LOCK", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"] {
        c.env_remove(k);
    }
}

/// Run `sysone bulk --model M --progress-s 0 ARGS`; returns (exit code, summary JSON, stderr).
fn bulk(args: &[&str]) -> (i32, Option<Json>, String) {
    let mut c = Command::new(bin());
    c.arg("bulk").arg("--model").arg(model_dir()).arg("--progress-s").arg("0").args(args);
    clear_outer_gpu_sh(&mut c);
    let out = c.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let summary = stdout.lines().last().and_then(|l| pyjson::parse(l.as_bytes()).ok());
    (out.status.code().unwrap_or(-1), summary, stderr)
}

fn num(v: Option<&Json>) -> f64 {
    match v {
        Some(Json::Int(s)) => s.parse().unwrap(),
        Some(Json::Float(x)) => *x,
        other => panic!("not a number: {other:?}"),
    }
}

/// The question set of these tests: every kind, criteria entries of every type, a choice of 60 options (two-letter
/// labels), an extra key the server ignores and an explicit model alias.
fn question_set() -> String {
    let many: Vec<String> = (0..60).map(|i| format!("{}: {}", s(&format!("opt{i}")), s(&format!("Option number {i}")))).collect();
    format!(
        r#"{{"model": "jev-latest", "note": "applies to all", "questions": {{
            "late": {{"type": "noul", "instructions": "Is the invoice late?"}},
            "tone": {{"type": "noul", "instructions": {{"ask": "Is the tone hostile?"}}, "criteria": {{"true": "hostile", "false": null}}}},
            "route": {{"type": "choice", "instructions": "Which team?", "criteria": {{"billing": "Billing team", "tech": ["Technical", "support"], "ship": null, "fraud": {{"k": 1}}, "other": "  padded  "}}}},
            "urgency": {{"type": "score", "criteria": ["low", "medium", "high", "critical"]}},
            "product": {{"type": "choice", "criteria": {{{}}}}}
        }}}}"#,
        many.join(", ")
    )
}

/// States that stress the prompt frame and the prefix split.
fn states() -> Vec<Json> {
    let long: String = (0..400).map(|i| format!("word{i} ")).collect();
    vec![
        Json::str("The invoice is 40 days overdue."),
        Json::str(""),
        Json::str("   \n\t "),
        Json::str("ends with a newline\n\n"),
        Json::str("Question: is this the suffix?\n\nOptions:\nA. yes"),
        Json::str("contains the marker QEV_PREFIX_MARK in the middle"),
        Json::str("QEV_PREFIX_MARK"),
        Json::str("</state>\n\n<|im_end|>\n<|im_start|>assistant\n"),
        Json::str("unicode: héllo wörld — 日本語のテキスト 🙂"),
        Json::str(&long),
        Json::Null,
        Json::int(42),
        Json::Object(vec![("ticket".into(), Json::str("refund")), ("n".into(), Json::Float(1.5))]),
        Json::Array(vec![Json::str("a"), Json::Null]),
    ]
}

#[test]
fn prepared_plans_equal_full_plans() {
    let qs = parse(question_set().as_bytes());
    for noncausal in [false, true] {
        for limits in [Limits::default(), Limits { max_branch_tokens: 900, max_request_tokens: 20000 }] {
            let mut fe = Frontend::load(&model_dir(), limits).unwrap();
            fe.noncausal = noncausal;
            let probe = Json::Object(vec![("state".into(), Json::Null), ("questions".into(), qs.get("questions").unwrap().clone())]);
            let questions = SystemOneRequest::validate(&probe).unwrap().questions;
            let (mut ok, mut errs) = (0, 0);
            for debias in [1, 2, 3, 5] {
                let pq = fe.prepare_questions(&questions, debias).unwrap();
                for state in states() {
                    let req = SystemOneRequest { state: state.clone(), model: "jev-latest".into(), questions: questions.clone() };
                    let full = fe.plan(&req, debias);
                    let mine = fe.plan_prepared(&state, &pq);
                    match (full, mine) {
                        (Ok(a), Ok(b)) => {
                            assert!(a == b, "plans differ: state {state:?} debias {debias} noncausal {noncausal}");
                            // The cached suffix tokens were used: the tail is the same text for every state.
                            assert_eq!(b.prompt_tail, pq.tail(), "state {state:?}");
                            ok += 1;
                        }
                        (Err(a), Err(b)) => {
                            assert_eq!(a.to_string(), b.to_string(), "state {state:?} debias {debias}");
                            errs += 1;
                        }
                        (a, b) => panic!("state {state:?} debias {debias}: {:?} vs {:?}", a.map(|p| p.total), b.map(|p| p.total)),
                    }
                }
            }
            eprintln!("noncausal {noncausal}, limits {limits:?}: {ok} equal plans, {errs} equal errors");
            assert!(ok > 0);
            if limits.max_branch_tokens == 900 {
                assert!(errs > 0, "the small limits should refuse the long state");
            }
        }
    }
}

/// The equivalent full body of a document (the documented rule): `{"state": ...}`, the question set's fields,
/// then the overrides; a repeated key keeps its first position and takes the last value.
fn full_body(qs: &Json, state: &Json, overrides: &[(String, Json)]) -> Json {
    let mut m: Vec<(String, Json)> = vec![("state".into(), state.clone())];
    for (k, v) in qs.as_object().unwrap().iter().chain(overrides) {
        match m.iter_mut().find(|(key, _)| key == k) {
            Some(slot) => slot.1 = v.clone(),
            None => m.push((k.clone(), v.clone())),
        }
    }
    Json::Object(m)
}

#[test]
fn question_set_results_equal_full_body_jobs() {
    let dir = tmp("equal");
    let qtext = question_set();
    let qs = parse(qtext.as_bytes());
    let qfile = dir.join("questions.json");
    std::fs::write(&qfile, &qtext).unwrap();

    // Documents: (id, state, query, overrides).
    let mut docs: Vec<(Json, Json, Option<&str>, Vec<(String, Json)>)> = Vec::new();
    for (i, st) in states().into_iter().enumerate() {
        docs.push((Json::str(format!("s{i}")), st, None, vec![]));
    }
    for i in 0..40 {
        let st = Json::str(sysone::bulk::synth::words(i, 20 + i * 37));
        let id = if i % 2 == 0 { Json::int(1000 + i as i64) } else { Json::str(format!("doc-{i}")) };
        let q = match i % 5 {
            0 => Some("debias=3"),
            1 => Some("debug=1&debias=2"),
            _ => None,
        };
        docs.push((id, st, q, vec![]));
    }
    let other_q = parse(br#"{"x": {"type": "noul", "instructions": "Different question?"}}"#);
    let ov = |k: &str, v: Json| vec![(k.to_string(), v)];
    docs.push((Json::str("ov-model-404"), Json::str("text"), None, ov("model", Json::str("not-served"))));
    docs.push((Json::str("ov-model-ok"), Json::str("text"), None, ov("model", Json::str("elephant-1"))));
    docs.push((Json::str("ov-model-type"), Json::str("text"), None, ov("model", Json::int(3))));
    docs.push((Json::str("ov-questions"), Json::str("text"), Some("debias=2"), ov("questions", other_q.clone())));
    docs.push((Json::str("ov-questions-empty"), Json::str("text"), None, ov("questions", Json::Object(vec![]))));
    docs.push((Json::str("ov-questions-bad"), Json::str("text"), None, ov("questions", Json::str("nope"))));
    docs.push((Json::str("ov-ignored"), Json::str("text"), None, ov("note", Json::str("mine"))));
    docs.push((Json::str("debias-bad"), Json::str("text"), Some("debias=abc"), vec![]));
    // Over the 3,000-token branch limit set below: the server's limit 422.
    docs.push((Json::str("too-long"), Json::str(sysone::bulk::synth::words(7, 4000)), None, vec![]));

    let mut doc_lines: Vec<String> = Vec::new();
    let mut job_lines: Vec<String> = Vec::new();
    for (id, st, q, o) in &docs {
        let id_t = id.dumps_compact().unwrap();
        let mut d = format!("{{\"id\": {id_t}, \"state\": {}", st.dumps_compact().unwrap());
        let mut j = format!("{{\"id\": {id_t}, \"body\": {}", full_body(&qs, st, o).dumps_compact().unwrap());
        if let Some(q) = q {
            d.push_str(&format!(", \"query\": {}", s(q)));
            j.push_str(&format!(", \"query\": {}", s(q)));
        }
        if !o.is_empty() {
            d.push_str(&format!(", \"overrides\": {}", Json::Object(o.clone()).dumps_compact().unwrap()));
        }
        doc_lines.push(d + "}");
        job_lines.push(j + "}");
    }
    // Bad document lines: reported and not processed (the jobs file has none).
    doc_lines.insert(3, "not json".into());
    doc_lines.insert(7, r#"{"id": "no-state"}"#.into());
    doc_lines.insert(9, r#"{"id": "s1", "state": "duplicate id"}"#.into());
    doc_lines.insert(12, r#"{"id": "x", "state": "y", "overrides": {"state": "z"}}"#.into());
    doc_lines.insert(14, String::new());
    let (dfile, jfile) = (dir.join("docs.jsonl"), dir.join("jobs.jsonl"));
    std::fs::write(&dfile, doc_lines.join("\n") + "\n").unwrap();
    std::fs::write(&jfile, job_lines.join("\n") + "\n").unwrap();

    let common = ["--engine", "mock", "--max-branch-tokens", "3000", "--max-step-tokens", "65536", "--window", "24"];
    let (out_d, out_j) = (dir.join("out-docs.jsonl"), dir.join("out-jobs.jsonl"));
    let mut a: Vec<&str> = common.to_vec();
    a.extend(["--workers", "process", "--mock-workers", "2"]);
    a.extend(["--questions", qfile.to_str().unwrap(), "--docs", dfile.to_str().unwrap(), "--output", out_d.to_str().unwrap()]);
    let (code, sum_d, err) = bulk(&a);
    assert_eq!(code, 0, "{err}");
    let sum_d = sum_d.unwrap();
    let mut a: Vec<&str> = common.to_vec();
    a.extend(["--input", jfile.to_str().unwrap(), "--output", out_j.to_str().unwrap()]);
    let (code, sum_j, err2) = bulk(&a);
    assert_eq!(code, 0, "{err2}");
    let sum_j = sum_j.unwrap();

    assert_eq!((num(sum_d.get("bad_lines")), num(sum_d.get("duplicate_ids"))), (3.0, 1.0), "{err}");
    assert_eq!(num(sum_d.get("jobs")), docs.len() as f64);
    let q = sum_d.get("question_set").unwrap();
    assert_eq!((num(q.get("docs_full_body")), num(q.get("docs_prepared"))), (6.0, (docs.len() - 6) as f64));
    assert_eq!((num(q.get("questions")), num(q.get("branches"))), (5.0, 5.0));
    assert!(sum_j.get("question_set").unwrap().is_null());
    let (rd, rj) = (results(&out_d), results(&out_j));
    assert_eq!(rd.len(), docs.len());
    let mut statuses = BTreeMap::<u16, usize>::new();
    for (id, (status, body)) in &rj {
        let (s2, b2) = rd.get(id).unwrap_or_else(|| panic!("no question-set result for {id}"));
        assert_eq!((status, body), (s2, b2), "id {id}");
        *statuses.entry(*status).or_default() += 1;
    }
    eprintln!("statuses {statuses:?}");
    assert_eq!(rd, rj);
    assert!(statuses[&200] >= 50 && statuses[&422] >= 4 && statuses[&404] == 1 && statuses[&500] == 1, "{statuses:?}");
    // The same steps from both inputs (the same sizes in the same order).
    assert_eq!(sum_d.get("steps"), sum_j.get("steps"));
    assert_eq!(sum_d.get("packed_tokens"), sum_j.get("packed_tokens"));

    // A directory of text files: id = file name, state = the text; hidden files and subdirectories are not
    // documents, a file that is not UTF-8 is a bad record.
    let ddir = dir.join("docs-dir");
    std::fs::create_dir_all(ddir.join("sub")).unwrap();
    let mut jobs = Vec::new();
    for i in 0..12 {
        let name = format!("file-{i:02}.txt");
        let text = format!("{}\nline two of {name}\n", sysone::bulk::synth::words(i, 30 + 50 * i));
        std::fs::write(ddir.join(&name), &text).unwrap();
        jobs.push(format!("{{\"id\": {}, \"body\": {}}}", s(&name), full_body(&qs, &Json::str(text), &[]).dumps_compact().unwrap()));
    }
    std::fs::write(ddir.join(".hidden"), "not a document").unwrap();
    std::fs::write(ddir.join("sub/inner.txt"), "not a document either").unwrap();
    std::fs::write(ddir.join("zz-binary"), [0xffu8, 0xfe, 0x00]).unwrap();
    let jfile2 = dir.join("jobs2.jsonl");
    std::fs::write(&jfile2, jobs.join("\n") + "\n").unwrap();
    let (out_dd, out_j2) = (dir.join("out-dir.jsonl"), dir.join("out-jobs2.jsonl"));
    let mut a: Vec<&str> = common.to_vec();
    a.extend(["--questions", qfile.to_str().unwrap(), "--docs", ddir.to_str().unwrap(), "--output", out_dd.to_str().unwrap()]);
    let (code, sum, err) = bulk(&a);
    assert_eq!(code, 0, "{err}");
    assert_eq!(num(sum.unwrap().get("bad_lines")), 1.0, "{err}");
    assert!(err.contains("zz-binary"), "{err}");
    let mut a: Vec<&str> = common.to_vec();
    a.extend(["--input", jfile2.to_str().unwrap(), "--output", out_j2.to_str().unwrap()]);
    let (code, _, err) = bulk(&a);
    assert_eq!(code, 0, "{err}");
    let (rdd, rj2) = (results(&out_dd), results(&out_j2));
    assert_eq!(rdd.len(), 12);
    assert_eq!(rdd, rj2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn question_set_runs_resume() {
    let dir = tmp("resume");
    let (q, d) = (dir.join("q.json"), dir.join("docs.jsonl"));
    let st = Command::new(bin())
        .args(["bulk", "gen", "--docs", "150", "--min-words", "100", "--max-words", "2000", "--max-questions", "6"])
        .arg("--question-set")
        .arg(&q)
        .arg("--output")
        .arg(&d)
        .status()
        .unwrap();
    assert!(st.success());
    let (clean, crashed) = (dir.join("clean.jsonl"), dir.join("crashed.jsonl"));
    let args = |out: &Path| -> Vec<String> {
        ["--engine", "mock", "--max-step-tokens", "8192", "--window", "32", "--questions", q.to_str().unwrap(), "--docs", d.to_str().unwrap()]
            .iter()
            .map(|s| s.to_string())
            .chain(["--output".to_string(), out.to_str().unwrap().to_string()])
            .collect()
    };
    let a = args(&clean);
    let (code, _, err) = bulk(&a.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(code, 0, "{err}");

    // A slow run (20 us per bucket token), killed after some results.
    let mut c = Command::new(bin());
    c.arg("bulk").arg("--model").arg(model_dir()).args(args(&crashed)).args(["--mock-ns-per-token", "20000", "--progress-s", "0"]);
    clear_outer_gpu_sh(&mut c);
    let mut child = c.stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let t = Instant::now();
    loop {
        let n = std::fs::read(&crashed).map(|b| b.iter().filter(|&&c| c == b'\n').count()).unwrap_or(0);
        if n >= 20 {
            break;
        }
        assert!(t.elapsed() < Duration::from_secs(60), "no progress");
        std::thread::sleep(Duration::from_millis(20));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    let before = results(&crashed).len();
    assert!(before >= 20 && before < 150, "{before}");
    let mut f = std::fs::OpenOptions::new().append(true).open(&crashed).unwrap();
    std::io::Write::write_all(&mut f, b"{\"id\":\"doc-000149\",\"status\":200,\"bo").unwrap();
    drop(f);

    let a = args(&crashed);
    let (code, summary, err) = bulk(&a.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(code, 0, "{err}");
    let s = summary.unwrap();
    assert_eq!(num(s.get("skipped_done")) as usize, before);
    assert!(num(s.get("truncated_partial_bytes")) >= 35.0);
    assert_eq!(num(s.get("jobs")) as usize, 150 - before);
    assert_eq!(results(&crashed), results(&clean));
    let (code, summary, _) = bulk(&a.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!((code, num(summary.unwrap().get("jobs"))), (0, 0.0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn question_sets_are_validated_before_any_work() {
    let dir = tmp("validate");
    let docs = dir.join("docs.jsonl");
    std::fs::write(&docs, "{\"id\": 1, \"state\": \"x\"}\n").unwrap();
    let out = dir.join("out.jsonl");
    let run = |qtext: &str, extra: &[&str]| -> String {
        let q = dir.join("q.json");
        std::fs::write(&q, qtext).unwrap();
        let mut a = vec!["--engine", "mock", "--questions", q.to_str().unwrap(), "--docs", docs.to_str().unwrap(), "--output", out.to_str().unwrap()];
        a.extend(extra);
        let (code, summary, err) = bulk(&a);
        assert_ne!(code, 0, "{qtext}: accepted\n{err}");
        assert!(summary.is_none(), "{qtext}: ran");
        assert!(!out.exists(), "{qtext}: the output was created");
        assert!(!err.contains("worker"), "{qtext}: a worker started\n{err}");
        err
    };
    let noul = r#"{"a": {"type": "noul"}}"#;
    let cases: Vec<(String, Vec<&str>, &str)> = vec![
        ("{nope".into(), vec![], "not valid JSON"),
        ("[1]".into(), vec![], "must be a JSON object"),
        (r#"{"model": "jev-latest"}"#.into(), vec![], "no \"questions\""),
        (format!(r#"{{"state": "x", "questions": {noul}}}"#), vec![], "must not hold \"state\""),
        (r#"{"questions": {}}"#.into(), vec![], "at least one question is required"),
        (r#"{"questions": []}"#.into(), vec![], "valid dictionary"),
        (r#"{"questions": {"a": {"type": "maybe"}}}"#.into(), vec![], "invalid (the server would answer every document 422)"),
        (r#"{"questions": {"a": {"type": "score", "criteria": [1,2,3,4,5,6,7,8,9,10,11]}}}"#.into(), vec![], "422"),
        (format!(r#"{{"model": "other", "questions": {noul}}}"#), vec![], "is not served"),
        (format!(r#"{{"model": 7, "questions": {noul}}}"#), vec![], "valid string"),
        (format!(r#"{{"questions": {noul}}}"#), vec!["--query", "debias=x"], "not an integer"),
        (format!(r#"{{"questions": {noul}}}"#), vec!["--max-branch-tokens", "40"], "no document fits the limits"),
    ];
    for (q, extra, want) in &cases {
        let err = run(q, extra);
        assert!(err.contains(want), "{q}: wanted {want:?} in:\n{err}");
    }
    // Argument combinations.
    let q = dir.join("ok.json");
    std::fs::write(&q, format!(r#"{{"questions": {noul}}}"#)).unwrap();
    for a in [
        vec!["--questions", q.to_str().unwrap(), "--output", out.to_str().unwrap()],
        vec!["--docs", docs.to_str().unwrap(), "--output", out.to_str().unwrap()],
        vec!["--input", docs.to_str().unwrap(), "--questions", q.to_str().unwrap(), "--docs", docs.to_str().unwrap(), "--output", out.to_str().unwrap()],
        vec!["--questions", q.to_str().unwrap(), "--docs", dir.join("missing").to_str().unwrap(), "--output", out.to_str().unwrap()],
    ] {
        let mut args = vec!["--engine", "mock"];
        args.extend(a.iter().copied());
        let (code, summary, err) = bulk(&args);
        assert!(code != 0 && summary.is_none() && !out.exists(), "{a:?} accepted:\n{err}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
