//! `sysone bulk --questions --docs` with the bf16 GPU engine (needs the checkpoint and the GPU; run it under the
//! GPU lock, which the bulk workers' nested gpu.sh then reuses):
//!
//! ```text
//! scripts/gpu.sh cargo test --release --test bulk_qset_gpu -- --nocapture
//! ```
//!
//! A question set over documents (synthetic text of 300-3,000 words, some with their own query or overrides), and
//! the equivalent full-body jobs, each through bulk with the same window and step size, one after the other: the
//! plans are identical, so the steps are identical, and every result line must be byte for byte the same apart
//! from `qev.timing` (no tolerance: the same batches on the same engine).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use sysone::frontend::pyjson::{self, Json};

fn model_dir() -> PathBuf {
    std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"))
}

fn parse(b: &[u8]) -> Json {
    pyjson::parse(b).unwrap_or_else(|e| panic!("not JSON ({e:?}): {}", String::from_utf8_lossy(b)))
}

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

fn results(path: &Path) -> BTreeMap<String, (u16, String)> {
    let mut m = BTreeMap::new();
    for l in std::fs::read(path).unwrap().split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        let v = parse(l);
        let id = v.get("id").unwrap().dumps_compact().unwrap();
        let status = match v.get("status") {
            Some(Json::Int(s)) => s.parse().unwrap(),
            other => panic!("{other:?}"),
        };
        assert!(m.insert(id.clone(), (status, untimed(v.get("body").unwrap()))).is_none(), "{id} twice");
    }
    m
}

fn bulk(out: &Path, input: &[&str]) -> Json {
    let _ = std::fs::remove_file(out);
    let o = Command::new(env!("CARGO_BIN_EXE_sysone"))
        .arg("bulk")
        .arg("--model")
        .arg(model_dir())
        .args(["--engine", "bf16", "--max-step-tokens", "16384", "--window", "64", "--progress-s", "0"])
        .args(input)
        .arg("--output")
        .arg(out)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "bulk failed:\n{err}");
    let summary = parse(String::from_utf8_lossy(&o.stdout).lines().last().unwrap().as_bytes());
    println!("{}", summary.dumps_compact().unwrap());
    summary
}

#[test]
fn question_set_equals_full_body_jobs_on_the_gpu() {
    assert!(model_dir().join("model.safetensors").exists() || model_dir().join("model.safetensors.index.json").exists(), "no checkpoint");
    let dir = std::env::temp_dir().join(format!("sysone-qset-gpu-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let qtext = r#"{"note": "shared", "questions": {
        "late": {"type": "noul", "instructions": "Is the customer waiting on a late shipment?"},
        "route": {"type": "choice", "instructions": "Which team should handle it?", "criteria": {"billing": "Billing team", "tech": "Technical support", "ship": "Shipping", "fraud": "Fraud review", "other": null}},
        "urgency": {"type": "score", "instructions": "How urgent is it?", "criteria": ["low", "medium", "high", "critical"]},
        "angry": {"type": "noul", "instructions": "Is the customer angry?", "criteria": {"true": "angry or hostile", "false": "calm"}}
    }}"#;
    let qs = parse(qtext.as_bytes());
    let qfile = dir.join("q.json");
    std::fs::write(&qfile, qtext).unwrap();
    let spec = sysone::bulk::synth::CorpusSpec { docs: 24, min_words: 300, max_words: 3000, min_questions: 1, max_questions: 1, seed: 11 };
    let (mut doc_lines, mut job_lines) = (Vec::new(), Vec::new());
    for i in 0..spec.docs {
        let (w, _) = sysone::bulk::synth::doc_shape(&spec, i);
        let state = Json::str(sysone::bulk::synth::words(i, w));
        let query = match i % 6 {
            1 => Some("debias=2"),
            4 => Some("debug=1&debias=3"),
            _ => None,
        };
        // Two documents override fields: one model alias (planned from its full body), one ignored key.
        let overrides = match i {
            5 => Some(("model", Json::str("qev-latest"))),
            9 => Some(("note", Json::str("mine"))),
            _ => None,
        };
        let mut body = vec![("state".to_string(), state.clone())];
        for (k, v) in qs.as_object().unwrap() {
            body.push((k.clone(), v.clone()));
        }
        let id = Json::str(format!("doc-{i:03}")).dumps_compact().unwrap();
        let mut d = format!("{{\"id\": {id}, \"state\": {}", state.dumps_compact().unwrap());
        if let Some((k, v)) = &overrides {
            d.push_str(&format!(", \"overrides\": {{\"{k}\": {}}}", v.dumps_compact().unwrap()));
            match body.iter_mut().find(|(key, _)| key == k) {
                Some(slot) => slot.1 = v.clone(),
                None => body.push((k.to_string(), v.clone())),
            }
        }
        let mut j = format!("{{\"id\": {id}, \"body\": {}", Json::Object(body).dumps_compact().unwrap());
        if let Some(q) = query {
            d.push_str(&format!(", \"query\": \"{q}\""));
            j.push_str(&format!(", \"query\": \"{q}\""));
        }
        doc_lines.push(d + "}");
        job_lines.push(j + "}");
    }
    let (dfile, jfile) = (dir.join("docs.jsonl"), dir.join("jobs.jsonl"));
    std::fs::write(&dfile, doc_lines.join("\n") + "\n").unwrap();
    std::fs::write(&jfile, job_lines.join("\n") + "\n").unwrap();

    let (out_d, out_j) = (dir.join("docs-out.jsonl"), dir.join("jobs-out.jsonl"));
    let sd = bulk(&out_d, &["--questions", qfile.to_str().unwrap(), "--docs", dfile.to_str().unwrap()]);
    let sj = bulk(&out_j, &["--input", jfile.to_str().unwrap()]);
    let (rd, rj) = (results(&out_d), results(&out_j));
    assert_eq!(rd.len(), spec.docs);
    assert!(rd.values().all(|(s, _)| *s == 200), "every document is answered");
    for (id, r) in &rj {
        assert_eq!(rd.get(id), Some(r), "id {id}");
    }
    assert_eq!(rd, rj);
    assert_eq!(sd.get("steps"), sj.get("steps"));
    assert_eq!(sd.get("packed_tokens"), sj.get("packed_tokens"));
    let q = sd.get("question_set").unwrap();
    assert_eq!((q.get("docs_full_body"), q.get("docs_prepared")), (Some(&Json::int(1)), Some(&Json::int(spec.docs as i64 - 1))));
    println!("{} documents: question-set results identical to the full-body jobs", rd.len());
    let _ = std::fs::remove_dir_all(&dir);
}
