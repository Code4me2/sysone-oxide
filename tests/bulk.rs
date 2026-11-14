//! `sysone bulk` without a GPU (mock engine; needs the checkpoint's tokenizer and chat template, like the server
//! tests, from `$SYSONE_MODEL_DIR`), through the real binary:
//!
//! - `results_equal_the_servers_responses`: a mixed corpus (the 298-case render corpus with its debias values,
//!   the raw schema bodies incl. invalid JSON, synthetic documents, debug requests) through bulk (three worker
//!   processes) and through the HTTP server: every id gets exactly one line, whose status and body equal the
//!   server's byte for byte apart from `qev.timing`; the results are keyed by id although the order differs.
//! - `resumes_after_a_crash`: SIGKILL in the middle of a run plus a torn last line; the rerun cuts the torn line,
//!   skips the ids already written, and the union equals a clean run.
//! - `packing_is_deterministic`: the same input gives the same sequence of steps (ids and buckets), with one
//!   worker or three.
//! - `slow_and_small_workers_take_less_and_crashed_workers_hand_back_their_work`: the multi-GPU logic on simulated
//!   workers of different speed and capacity, one of which dies mid-run.
//! - `gpu_workers_follow_gpu_sh_locks`: workers start through scripts/gpu.sh against a fake 3-GPU nvidia-smi:
//!   each gets its own GPU and lock; a GPU whose lock is held elsewhere gets no work; a GPU twice is refused
//!   outside gpu.sh and runs two engines inside it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sysone::api::client::Client;
use sysone::api::server::{self, AppState, ServerInfo, ServerOptions};
use sysone::contract::{DecisionEngine, MockEngine};
use sysone::frontend::pyjson::{self, Json};
use sysone::frontend::{Frontend, Limits};
use sysone::sched::{EngineFactory, SchedConfig};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

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
    let d = std::env::temp_dir().join(format!("sysone-bulk-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn parse(b: &[u8]) -> Json {
    pyjson::parse(b).unwrap_or_else(|e| panic!("not JSON ({e:?}): {}", String::from_utf8_lossy(b)))
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

/// Result lines by id: (status, untimed body), and the ids in file order.
fn results(path: &Path) -> (BTreeMap<String, (u16, String)>, Vec<String>) {
    let text = std::fs::read(path).unwrap();
    let mut m = BTreeMap::new();
    let mut order = Vec::new();
    for l in text.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        let v = parse(l);
        let id = v.get("id").unwrap().dumps_compact().unwrap();
        let status = match v.get("status") {
            Some(Json::Int(s)) => s.parse().unwrap(),
            other => panic!("status {other:?}"),
        };
        assert!(m.insert(id.clone(), (status, untimed(v.get("body").unwrap()))).is_none(), "id {id} twice in {}", path.display());
        order.push(id);
    }
    (m, order)
}

/// Run `sysone bulk ARGS` (with --model); returns (exit code, summary JSON, stderr).
/// These tests drive their own (fake-GPU) gpu.sh locks: drop what an outer scripts/gpu.sh (the whole suite run
/// under the real lock) exported, or a nested gpu.sh refuses locks its "parent" does not hold.
fn clear_outer_gpu_sh(c: &mut Command) {
    for k in ["SYSONE_GPU_HELD", "SYSONE_GPU_LOCKED", "SYSONE_GPU", "SYSONE_GPU_LOCK", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"] {
        c.env_remove(k);
    }
}

fn bulk(args: &[&str], envs: &[(&str, &str)]) -> (i32, Option<Json>, String) {
    let model = model_dir();
    let mut c = Command::new(bin());
    c.arg("bulk").arg("--model").arg(&model).arg("--progress-s").arg("0").args(args);
    clear_outer_gpu_sh(&mut c);
    for (k, v) in envs {
        c.env(k, v);
    }
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

/// A synthetic corpus of `n` documents (`sysone bulk gen` with the given word range).
fn corpus(dir: &Path, n: usize, min_words: usize, max_words: usize) -> PathBuf {
    let p = dir.join("corpus.jsonl");
    let st = Command::new(bin())
        .args(["bulk", "gen", "--docs", &n.to_string(), "--min-words", &min_words.to_string(), "--max-words", &max_words.to_string()])
        .arg("--output")
        .arg(&p)
        .status()
        .unwrap();
    assert!(st.success());
    p
}

/// The mixed corpus: the render corpus (with its debias), the raw schema bodies, synthetic documents (some with
/// `debug=1`). Returns (path, jobs as (id, query, raw body)).
fn mixed_corpus(dir: &Path) -> (PathBuf, Vec<(String, String, Vec<u8>)>) {
    let mut jobs = Vec::new();
    let mut lines = Vec::new();
    for l in std::fs::read_to_string(repo().join("reference/corpus/requests.jsonl")).unwrap().lines() {
        let v = parse(l.as_bytes());
        let id = v.get("id").unwrap().as_str().unwrap().to_string();
        let debias = num(v.get("debias")) as i64;
        let body = v.get("body").unwrap().dumps_compact().unwrap();
        let query = format!("debias={debias}");
        lines.push(format!("{{\"id\": {}, \"body\": {body}, \"query\": \"{query}\"}}", Json::str(id.clone()).dumps_compact().unwrap()));
        jobs.push((id, query, body.into_bytes()));
    }
    for l in std::fs::read_to_string(repo().join("tests/golden/dev/assemble/schema.jsonl")).unwrap().lines() {
        let v = parse(l.as_bytes());
        let id = v.get("id").unwrap().as_str().unwrap().to_string();
        let raw = v.get("body").unwrap().as_str().unwrap().to_string();
        lines.push(format!(
            "{{\"id\": {}, \"body_text\": {}}}",
            Json::str(id.clone()).dumps_compact().unwrap(),
            Json::str(raw.clone()).dumps_compact().unwrap()
        ));
        jobs.push((id, String::new(), raw.into_bytes()));
    }
    for i in 0..60 {
        let body = sysone::bulk::synth::body(i, 50 + (i * 97) % 1500, 1 + i % 9);
        // Integer ids, some debug requests (allowed_token_mass from the mock), one with a query of its own.
        let query = match i % 10 {
            0 => "debug=1&debias=2".to_string(),
            5 => "debias=3".to_string(),
            _ => String::new(),
        };
        let id = 900000 + i;
        if query.is_empty() {
            lines.push(format!("{{\"id\": {id}, \"body\": {body}}}"));
        } else {
            lines.push(format!("{{\"id\": {id}, \"body\": {body}, \"query\": \"{query}\"}}"));
        }
        jobs.push((id.to_string(), query, body.into_bytes()));
    }
    // Bad lines: reported, not processed.
    lines.insert(5, "not json".into());
    lines.insert(9, "{\"body\": {}}".into());
    lines.insert(11, String::new());
    let p = dir.join("mixed.jsonl");
    std::fs::write(&p, lines.join("\n") + "\n").unwrap();
    (p, jobs)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn results_equal_the_servers_responses() {
    let dir = tmp("equal");
    let (input, jobs) = mixed_corpus(&dir);
    let out = dir.join("out.jsonl");
    let (code, summary, err) = tokio::task::spawn_blocking({
        let (input, out) = (input.clone(), out.clone());
        move || {
            bulk(
                &[
                    "--engine", "mock", "--workers", "process", "--mock-workers", "3", "--max-step-tokens", "65536", "--window", "48",
                    "--input", input.to_str().unwrap(), "--output", out.to_str().unwrap(),
                ],
                &[],
            )
        }
    })
    .await
    .unwrap();
    assert_eq!(code, 0, "{err}");
    let s = summary.unwrap();
    assert_eq!((num(s.get("bad_lines")), num(s.get("jobs"))), (2.0, jobs.len() as f64), "{err}");

    // The server, on the same frontend and limits, with the mock engine.
    let limits = Limits::default();
    let frontend = Arc::new(Frontend::load(&model_dir(), limits).unwrap());
    let max_tokens = limits.max_request_tokens;
    let engine: EngineFactory = Box::new(move || Ok(Box::new(MockEngine { max_tokens }) as Box<dyn DecisionEngine>));
    let info = ServerInfo::new("elephant-1", "qwen3_5");
    let state =
        AppState::start_with(frontend, info, SchedConfig::default(), engine, ServerOptions { cpu_workers: 4, ..Default::default() }).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (stop, rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        server::serve(listener, state, async {
            let _ = rx.await;
        })
        .await
        .unwrap()
    });
    let mut c = Client::connect(&addr).await.unwrap();
    let (got, order) = results(&out);
    assert_eq!(got.len(), jobs.len());
    let mut statuses = BTreeMap::<u16, usize>::new();
    let mut mismatches = Vec::new();
    for (id, query, body) in &jobs {
        let path = if query.is_empty() { "/v1/systemone".to_string() } else { format!("/v1/systemone?{query}") };
        let r = c.post(&path, body.clone()).await.unwrap();
        let key = if id.parse::<u64>().is_ok() { id.clone() } else { Json::str(id.clone()).dumps_compact().unwrap() };
        let (status, mine) = got.get(&key).unwrap_or_else(|| panic!("no result for {key}"));
        *statuses.entry(*status).or_default() += 1;
        let theirs = untimed(&parse(&r.body));
        if r.status.as_u16() != *status || theirs != *mine {
            mismatches.push(format!("{key}: server {} {:.300}\n  bulk {status} {:.300}", r.status, theirs, mine));
        }
    }
    let _ = stop.send(());
    task.await.unwrap();
    eprintln!("statuses: {statuses:?}");
    assert!(mismatches.is_empty(), "{} mismatches:\n{}", mismatches.len(), mismatches.join("\n"));
    assert!(statuses[&200] > 200 && statuses[&422] > 40 && statuses.get(&404).copied().unwrap_or(0) >= 4, "{statuses:?}");
    // Results are keyed by id: the file order is not the input order (errors first, then by packing).
    let input_order: Vec<&String> = jobs.iter().map(|j| &j.0).collect();
    let file_order: Vec<String> =
        order.iter().map(|k| pyjson::parse(k.as_bytes()).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or(k.clone())).collect();
    assert_ne!(file_order.iter().collect::<Vec<_>>(), input_order);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resumes_after_a_crash() {
    let dir = tmp("resume");
    let input = corpus(&dir, 150, 100, 2000);
    let (clean, crashed) = (dir.join("clean.jsonl"), dir.join("crashed.jsonl"));
    let args = |out: &Path| -> Vec<String> {
        ["--engine", "mock", "--max-step-tokens", "8192", "--window", "32", "--input", input.to_str().unwrap(), "--output", out.to_str().unwrap()]
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    let a: Vec<String> = args(&clean);
    let (code, _, err) = bulk(&a.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    assert_eq!(code, 0, "{err}");

    // A slow run (20 us per bucket token: ~0.16 s per 8k step), killed after some results.
    let mut child = Command::new(bin())
        .arg("bulk")
        .arg("--model")
        .arg(model_dir())
        .args(args(&crashed))
        .args(["--mock-ns-per-token", "20000", "--progress-s", "0"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
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
    let before = results(&crashed).0.len();
    assert!(before >= 20 && before < 150, "{before}");
    // A torn last line, as a crash in the middle of a write leaves it.
    let mut f = std::fs::OpenOptions::new().append(true).open(&crashed).unwrap();
    std::io::Write::write_all(&mut f, b"{\"id\":\"doc-000149\",\"status\":200,\"bo").unwrap();
    drop(f);

    let a: Vec<String> = args(&crashed);
    let (code, summary, err) = bulk(&a.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    assert_eq!(code, 0, "{err}");
    let s = summary.unwrap();
    assert_eq!(num(s.get("resumed_lines")) as usize, before);
    assert_eq!(num(s.get("skipped_done")) as usize, before);
    assert!(num(s.get("truncated_partial_bytes")) >= 35.0);
    assert_eq!(num(s.get("jobs")) as usize, 150 - before);
    assert_eq!(results(&crashed).0, results(&clean).0);
    // Nothing left to do: a third run writes nothing.
    let (code, summary, _) = bulk(&a.iter().map(String::as_str).collect::<Vec<_>>(), &[]);
    assert_eq!((code, num(summary.unwrap().get("jobs"))), (0, 0.0));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The steps of a trace file: (ids, bucket) in step order.
fn steps(trace: &Path) -> Vec<(String, u64, String)> {
    std::fs::read_to_string(trace)
        .unwrap()
        .lines()
        .map(|l| {
            let v = parse(l.as_bytes());
            (
                v.get("ids").unwrap().dumps_compact().unwrap(),
                num(v.get("bucket")) as u64,
                v.get("worker").unwrap().as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[test]
fn packing_is_deterministic() {
    let dir = tmp("determinism");
    let input = corpus(&dir, 200, 100, 3000);
    let run = |name: &str, workers: &str| -> (BTreeMap<String, (u16, String)>, Vec<(String, u64, String)>) {
        let (out, trace) = (dir.join(format!("{name}.jsonl")), dir.join(format!("{name}.trace")));
        let (code, summary, err) = bulk(
            &[
                "--engine", "mock", "--mock-workers", workers, "--max-step-tokens", "16384", "--fine-ladder", "--window", "40",
                "--input", input.to_str().unwrap(), "--output", out.to_str().unwrap(), "--trace", trace.to_str().unwrap(),
            ],
            &[],
        );
        assert_eq!(code, 0, "{err}");
        // Steps are full: under 2% padding overall (the server's FIFO batcher pads ~10% on such a mix).
        assert!(num(summary.unwrap().get("padding_frac")) < 0.02);
        (results(&out).0, steps(&trace))
    };
    let (r1, s1) = run("a", "1");
    let (r2, s2) = run("b", "1");
    let (r3, s3) = run("c", "3");
    assert_eq!(s1, s2);
    let strip = |s: &[(String, u64, String)]| s.iter().map(|(i, b, _)| (i.clone(), *b)).collect::<Vec<_>>();
    assert_eq!(strip(&s1), strip(&s3), "the step sequence does not depend on the number of (equal) workers");
    assert!(s3.iter().map(|s| &s.2).collect::<std::collections::HashSet<_>>().len() > 1, "three workers shared the steps");
    assert_eq!(r1, r2);
    assert_eq!(r1, r3);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn slow_and_small_workers_take_less_and_crashed_workers_hand_back_their_work() {
    let dir = tmp("balance");
    let input = corpus(&dir, 240, 100, 3000);
    let (out, trace) = (dir.join("out.jsonl"), dir.join("trace"));
    // Workers 0 and 1: 16k steps at 2 us/token; worker 2: 4k steps at 8 us/token (slower and smaller).
    let (code, s, err) = bulk(
        &[
            "--engine", "mock", "--workers", "process", "--mock-workers", "3", "--mock-ns-per-token", "2000,2000,8000",
            "--mock-max-tokens", "16384,16384,4096", "--window", "64", "--input", input.to_str().unwrap(),
            "--output", out.to_str().unwrap(), "--trace", trace.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{err}");
    let s = s.unwrap();
    let ws = s.get("workers").unwrap().as_array().unwrap();
    let share: Vec<f64> = ws.iter().map(|w| num(w.get("token_share"))).collect();
    eprintln!("token shares {share:?}");
    assert!(share[2] < 0.5 * share[0].min(share[1]), "{share:?}");
    for (_, bucket, worker) in steps(&trace) {
        if worker == "mock2" {
            assert!(bucket <= 4096, "a {bucket}-token step on the 4k worker");
        }
    }
    assert_eq!(results(&out).0.len(), 240);
    // Requests over 4,096 tokens only ever ran on workers 0 and 1 (the trace has no larger step on mock2).

    // Worker 1 dies after 3 steps: its queued steps go back to the window and the others finish them.
    let out2 = dir.join("out2.jsonl");
    let (code, s, err) = bulk(
        &[
            "--engine", "mock", "--workers", "process", "--mock-workers", "3", "--mock-ns-per-token", "3000",
            "--mock-exit-after", "0,3,0", "--max-step-tokens", "8192", "--window", "64", "--input", input.to_str().unwrap(),
            "--output", out2.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{err}");
    let s = s.unwrap();
    let ws = s.get("workers").unwrap().as_array().unwrap();
    assert!(ws[1].get("error").is_some_and(|e| !e.is_null()), "{err}");
    assert_eq!(num(ws[1].get("steps")), 3.0);
    assert_eq!(num(s.get("failed")), 0.0);
    assert_eq!(results(&out2).0, results(&out).0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gpu_workers_follow_gpu_sh_locks() {
    let dir = tmp("locks");
    let locks = dir.join("locks");
    std::fs::create_dir_all(&locks).unwrap();
    let input = corpus(&dir, 60, 100, 1500);
    let smi = repo().join("scripts/tests/fake-nvidia-smi");
    let sh = repo().join("scripts/gpu.sh");
    let envs: Vec<(String, String)> = vec![
        ("SYSONE_NVIDIA_SMI".into(), smi.display().to_string()),
        ("SYSONE_GPU_LOCK_DIR".into(), locks.display().to_string()),
        ("FAKE_GPUS".into(), "3".into()),
    ];
    let envr: Vec<(&str, &str)> = envs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let common = |out: &Path, gpus: &str| -> Vec<String> {
        [
            "--engine", "mock", "--gpus", gpus, "--gpu-sh", sh.to_str().unwrap(), "--mock-ns-per-token", "2000", "--max-step-tokens",
            "4096", "--window", "16", "--input", input.to_str().unwrap(), "--output", out.to_str().unwrap(),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    };
    let as_str = |v: &[String]| v.iter().map(String::clone).collect::<Vec<_>>();

    // GPU 1's lock is held by someone else: its worker waits, gets no work, and is stopped at the end; the lock
    // holder is untouched.
    let mut holder = Command::new("flock").arg(locks.join("sysone-gpu1.lock")).args(["sleep", "60"]).spawn().unwrap();
    std::thread::sleep(Duration::from_millis(200));
    let out = dir.join("a.jsonl");
    let args = as_str(&common(&out, "0,1,2"));
    let (code, s, err) = bulk(&args.iter().map(String::as_str).collect::<Vec<_>>(), &envr);
    assert_eq!(code, 0, "{err}");
    let s = s.unwrap();
    let ws = s.get("workers").unwrap().as_array().unwrap();
    let info = |w: &Json, k: &str| -> String {
        w.get("info").unwrap().as_array().unwrap().iter().find_map(|p| {
            let p = p.as_array().unwrap();
            (p[0].as_str() == Some(k)).then(|| p[1].as_str().unwrap().to_string())
        }).unwrap_or_default()
    };
    assert_eq!(info(&ws[0], "cuda_visible_devices"), "0");
    assert_eq!(info(&ws[0], "sysone_gpu_held"), locks.join("sysone-gpu0.lock").display().to_string());
    assert_eq!(info(&ws[2], "cuda_visible_devices"), "2");
    assert_eq!(num(ws[1].get("steps")), 0.0);
    assert!(ws[1].get("error").is_some_and(|e| !e.is_null()));
    assert_eq!(results(&out).0.len(), 60);
    assert!(holder.try_wait().unwrap().is_none(), "the lock holder was disturbed");
    holder.kill().unwrap();
    holder.wait().unwrap();

    // The same GPU twice: refused outside gpu.sh (the second worker would wait for the first one's lock) ...
    let out = dir.join("b.jsonl");
    let args = as_str(&common(&out, "0,0"));
    let (code, _, err) = bulk(&args.iter().map(String::as_str).collect::<Vec<_>>(), &envr);
    assert_ne!(code, 0);
    assert!(err.contains("names a GPU twice"), "{err}");
    // ... and two engines on it under gpu.sh, which holds the lock for both (nested gpu.sh runs them unlocked).
    let mut c = Command::new(&sh);
    clear_outer_gpu_sh(&mut c);
    c.args(["--gpu", "0", bin(), "bulk", "--model"]).arg(model_dir()).args(["--progress-s", "0"]).args(&args);
    for (k, v) in &envr {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "{err}");
    let s = parse(String::from_utf8_lossy(&o.stdout).lines().last().unwrap().as_bytes());
    let ws = s.get("workers").unwrap().as_array().unwrap();
    assert_eq!(ws.len(), 2);
    for w in ws {
        assert!(w.get("error").unwrap().is_null(), "{err}");
        assert_eq!(info(w, "sysone_gpu_held"), locks.join("sysone-gpu0.lock").display().to_string());
    }
    // Under gpu.sh --gpu 0, a worker for GPU 1 is refused by its nested gpu.sh (taking more locks could deadlock).
    let out = dir.join("c.jsonl");
    let args = as_str(&common(&out, "0,1"));
    let mut c = Command::new(&sh);
    clear_outer_gpu_sh(&mut c);
    c.args(["--gpu", "0", bin(), "bulk", "--model"]).arg(model_dir()).args(["--progress-s", "0"]).args(&args);
    for (k, v) in &envr {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "{err}");
    assert!(err.contains("nested gpu.sh needs lock"), "{err}");
    let s = parse(String::from_utf8_lossy(&o.stdout).lines().last().unwrap().as_bytes());
    let ws = s.get("workers").unwrap().as_array().unwrap();
    assert_eq!(num(ws[1].get("steps")), 0.0);
    assert_eq!(results(&out).0.len(), 60);
    let _ = std::fs::remove_dir_all(&dir);
}
