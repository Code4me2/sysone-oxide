//! Readers for the W1b model goldens (DESIGN §9.1): the 52-request model corpus, per-run results (slot logits
//! and probabilities per branch) and the per-layer hidden-state dumps. Used by the engine's end-to-end gates
//! (tests/engine_*.rs) and the forward benchmark. Probabilities are compared per option as
//! `softmax(slot logits)` at T = 1 (`tests/tolerances.toml`).

use crate::contract::DecisionBatch;
use crate::frontend::{Frontend, Plan};
use anyhow::{Context as _, Result, bail, ensure};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One case of `reference/corpus_model/requests.jsonl`.
#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub debias: i64,
    /// The request body, keys in their original order.
    pub body: String,
    pub dump_hidden: bool,
    pub expect_ok: bool,
}

pub fn model_corpus(repo: &Path) -> Result<Vec<Case>> {
    let path = repo.join("reference/corpus_model/requests.jsonl");
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for line in text.split('\n').filter(|l| !l.trim().is_empty()) {
        let j = crate::frontend::pyjson::parse(line.as_bytes()).map_err(|e| anyhow::anyhow!("corpus line: {e:?}"))?;
        let s = |k: &str| j.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let debias = match j.get("debias") {
            Some(crate::frontend::pyjson::Json::Int(n)) => n.parse()?,
            _ => 1,
        };
        let body = j.get("request").context("case without request")?.dumps_compact().map_err(anyhow::Error::msg)?;
        out.push(Case {
            id: s("id").context("case without id")?,
            debias,
            body,
            dump_hidden: matches!(j.get("dump_hidden"), Some(crate::frontend::pyjson::Json::Bool(true))),
            expect_ok: s("expect").as_deref() == Some("ok"),
        });
    }
    Ok(out)
}

/// One branch of a golden result.
#[derive(Clone, Debug)]
pub struct GoldenBranch {
    pub qid: String,
    pub n_slots: usize,
    pub logits: Vec<f64>,
    pub probs: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct GoldenResult {
    pub id: String,
    pub status: u64,
    pub prefix_tokens: Option<usize>,
    pub branches: Vec<GoldenBranch>,
    /// The response body qev sent.
    pub response: Value,
}

/// `tests/golden/dev/model/runs/{run}.json`, by request id.
pub fn golden_run(repo: &Path, run: &str) -> Result<HashMap<String, GoldenResult>> {
    let path = repo.join(format!("tests/golden/dev/model/runs/{run}.json"));
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?)?;
    let mut out = HashMap::new();
    for r in v["results"].as_array().context("no results")? {
        let f = |x: &Value| x.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>()).unwrap_or_default();
        let branches = r["branches"]
            .as_array()
            .map(|bs| {
                bs.iter()
                    .map(|b| GoldenBranch {
                        qid: b["qid"].as_str().unwrap_or_default().to_string(),
                        n_slots: b["n_slots"].as_u64().unwrap_or(0) as usize,
                        logits: f(&b["logits"]),
                        probs: f(&b["probs"]),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let id = r["id"].as_str().context("result without id")?.to_string();
        out.insert(id.clone(), GoldenResult {
            id,
            status: r["status"].as_u64().unwrap_or(0),
            prefix_tokens: r["prefix_tokens"].as_u64().map(|x| x as usize),
            branches,
            response: r["response"].clone(),
        });
    }
    Ok(out)
}

/// Softmax at T = 1, in f64.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = logits.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

/// A planned corpus request.
pub struct Planned {
    pub case: Case,
    pub plan: Plan,
}

/// Plan every corpus case that the reference answered with 200.
pub fn plan_corpus(fe: &Frontend, cases: &[Case], ok: impl Fn(&str) -> bool) -> Result<Vec<Planned>> {
    let mut out = Vec::new();
    for c in cases {
        if !ok(&c.id) {
            continue;
        }
        let req = crate::api::schema::SystemOneRequest::from_body(c.body.as_bytes())
            .map_err(|e| anyhow::anyhow!("{}: request rejected: {e:?}", c.id))?;
        let plan = fe.plan(&req, c.debias).map_err(|e| anyhow::anyhow!("{}: {e}", c.id))?;
        out.push(Planned { case: c.clone(), plan });
    }
    Ok(out)
}

/// A batch of one planned request.
pub fn batch_of(plans: &[&Plan]) -> Result<(DecisionBatch, Vec<usize>)> {
    let mut b = DecisionBatch::default();
    let mut first = Vec::new();
    for p in plans {
        first.push(p.push_into(&mut b)?);
    }
    b.validate()?;
    Ok((b, first))
}

/// A safetensors file read whole (hidden dumps are a few MB).
pub struct SafeTensors {
    pub metadata: HashMap<String, String>,
    tensors: HashMap<String, (String, Vec<usize>, Vec<u8>)>,
}

impl SafeTensors {
    pub fn read(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        ensure!(bytes.len() >= 8, "{}: too short", path.display());
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n])?;
        let data = &bytes[8 + n..];
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for (k, v) in hdr {
            if k == "__metadata__" {
                for (mk, mv) in v.as_object().into_iter().flatten() {
                    metadata.insert(mk.clone(), mv.as_str().unwrap_or_default().to_string());
                }
                continue;
            }
            let dtype = v["dtype"].as_str().context("dtype")?.to_string();
            let shape = v["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect();
            let off = v["data_offsets"].as_array().context("offsets")?;
            let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
            ensure!(b <= data.len() && a <= b, "{k}: offsets out of range");
            tensors.insert(k, (dtype, shape, data[a..b].to_vec()));
        }
        Ok(Self { metadata, tensors })
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.1.as_slice())
    }

    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (dt, _, b) = self.tensors.get(name).with_context(|| format!("no tensor {name}"))?;
        ensure!(dt == "F32", "{name} is {dt}");
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn i64(&self, name: &str) -> Result<Vec<i64>> {
        let (dt, _, b) = self.tensors.get(name).with_context(|| format!("no tensor {name}"))?;
        ensure!(dt == "I64", "{name} is {dt}");
        Ok(b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// Path of a hidden dump listed in `tests/golden/dev/model/manifest.json`.
pub fn hidden_dump(repo: &Path, run: &str, id: &str) -> Result<PathBuf> {
    let m: Value = serde_json::from_str(&std::fs::read_to_string(repo.join("tests/golden/dev/model/manifest.json"))?)?;
    let root = PathBuf::from(m["root"].as_str().context("manifest root")?);
    let rel = format!("{run}/hidden_{id}.safetensors");
    if m["files"].get(&rel).is_none() {
        bail!("{rel} is not in the manifest");
    }
    Ok(root.join(rel))
}

/// Ids of the requests with hidden dumps.
pub fn hidden_ids(repo: &Path, run: &str) -> Result<Vec<String>> {
    let m: Value = serde_json::from_str(&std::fs::read_to_string(repo.join("tests/golden/dev/model/manifest.json"))?)?;
    let mut ids: Vec<String> = m["files"]
        .as_object()
        .context("files")?
        .keys()
        .filter_map(|k| k.strip_prefix(&format!("{run}/hidden_")).and_then(|r| r.strip_suffix(".safetensors")))
        .map(str::to_string)
        .collect();
    ids.sort();
    Ok(ids)
}

/// Path of a dense hidden dump (gate (a), `reference/hidden_dense.py`) listed in
/// `tests/golden/dev/model/hidden_dense_manifest.json`; `run` is e.g. `fp32_served` or `bf16_served`.
pub fn hidden_dense_dump(repo: &Path, run: &str, id: &str) -> Result<PathBuf> {
    let m = dense_manifest(repo)?;
    let root = PathBuf::from(m["root"].as_str().context("manifest root")?);
    let rel = format!("{run}/hidden_{id}.safetensors");
    let entry = m["files"].get(&rel).with_context(|| format!("{rel} is not in the dense manifest"))?;
    let path = root.join(&rel);
    let len = std::fs::metadata(&path).with_context(|| format!("{}", path.display()))?.len();
    ensure!(Some(len) == entry["bytes"].as_u64(), "{}: size differs from the manifest", path.display());
    Ok(path)
}

fn dense_manifest(repo: &Path) -> Result<Value> {
    let path = repo.join("tests/golden/dev/model/hidden_dense_manifest.json");
    Ok(serde_json::from_str(&std::fs::read_to_string(&path).with_context(|| format!("{}", path.display()))?)?)
}

/// Ids of the requests with dense hidden dumps for `run`, sorted (the gate's row order).
pub fn hidden_dense_ids(repo: &Path, run: &str) -> Result<Vec<String>> {
    let m = dense_manifest(repo)?;
    let mut ids: Vec<String> = m["files"]
        .as_object()
        .context("files")?
        .keys()
        .filter_map(|k| k.strip_prefix(&format!("{run}/hidden_")).and_then(|r| r.strip_suffix(".safetensors")))
        .map(str::to_string)
        .collect();
    ids.sort();
    Ok(ids)
}

/// The dense row set's prefix rows (`reference/hidden_dense.py dense_prefix_rows`): every `max(1, P / 1024)`-th
/// row plus the last two rows.
pub fn dense_prefix_rows(p: usize) -> Vec<usize> {
    let s = (p / 1024).max(1);
    let mut rows: Vec<usize> = (0..p).step_by(s).collect();
    for r in [p.saturating_sub(2), p.saturating_sub(1)] {
        if r < p && !rows.contains(&r) {
            rows.push(r);
        }
    }
    rows.sort_unstable();
    rows
}
