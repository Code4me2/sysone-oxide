//! `sysone bulk compare A B`: compare two result files (bulk output, or `loadgen --corpus --responses`) by id.
//!
//! Both must hold the same ids. For each id the statuses must match and the bodies must be equal as JSON apart
//! from `qev.timing` and from floats under `answers` and `qev.diagnostics.*.slot_logits`: probabilities (`noul`,
//! `probabilities.*`) are summarised as |dp|, slot logits as |dlogit|, derived floats (`score`, `confidence`)
//! separately; a different `choice` is counted with its top-1 margin in B. The report is one JSON line; the
//! exit code is 0 when ids, statuses and every non-float field agree (thresholds are the caller's: e.g. the
//! `[batch_invariance]` limits for two packings of the same requests).

use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;

use crate::frontend::pyjson::{self, Json};

fn num(j: &Json) -> Option<f64> {
    match j {
        Json::Float(x) => Some(*x),
        Json::Int(s) => s.parse().ok(),
        _ => None,
    }
}

/// id (compact JSON) → (status, body).
pub fn load(path: &Path) -> Result<BTreeMap<String, (u16, Json)>> {
    let text = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut m = BTreeMap::new();
    for (i, l) in text.split(|&b| b == b'\n').enumerate().filter(|(_, l)| !l.is_empty()) {
        let v = pyjson::parse(l).map_err(|e| anyhow::anyhow!("{} line {}: {e:?}", path.display(), i + 1))?;
        let id = v.get("id").context("no id")?.dumps_compact().map_err(anyhow::Error::msg)?;
        let status = v.get("status").and_then(num).context("no status")? as u16;
        let body = v.get("body").cloned().unwrap_or(Json::Null);
        anyhow::ensure!(m.insert(id.clone(), (status, body)).is_none(), "{}: id {id} twice", path.display());
    }
    Ok(m)
}

#[derive(Default, Debug)]
pub struct Diffs {
    pub dp: Vec<f64>,
    pub dlogit: Vec<f64>,
    pub derived: Vec<f64>,
    pub other: Vec<String>,
    /// (where, top-1 margin of B's probabilities)
    pub flips: Vec<(String, f64)>,
}

fn top1_margin(probs: Option<&Json>) -> f64 {
    let mut p: Vec<f64> = probs.and_then(|j| j.as_object()).map(|o| o.iter().filter_map(|(_, v)| num(v)).collect()).unwrap_or_default();
    p.sort_by(|a, b| b.total_cmp(a));
    p.first().copied().unwrap_or(0.0) - p.get(1).copied().unwrap_or(0.0)
}

/// Walk `a` against `b` at `path`.
pub fn walk(path: &str, a: &Json, b: &Json, margin: f64, d: &mut Diffs) {
    let in_answers = path.starts_with("answers.");
    match (a, b) {
        (Json::Object(x), Json::Object(y)) => {
            let skip = |k: &str| path == "qev" && k == "timing";
            let kx: Vec<&str> = x.iter().map(|(k, _)| k.as_str()).filter(|k| !skip(k)).collect();
            let ky: Vec<&str> = y.iter().map(|(k, _)| k.as_str()).filter(|k| !skip(k)).collect();
            if kx != ky {
                d.other.push(format!("{path}: keys {kx:?} vs {ky:?}"));
                return;
            }
            let m = if b.get("probabilities").is_some() { top1_margin(b.get("probabilities")) } else { margin };
            for k in kx {
                let p = if path.is_empty() { k.to_string() } else { format!("{path}.{k}") };
                walk(&p, a.get(k).unwrap(), b.get(k).unwrap(), m, d);
            }
        }
        (Json::Array(x), Json::Array(y)) if x.len() == y.len() => {
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                walk(&format!("{path}[{i}]"), u, v, margin, d);
            }
        }
        (x, y) if matches!(x, Json::Float(_)) || matches!(y, Json::Float(_)) => match (num(x), num(y)) {
            (Some(u), Some(v)) => {
                let diff = (u - v).abs();
                if path.contains(".slot_logits") {
                    d.dlogit.push(diff);
                } else if in_answers && (path.contains(".probabilities.") || path.ends_with(".noul")) {
                    d.dp.push(diff);
                } else if in_answers {
                    d.derived.push(diff);
                } else if diff != 0.0 {
                    d.other.push(format!("{path}: {u} vs {v}"));
                }
            }
            _ => d.other.push(format!("{path}: {x:?} vs {y:?}")),
        },
        (Json::Str(x), Json::Str(y)) if in_answers && path.ends_with(".choice") && x != y => {
            d.flips.push((format!("{path}: {x} vs {y}"), margin));
        }
        (x, y) => {
            if x != y {
                d.other.push(format!("{path}: {:.120} vs {:.120}", x.dumps_compact().unwrap_or_default(), y.dumps_compact().unwrap_or_default()));
            }
        }
    }
}

fn quant(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((q * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1]
}

pub fn main(a: &Path, b: &Path) -> Result<i32> {
    let (ra, rb) = (load(a)?, load(b)?);
    let only_a: Vec<&String> = ra.keys().filter(|k| !rb.contains_key(*k)).collect();
    let only_b: Vec<&String> = rb.keys().filter(|k| !ra.contains_key(*k)).collect();
    let mut d = Diffs::default();
    let (mut status_diff, mut identical, mut both) = (Vec::new(), 0usize, 0usize);
    for (id, (sa, ba)) in &ra {
        let Some((sb, bb)) = rb.get(id) else { continue };
        both += 1;
        if sa != sb {
            status_diff.push(format!("{id}: {sa} vs {sb}"));
            continue;
        }
        let before = (d.other.len(), d.dp.iter().filter(|&&x| x != 0.0).count(), d.dlogit.iter().filter(|&&x| x != 0.0).count(),
                      d.derived.iter().filter(|&&x| x != 0.0).count(), d.flips.len());
        walk("", ba, bb, 1.0, &mut d);
        let after = (d.other.len(), d.dp.iter().filter(|&&x| x != 0.0).count(), d.dlogit.iter().filter(|&&x| x != 0.0).count(),
                     d.derived.iter().filter(|&&x| x != 0.0).count(), d.flips.len());
        if before == after {
            identical += 1;
        }
        for e in d.other.iter_mut().skip(before.0) {
            *e = format!("{id}: {e}");
        }
    }
    let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    let flips: Vec<serde_json::Value> = d.flips.iter().map(|(w, m)| serde_json::json!({"where": w, "margin_b": m})).collect();
    let report = serde_json::json!({
        "a": a.display().to_string(), "b": b.display().to_string(),
        "ids_a": ra.len(), "ids_b": rb.len(), "only_a": only_a.len(), "only_b": only_b.len(), "compared": both,
        "status_mismatches": status_diff.len(), "other_mismatches": d.other.len(),
        "identical_bodies": identical,
        "dp_n": d.dp.len(), "dp_max": max(&d.dp), "dp_p99": quant(&mut d.dp.clone(), 0.99),
        "dp_identical": d.dp.iter().filter(|&&x| x == 0.0).count(),
        "dlogit_n": d.dlogit.len(), "dlogit_max": max(&d.dlogit),
        "derived_max": max(&d.derived),
        "choice_flips": flips,
        "examples": status_diff.iter().chain(d.other.iter()).take(5).collect::<Vec<_>>(),
    });
    println!("{report}");
    Ok(if only_a.is_empty() && only_b.is_empty() && status_diff.is_empty() && d.other.is_empty() { 0 } else { 1 })
}
