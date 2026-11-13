//! Slot logits → typed answers: calibration temperature, softmax, chunk recombination, debias averaging and the
//! answer builders of `qev/answers.py`, op for op in f64 (Python floats), including Python 3.12's compensated
//! `sum()` and `round(x, 6)`.

use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::path::Path;

use crate::api::schema::{Answer, Entry, QuestionKind, SystemOneRequest};
use crate::frontend::plan::{NONE_OPTION, Plan};
use crate::frontend::pyjson::{self, Json, py_fsum, py_round6};

/// Temperatures per `"kind"` or `"kind:K"` (`qev.calibrate.Calibration`). Missing keys fall back to the
/// primitive-level temperature, then to 1.0. elephant-1 ships no `calibration.json`, so every T is 1.0.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Calibration {
    pub temperatures: HashMap<String, f64>,
    pub source: String,
}

impl Calibration {
    pub fn none() -> Self {
        Calibration { temperatures: HashMap::new(), source: "none".into() }
    }

    pub fn temperature(&self, kind: &str, k: usize) -> f64 {
        self.temperatures.get(&format!("{kind}:{k}")).or_else(|| self.temperatures.get(kind)).copied().unwrap_or(1.0)
    }

    /// `Calibration.load`: `{"temperatures": {key: number}}`, values converted with Python's `float()`.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let v = pyjson::parse(&bytes).with_context(|| format!("parsing {}", path.display()))?;
        let mut temperatures = HashMap::new();
        if let Some(t) = v.get("temperatures") {
            let Some(t) = t.as_object() else { bail!("{}: temperatures is not an object", path.display()) };
            for (k, x) in t {
                let f = match x {
                    Json::Float(f) => *f,
                    Json::Int(s) => s.parse::<f64>()?,
                    Json::Bool(b) => *b as u8 as f64,
                    Json::Str(s) => pyjson::py_strip(s).parse::<f64>().with_context(|| format!("temperature {k}"))?,
                    _ => bail!("{}: temperature {k} is not a number", path.display()),
                };
                temperatures.insert(k.clone(), f);
            }
        }
        Ok(Calibration { temperatures, source: path.display().to_string() })
    }

    /// `<model_dir>/calibration.json` if present, else no calibration.
    pub fn load_dir(model_dir: &Path) -> Result<Self> {
        let p = model_dir.join("calibration.json");
        if p.exists() { Self::load(&p) } else { Ok(Self::none()) }
    }
}

/// `qev.calibrate.softmax`.
pub fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let z: Vec<f64> = logits.iter().map(|x| x / temperature).collect();
    let m = z.iter().copied().fold(z[0], py_max);
    let w: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
    let s = py_fsum(w.iter().copied());
    w.iter().map(|x| x / s).collect()
}

/// Python's `max(a, b)` for floats: b only if strictly greater.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// `max(range(len(p)), key=p.__getitem__)`: the first index of the maximum.
fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..p.len() {
        if p[i] > p[best] {
            best = i;
        }
    }
    best
}

fn normalize(probs: &[f64]) -> Vec<f64> {
    let total = py_fsum(probs.iter().copied());
    if total <= 0.0 {
        return vec![1.0 / probs.len() as f64; probs.len()];
    }
    probs.iter().map(|p| p / total).collect()
}

pub fn choice_confidence(probs: &[f64]) -> f64 {
    if probs.len() == 1 {
        return 1.0;
    }
    let p = normalize(probs);
    let uniform = 1.0 / p.len() as f64;
    let peak = p.iter().copied().fold(p[0], py_max);
    (peak - uniform) / (1.0 - uniform)
}

pub fn score_confidence(probs: &[f64]) -> f64 {
    if probs.len() == 1 {
        return 1.0;
    }
    let p = normalize(probs);
    let k = p.len();
    let mode = argmax(&p);
    let distance = py_fsum(p.iter().enumerate().map(|(i, pi)| pi * (i as f64 - mode as f64).abs()));
    let center = (k - 1) as f64 / 2.0;
    let uniform_mad = py_fsum((0..k).map(|i| (i as f64 - center).abs())) / k as f64;
    py_max(0.0, 1.0 - distance / uniform_mad)
}

pub fn expected_score(probs: &[f64]) -> f64 {
    let p = normalize(probs);
    py_fsum(p.iter().enumerate().map(|(i, pi)| i as f64 * pi))
}

pub fn make_noul(p_yes: f64) -> Answer {
    // min(1.0, max(0.0, p))
    let x = py_max(0.0, p_yes);
    Answer::Noul { noul: py_round6(if x < 1.0 { x } else { 1.0 }) }
}

pub fn make_choice(options: &[String], probs: &[f64]) -> Answer {
    let p = normalize(probs);
    let best = argmax(&p);
    Answer::Choice {
        choice: options[best].clone(),
        probabilities: options.iter().zip(&p).map(|(o, pi)| (o.clone(), py_round6(*pi))).collect(),
        confidence: py_round6(choice_confidence(&p)),
    }
}

pub fn make_score(levels: &[Entry], probs: &[f64]) -> Answer {
    let p = normalize(probs);
    Answer::Score {
        score: py_round6(expected_score(&p)),
        legend: levels.iter().enumerate().map(|(i, l)| (i.to_string(), l.clone())).collect(),
        probabilities: p.iter().enumerate().map(|(i, pi)| (i.to_string(), py_round6(*pi))).collect(),
        confidence: py_round6(score_confidence(&p)),
    }
}

/// Answers in qev's order, plus its `debug` diagnostics per question.
#[derive(Clone, Debug, PartialEq)]
pub struct Assembled {
    pub answers: Vec<(String, Answer)>,
    /// `{qid: {allowed_token_mass?, slot_logits, chunks?, temperature}}` (the `diagnostics` of qev's debug meta).
    /// `allowed_token_mass` is present only when masses were given.
    pub diagnostics: Json,
}

struct Acc {
    kind: QuestionKind,
    probs: Vec<f64>,
    labels: Vec<String>,
    descriptions: Vec<Entry>,
    n: usize,
    mass: f64,
    logits: Vec<Json>,
}

struct Chunked {
    scores: HashMap<String, (f64, f64)>,
    weights: Vec<f64>,
    mass: f64,
    n: usize,
    logits: Vec<Json>,
}

fn logits_json(l: &[f64]) -> Json {
    Json::Array(l.iter().map(|x| Json::Float(*x)).collect())
}

/// The answer half of `decide()`: `row(i)` is branch i's slot logits (at least `n_slots` wide), `masses` the
/// optional per-branch `allowed_token_mass`. `max_slots` is the engine's slot count (the chunked temperature key).
pub fn assemble<'a>(
    req: &SystemOneRequest,
    plan: &Plan,
    calibration: &Calibration,
    max_slots: usize,
    row: impl Fn(usize) -> &'a [f32],
    masses: Option<&[f64]>,
) -> Result<Assembled> {
    ensure!(masses.is_none_or(|m| m.len() == plan.branches.len()), "one mass per branch");
    let mut acc: Vec<(usize, Acc)> = Vec::new();
    let mut chunked: Vec<(usize, Chunked)> = Vec::new();
    for (i, br) in plan.branches.iter().enumerate() {
        let k = br.n_slots();
        let r = row(i);
        ensure!(r.len() >= k, "branch {i}: {} logits for {k} slots", r.len());
        let logits: Vec<f64> = r[..k].iter().map(|&x| x as f64).collect();
        let probs = softmax(&logits, calibration.temperature(br.kind.as_str(), k));
        let mass = masses.map_or(0.0, |m| m[i]);
        if br.chunk.is_some() {
            let pos = chunked.iter().position(|(q, _)| *q == br.question).unwrap_or_else(|| {
                chunked.push((
                    br.question,
                    Chunked { scores: HashMap::new(), weights: Vec::new(), mass: 0.0, n: 0, logits: Vec::new() },
                ));
                chunked.len() - 1
            });
            let c = &mut chunked[pos].1;
            let none = br.labels.iter().position(|l| l == NONE_OPTION).context("chunk without the none slot")?;
            let keep = py_max(1.0 - probs[none], 1e-9);
            for (lab, p) in br.labels.iter().zip(&probs) {
                if lab != NONE_OPTION {
                    c.scores.insert(lab.clone(), (p / keep, keep));
                }
            }
            c.weights.push(keep);
            c.mass += mass;
            c.n += 1;
            c.logits.push(logits_json(&logits));
            continue;
        }
        let pos = acc.iter().position(|(q, _)| *q == br.question).unwrap_or_else(|| {
            acc.push((
                br.question,
                Acc {
                    kind: br.kind,
                    probs: vec![0.0; k],
                    labels: vec![String::new(); k],
                    descriptions: vec![Json::Null; k],
                    n: 0,
                    mass: 0.0,
                    logits: Vec::new(),
                },
            ));
            acc.len() - 1
        });
        let a = &mut acc[pos].1;
        ensure!(a.probs.len() == k && br.order.len() == k, "branch {i}: slot count differs between shifts");
        for (pos_i, &orig_i) in br.order.iter().enumerate() {
            a.probs[orig_i] += probs[pos_i];
            a.labels[orig_i] = br.labels[pos_i].clone();
            a.descriptions[orig_i] = br.descriptions[pos_i].clone();
        }
        a.n += 1;
        a.mass += mass;
        a.logits.push(logits_json(&logits));
    }

    let mut answers = Vec::with_capacity(acc.len() + chunked.len());
    let mut diag = Vec::with_capacity(answers.capacity());
    let qid = |q: usize| req.questions[q].0.clone();
    let diag_entry = |mass: f64, n: usize, logits: Vec<Json>, chunks: Option<usize>, t: f64| {
        let mut d = Vec::with_capacity(4);
        if masses.is_some() {
            d.push(("allowed_token_mass".to_string(), Json::Float(mass / n as f64)));
        }
        d.push(("slot_logits".to_string(), Json::Array(logits)));
        if let Some(c) = chunks {
            d.push(("chunks".to_string(), Json::int(c as i64)));
        }
        d.push(("temperature".to_string(), Json::Float(t)));
        Json::Object(d)
    };
    for (q, c) in chunked {
        let total_w = py_fsum(c.weights.iter().copied());
        let total_w = if total_w == 0.0 { 1.0 } else { total_w };
        let crate::api::schema::Question::Choice { criteria, .. } = &req.questions[q].1 else {
            bail!("chunked branch of a non-choice question");
        };
        let names: Vec<String> = criteria.iter().map(|(n, _)| n.clone()).collect();
        let probs = names
            .iter()
            .map(|n| c.scores.get(n).map(|(p, w)| p * (w / total_w)).with_context(|| format!("no score for {n:?}")))
            .collect::<Result<Vec<f64>>>()?;
        answers.push((qid(q), make_choice(&names, &probs)));
        diag.push((qid(q), diag_entry(c.mass, c.n, c.logits, Some(c.n), calibration.temperature("choice", max_slots))));
    }
    for (q, a) in acc {
        let probs: Vec<f64> = a.probs.iter().map(|p| p / a.n as f64).collect();
        let answer = match a.kind {
            QuestionKind::Noul => make_noul(probs[0]),
            QuestionKind::Choice => make_choice(&a.labels, &probs),
            QuestionKind::Score => make_score(&a.descriptions, &probs),
        };
        answers.push((qid(q), answer));
        let t = calibration.temperature(a.kind.as_str(), probs.len());
        diag.push((qid(q), diag_entry(a.mass, a.n, a.logits, None, t)));
    }
    Ok(Assembled { answers, diagnostics: Json::Object(diag) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_fallbacks() {
        let mut c = Calibration::none();
        assert_eq!(c.temperature("choice", 3), 1.0);
        c.temperatures.insert("choice".into(), 2.0);
        c.temperatures.insert("choice:3".into(), 3.0);
        assert_eq!(c.temperature("choice", 3), 3.0);
        assert_eq!(c.temperature("choice", 4), 2.0);
        assert_eq!(c.temperature("noul", 2), 1.0);
    }

    #[test]
    fn answer_builders() {
        assert_eq!(make_noul(1.2), Answer::Noul { noul: 1.0 });
        assert_eq!(make_noul(-0.1), Answer::Noul { noul: 0.0 });
        let Answer::Choice { choice, confidence, .. } = make_choice(&["a".into(), "b".into()], &[0.5, 0.5]) else {
            panic!()
        };
        assert_eq!((choice.as_str(), confidence), ("a", 0.0));
        let Answer::Score { score, confidence, .. } = make_score(&[Json::Null, Json::Null, Json::Null], &[0.0, 0.0, 1.0]) else { panic!() };
        assert_eq!((score, confidence), (2.0, 1.0));
        assert_eq!(softmax(&[0.0, 0.0], 1.0), [0.5, 0.5]);
    }
}
