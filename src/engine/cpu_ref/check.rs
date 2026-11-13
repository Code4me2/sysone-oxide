//! Compare the CPU reference with the qev fp32 run recorded by `reference/hf_check/run_qev.py`.
//!
//! All recorded requests go into one packed [`DecisionBatch`]; each request also runs alone. Slot logits and
//! probabilities (softmax at T = 1, elephant-1 ships no calibration) are compared against the reference's
//! sequential path (prefix forward, then each suffix off a copy of the cache) and its batched path.

use super::CpuRefEngine;
use super::ops::softmax;
use crate::contract::{DecisionBatch, DecisionEngine, MAX_SLOTS};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Deserialize)]
pub struct Recorded {
    pub provenance: serde_json::Value,
    pub slot_ids: Vec<u32>,
    pub requests: Vec<RecordedRequest>,
}

#[derive(Debug, Deserialize)]
pub struct RecordedRequest {
    pub name: String,
    pub prefix_ids: Vec<u32>,
    pub branches: Vec<RecordedBranch>,
}

#[derive(Debug, Deserialize)]
pub struct RecordedBranch {
    pub qid: String,
    pub kind: String,
    pub n_slots: usize,
    pub suffix_ids: Vec<u32>,
    pub logits_seq: Vec<f32>,
    pub logits_bat: Vec<f32>,
}

impl Recorded {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    fn batch(reqs: &[&RecordedRequest]) -> Result<DecisionBatch> {
        let mut b = DecisionBatch::default();
        for r in reqs {
            let s: Vec<(&[u32], u16)> = r.branches.iter().map(|x| (x.suffix_ids.as_slice(), x.n_slots as u16)).collect();
            b.push_request(&r.prefix_ids, &s)?;
        }
        b.validate()?;
        Ok(b)
    }
}

/// max and p99 of |a - b| over paired values.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stat {
    pub max: f64,
    pub p99: f64,
    pub n: usize,
}

impl Stat {
    pub fn of(mut d: Vec<f64>) -> Self {
        if d.is_empty() {
            return Stat::default();
        }
        d.sort_by(f64::total_cmp);
        let p99 = d[((d.len() as f64 * 0.99).ceil() as usize).clamp(1, d.len()) - 1];
        Stat { max: *d.last().unwrap(), p99, n: d.len() }
    }
}

impl std::fmt::Display for Stat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "max {:.3e}  p99 {:.3e}  (n={})", self.max, self.p99, self.n)
    }
}

#[derive(Debug, Default)]
pub struct Report {
    /// CPU packed vs qev sequential: |dlogit| and |dp|.
    pub logit_vs_seq: Stat,
    pub prob_vs_seq: Stat,
    /// CPU packed vs qev batched.
    pub logit_vs_bat: Stat,
    pub prob_vs_bat: Stat,
    /// qev's own sequential vs batched spread.
    pub ref_logit_seq_vs_bat: Stat,
    pub ref_prob_seq_vs_bat: Stat,
    /// CPU packed vs each request alone (bitwise equality expected).
    pub packed_vs_alone_logit: Stat,
    pub packed_vs_alone_bitwise: bool,
    pub top_choice_agree: usize,
    pub branches: usize,
    pub tokens_packed: usize,
    pub seconds_packed: f64,
    pub seconds_alone: f64,
}

fn diffs(a: &[f32], b: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let pa = softmax(a);
    let pb = softmax(b);
    let dl = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).collect();
    let dp = pa.iter().zip(&pb).map(|(x, y)| (x - y).abs()).collect();
    (dl, dp)
}

fn argmax(x: &[f32]) -> usize {
    x.iter().enumerate().fold(0, |b, (i, v)| if *v > x[b] { i } else { b })
}

/// Run the comparison. `verbose` prints a per-branch table.
pub fn compare(engine: &mut CpuRefEngine, rec: &Recorded, verbose: bool) -> Result<Report> {
    ensure!(rec.slot_ids.len() <= MAX_SLOTS, "{} recorded slot ids", rec.slot_ids.len());
    ensure!(engine.slot_ids[..rec.slot_ids.len()] == rec.slot_ids[..],
            "slot ids from tokenizer.json differ from the ids qev verified");
    let reqs: Vec<&RecordedRequest> = rec.requests.iter().collect();
    let packed = Recorded::batch(&reqs)?;
    let mut rep = Report { tokens_packed: packed.num_tokens(), ..Default::default() };

    let t0 = Instant::now();
    let out = engine.run(&packed)?;
    rep.seconds_packed = t0.elapsed().as_secs_f64();
    if verbose {
        println!("packed: {} requests, {} branches, {} tokens, {:.1}s", reqs.len(), packed.num_branches(),
                 packed.num_tokens(), rep.seconds_packed);
    }

    let (mut ls, mut ps, mut lb, mut pb, mut lr, mut pr, mut la) = Default::default();
    let ext = |v: &mut Vec<f64>, x: Vec<f64>| v.extend(x);
    let mut bitwise = true;
    let mut row = 0;
    for r in &reqs {
        let t1 = Instant::now();
        let alone = engine.run(&Recorded::batch(&[r])?)?;
        rep.seconds_alone += t1.elapsed().as_secs_f64();
        for (i, b) in r.branches.iter().enumerate() {
            let k = b.n_slots;
            let mine = out.row(row, k);
            let (dl, dp) = diffs(mine, &b.logits_seq);
            if verbose {
                let p = softmax(mine);
                let q = softmax(&b.logits_seq);
                println!("  {:18} {:10} {:6} max|dlogit| {:.2e} max|dp| {:.2e}  p_cpu {:?}  p_ref {:?}", r.name,
                         b.qid, b.kind, dl.iter().cloned().fold(0.0, f64::max), dp.iter().cloned().fold(0.0, f64::max),
                         p.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(),
                         q.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>());
            }
            ext(&mut ls, dl);
            ext(&mut ps, dp);
            let (dl, dp) = diffs(mine, &b.logits_bat);
            ext(&mut lb, dl);
            ext(&mut pb, dp);
            let (dl, dp) = diffs(&b.logits_seq, &b.logits_bat);
            ext(&mut lr, dl);
            ext(&mut pr, dp);
            let a = alone.row(i, k);
            bitwise &= a.iter().zip(mine).all(|(x, y)| x.to_bits() == y.to_bits());
            ext(&mut la, diffs(a, mine).0);
            rep.top_choice_agree += (argmax(mine) == argmax(&b.logits_seq)) as usize;
            rep.branches += 1;
            row += 1;
        }
    }
    rep.logit_vs_seq = Stat::of(ls);
    rep.prob_vs_seq = Stat::of(ps);
    rep.logit_vs_bat = Stat::of(lb);
    rep.prob_vs_bat = Stat::of(pb);
    rep.ref_logit_seq_vs_bat = Stat::of(lr);
    rep.ref_prob_seq_vs_bat = Stat::of(pr);
    rep.packed_vs_alone_logit = Stat::of(la);
    rep.packed_vs_alone_bitwise = bitwise;
    Ok(rep)
}

impl Report {
    pub fn print(&self) {
        println!("CPU packed vs qev fp32 sequential: |dlogit| {}", self.logit_vs_seq);
        println!("                                   |dp|     {}", self.prob_vs_seq);
        println!("CPU packed vs qev fp32 batched:    |dlogit| {}", self.logit_vs_bat);
        println!("                                   |dp|     {}", self.prob_vs_bat);
        println!("qev sequential vs batched (ref):   |dlogit| {}", self.ref_logit_seq_vs_bat);
        println!("                                   |dp|     {}", self.ref_prob_seq_vs_bat);
        println!("CPU packed vs alone:               |dlogit| {}  bitwise identical: {}", self.packed_vs_alone_logit,
                 self.packed_vs_alone_bitwise);
        println!("top choice agrees on {}/{} branches; packed {} tokens in {:.1}s, alone total {:.1}s",
                 self.top_choice_agree, self.branches, self.tokens_packed, self.seconds_packed, self.seconds_alone);
    }
}
