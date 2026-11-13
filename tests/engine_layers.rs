//! Per-layer hidden-state gate of the bf16 GPU engine (DESIGN §9.2 "Model"; Phase 2a, I1).
//!
//! For the 10 requests with W1b hidden dumps, the engine runs each request alone with debug taps on the residual
//! stream after layers 0, 3, 4, 15, 31 and on the final-norm output. At the dumped rows (8 sampled prefix rows
//! and every branch's answer row) the relative error `||h - h_fp32|| / ||h_fp32||` per row is compared with
//! qev fp32 (served path, the truth). Gate: `[model.hidden.layerN]` rel_p99/rel_max of `tests/tolerances.toml`
//! (1.5 x the bf16 reference's own error vs fp32); the rule is also checked directly against the bf16 dumps
//! (sysone p99/max <= 1.5 x bf16-reference p99/max).
//!
//! `scripts/gpu.sh cargo test --release --test engine_layers -- --nocapture`

use std::path::PathBuf;

use sysone::engine::forward::Taps;
use sysone::engine::goldens::{self, SafeTensors};
use sysone::engine::gpu::{EngineOptions, GpuEngine};
use sysone::frontend::{Frontend, Limits};
use sysone::tolerances::{Stats, Tolerances};

const LAYERS: [usize; 5] = [0, 3, 4, 15, 31];

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn model_dir() -> PathBuf {
    std::env::var_os("SYSONE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/velvet/Desktop/hf_models/elephant-1"))
}

fn rel(a: &[f32], truth: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(truth) {
        num += (*x as f64 - *y as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

#[test]
fn hidden_states_vs_fp32_truth() {
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let ids = goldens::hidden_ids(&repo(), "fp32_batched").unwrap();
    let cases = goldens::model_corpus(&repo()).unwrap();
    let plans = goldens::plan_corpus(&fe, &cases, |id| ids.iter().any(|x| x == id)).unwrap();
    assert_eq!(plans.len(), ids.len(), "every dumped request plans");
    let taps = Taps { layers: LAYERS.to_vec(), norm: true };
    let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(32768), warmup: false, cuda_graphs: false, taps,
                               ..Default::default() };
    let mut e = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
    let names: Vec<String> = LAYERS.iter().map(|l| format!("layer{l}")).chain(["norm".to_string()]).collect();
    // Per tap: (sysone vs fp32, bf16 reference vs fp32, sysone vs bf16 reference).
    let mut errs: Vec<[Vec<f64>; 3]> = vec![Default::default(); names.len()];
    // Row labels, in the order errors are pushed (same for every tap).
    let mut labels: Vec<String> = Vec::new();
    for p in &plans {
        let id = &p.case.id;
        let (b, _) = goldens::batch_of(&[&p.plan]).unwrap();
        use sysone::contract::DecisionEngine;
        e.run(&b).unwrap();
        let f32d = SafeTensors::read(&goldens::hidden_dump(&repo(), "fp32_batched", id).unwrap()).unwrap();
        let bf16d = SafeTensors::read(&goldens::hidden_dump(&repo(), "bf16_batched", id).unwrap()).unwrap();
        // Same token ids as the reference.
        let pre: Vec<u32> = f32d.i64("ids/prefix").unwrap().iter().map(|&x| x as u32).collect();
        assert_eq!(pre, p.plan.prefix_ids, "{id}: prefix ids differ from the dump");
        for (k, br) in p.plan.branches.iter().enumerate() {
            let s: Vec<u32> = f32d.i64(&format!("ids/suffix{k}")).unwrap().iter().map(|&x| x as u32).collect();
            assert_eq!(s, br.suffix_ids, "{id}: suffix {k} ids differ from the dump");
        }
        let prefix_rows: Vec<usize> = f32d.i64("prefix_rows").unwrap().iter().map(|&x| x as usize).collect();
        let rows: Vec<usize> = prefix_rows.iter().copied().chain(b.answer_rows.iter().map(|&r| r as usize)).collect();
        let t = b.num_tokens();
        labels.extend(prefix_rows.iter().map(|r| format!("{id} prefix row {r}")));
        labels.extend((0..b.num_branches()).map(|k| format!("{id} answer {k}")));
        for (ti, name) in names.iter().enumerate() {
            let ours = e.read_tap(name, t).unwrap();
            let mut truth = f32d.f32(&format!("{name}/prefix")).unwrap();
            truth.extend(f32d.f32(&format!("{name}/answer")).unwrap());
            let mut refb = bf16d.f32(&format!("{name}/prefix")).unwrap();
            refb.extend(bf16d.f32(&format!("{name}/answer")).unwrap());
            let h = 4096;
            assert_eq!(truth.len(), rows.len() * h, "{id} {name}: dump rows");
            for (k, &r) in rows.iter().enumerate() {
                let o = &ours[r * h..(r + 1) * h];
                let (tr, rb) = (&truth[k * h..(k + 1) * h], &refb[k * h..(k + 1) * h]);
                errs[ti][0].push(rel(o, tr));
                errs[ti][1].push(rel(rb, tr));
                errs[ti][2].push(rel(o, rb));
            }
        }
        println!("{id}: T={t}, {} rows checked per tap", rows.len());
    }
    let t = Tolerances::load_default().unwrap();
    let mut failed = Vec::new();
    for (ti, name) in names.iter().enumerate() {
        let [ours, refb, cross] = &errs[ti];
        let (so, sr, sc) = (Stats::of(ours).unwrap(), Stats::of(refb).unwrap(), Stats::of(cross).unwrap());
        let key = format!("model.hidden.{name}");
        let (p99, max) = (t.f64(&format!("{key}.rel_p99")).unwrap(), t.f64(&format!("{key}.rel_max")).unwrap());
        let factor = t.f64("factor").unwrap();
        println!("{name:7} sysone vs fp32: {so}\n        bf16 ref vs fp32: {sr}\n        sysone vs bf16 ref: {sc}\n        \
                  gate [{key}] rel_p99 {p99} rel_max {max}; rule {factor} x bf16-ref: p99 {:.3e} max {:.3e}",
                 factor * sr.p99, factor * sr.max);
        if so.p99 > p99 || so.max > max {
            failed.push(format!("{name}: p99 {:.3e} (limit {p99}), max {:.3e} (limit {max})", so.p99, so.max));
        }
        let mut worst: Vec<usize> = (0..ours.len()).collect();
        worst.sort_by(|&a, &b| ours[b].total_cmp(&ours[a]));
        for &k in worst.iter().take(3) {
            println!("        worst: {}: sysone {:.3e}, bf16 ref {:.3e}", labels[k], ours[k], refb[k]);
        }
        if so.p99 > factor * sr.p99 || so.max > factor * sr.max {
            println!("        NOTE: {name} is outside {factor} x this run's bf16-reference error");
        }
    }
    assert!(failed.is_empty(), "hidden-state gate failures:\n  {}", failed.join("\n  "));
}

/// Diagnostic (ignored): per-row relative error over every prefix row of one request, against an fp32 dump
/// made with `reference/layer_dump.py --dtype float32 --prefix-rows 0,1,...` (`SYSONE_DUMP=path`,
/// `SYSONE_DUMP_ID=request id`).
#[test]
#[ignore]
fn hidden_state_row_profile() {
    let (Some(path), Some(id)) = (std::env::var_os("SYSONE_DUMP"), std::env::var_os("SYSONE_DUMP_ID")) else {
        panic!("set SYSONE_DUMP and SYSONE_DUMP_ID");
    };
    let id = id.to_string_lossy().to_string();
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let cases = goldens::model_corpus(&repo()).unwrap();
    let plans = goldens::plan_corpus(&fe, &cases, |x| x == id).unwrap();
    let taps = Taps { layers: LAYERS.to_vec(), norm: true };
    let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(32768), warmup: false, cuda_graphs: false, taps,
                               ..Default::default() };
    let mut e = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
    let (b, _) = goldens::batch_of(&[&plans[0].plan]).unwrap();
    use sysone::contract::DecisionEngine;
    e.run(&b).unwrap();
    let d = SafeTensors::read(std::path::Path::new(&path)).unwrap();
    let rows: Vec<usize> = d.i64("prefix_rows").unwrap().iter().map(|&x| x as usize).collect();
    for name in LAYERS.iter().map(|l| format!("layer{l}")).chain(["norm".to_string()]) {
        let ours = e.read_tap(&name, b.num_tokens()).unwrap();
        let truth = d.f32(&format!("{name}/prefix")).unwrap();
        let errs: Vec<f64> = rows
            .iter()
            .enumerate()
            .map(|(k, &r)| rel(&ours[r * 4096..(r + 1) * 4096], &truth[k * 4096..(k + 1) * 4096]))
            .collect();
        let line: Vec<String> = errs.iter().map(|x| format!("{:.0}", x * 1e4)).collect();
        println!("{name}: sysone vs fp32 {} (x1e-4 per prefix row)\n    {}", Stats::of(&errs).unwrap(), line.join(" "));
        // Optional bf16 reference dump of the same rows: its own error vs fp32, row by row.
        if let Some(p) = std::env::var_os("SYSONE_DUMP_BF16") {
            let r = SafeTensors::read(std::path::Path::new(&p)).unwrap().f32(&format!("{name}/prefix")).unwrap();
            let re: Vec<f64> = (0..rows.len())
                .map(|k| rel(&r[k * 4096..(k + 1) * 4096], &truth[k * 4096..(k + 1) * 4096]))
                .collect();
            let worse = errs.iter().zip(&re).filter(|(a, b)| a > b).count();
            println!("    bf16 ref vs fp32 {}; sysone worse than the reference on {worse}/{} rows",
                     Stats::of(&re).unwrap(), rows.len());
        }
    }
}
