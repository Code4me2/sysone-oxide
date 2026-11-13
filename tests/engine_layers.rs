//! Per-layer hidden-state gate (a) of the bf16 GPU engine (DESIGN §9.2 "Model"; Phase 2a I1, redefined in A1).
//!
//! For the 10 requests with dense hidden dumps (`reference/hidden_dense.py`), the engine runs each request alone
//! with debug taps on the residual stream after layers 0, 3, 4, 15, 31 and on the final-norm output. On the dense
//! row set (every prefix row, or a stride of ~1k rows for the 4k and 16k requests, plus every answer row: 3,829
//! rows per tap) the relative error `||h - h_fp32|| / ||h_fp32||` per row is compared with qev fp32 (served path,
//! the truth), and so is the bf16 served reference's on the SAME rows. Gate per tap ([`HiddenGate`],
//! `[model.hidden]` of `tests/tolerances.toml`): sysone p99 <= factor x reference p99; median per-row log ratio
//! <= ln 1.5; sign test on rows > 1.5x worse vs > 1.5x better (one-sided binomial, alpha 0.01). Max is reported.
//!
//! `scripts/gpu.sh cargo test --release --test engine_layers -- --nocapture`

use std::path::PathBuf;

use sysone::engine::forward::Taps;
use sysone::engine::goldens::{self, SafeTensors};
use sysone::engine::gpu::{EngineOptions, GpuEngine, gdn_backend_from_env};
use sysone::frontend::{Frontend, Limits};
use sysone::tolerances::{HiddenGate, Stats, Tolerances};

const LAYERS: [usize; 5] = [0, 3, 4, 15, 31];
const H: usize = 4096;

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

fn tap_names() -> Vec<String> {
    let names: Vec<String> = LAYERS.iter().map(|l| format!("layer{l}")).chain(["norm".to_string()]).collect();
    assert_eq!(names, HiddenGate::TAPS, "taps differ from the gate's");
    names
}

/// One dense dump: its prefix rows (checked against the row set) and, per tap, the dumped rows
/// (prefix rows then answer rows) as one `[R, H]` buffer.
struct Dense {
    prefix_rows: Vec<usize>,
    branches: usize,
    rows: Vec<Vec<f32>>,
    file: SafeTensors,
}

fn dense(run: &str, id: &str) -> Dense {
    let f = SafeTensors::read(&goldens::hidden_dense_dump(&repo(), run, id).unwrap()).unwrap();
    let p = f.i64("ids/prefix").unwrap().len();
    let prefix_rows: Vec<usize> = f.i64("prefix_rows").unwrap().iter().map(|&x| x as usize).collect();
    assert_eq!(prefix_rows, goldens::dense_prefix_rows(p), "{run} {id}: prefix rows are not the dense row set");
    let branches = (0..).take_while(|k| f.shape(&format!("ids/suffix{k}")).is_some()).count();
    let rows = tap_names()
        .iter()
        .map(|name| {
            let mut v = f.f32(&format!("{name}/prefix")).unwrap();
            v.extend(f.f32(&format!("{name}/answer")).unwrap());
            assert_eq!(v.len(), (prefix_rows.len() + branches) * H, "{run} {id} {name}: dump rows");
            v
        })
        .collect();
    Dense { prefix_rows, branches, rows, file: f }
}

/// The reference's p99 on the dense rows must be the one tolerances.toml was written from (same dumps, same rows).
fn check_measured(t: &Tolerances, tap: &str, refe: &Stats) {
    let k = |m: &str| t.f64(&format!("model.hidden.{tap}.{m}")).unwrap();
    assert_eq!(refe.n as f64, k("measured_rows"), "{tap}: row count differs from tolerances.toml");
    let m = k("measured_rel_p99");
    assert!((refe.p99 - m).abs() <= 1e-9 * m, "{tap}: reference p99 {} != measured_rel_p99 {m}", refe.p99);
}

#[test]
fn hidden_states_vs_fp32_truth() {
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let ids = goldens::hidden_dense_ids(&repo(), "fp32_served").unwrap();
    assert_eq!(ids, goldens::hidden_dense_ids(&repo(), "bf16_served").unwrap());
    let cases = goldens::model_corpus(&repo()).unwrap();
    let plans = goldens::plan_corpus(&fe, &cases, |id| ids.iter().any(|x| x == id)).unwrap();
    assert_eq!(plans.len(), ids.len(), "every dumped request plans");
    let taps = Taps { layers: LAYERS.to_vec(), norm: true, ..Default::default() };
    let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(32768), warmup: false, cuda_graphs: false, taps,
                               gdn_backend: gdn_backend_from_env().unwrap(), ..Default::default() };
    let mut e = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
    eprintln!("engine_layers: GDN backend {}", e.gdn_backend().name());
    let names = tap_names();
    // Per tap: (sysone vs fp32, bf16 reference vs fp32, sysone vs bf16 reference), in the gate's row order
    // (requests sorted by id; per request prefix rows then answer rows).
    let mut errs: Vec<[Vec<f64>; 3]> = vec![Default::default(); names.len()];
    let mut labels: Vec<String> = Vec::new();
    let mut plans = plans;
    plans.sort_by(|a, b| a.case.id.cmp(&b.case.id));
    for p in &plans {
        let id = &p.case.id;
        let (b, _) = goldens::batch_of(&[&p.plan]).unwrap();
        use sysone::contract::DecisionEngine;
        e.run(&b).unwrap();
        let f32d = dense("fp32_served", id);
        let bf16d = dense("bf16_served", id);
        // Same token ids as the reference.
        let pre: Vec<u32> = f32d.file.i64("ids/prefix").unwrap().iter().map(|&x| x as u32).collect();
        assert_eq!(pre, p.plan.prefix_ids, "{id}: prefix ids differ from the dump");
        assert_eq!(f32d.branches, p.plan.branches.len(), "{id}: branch count");
        for (k, br) in p.plan.branches.iter().enumerate() {
            let s: Vec<u32> = f32d.file.i64(&format!("ids/suffix{k}")).unwrap().iter().map(|&x| x as u32).collect();
            assert_eq!(s, br.suffix_ids, "{id}: suffix {k} ids differ from the dump");
        }
        assert_eq!(f32d.prefix_rows, bf16d.prefix_rows);
        let rows: Vec<usize> =
            f32d.prefix_rows.iter().copied().chain(b.answer_rows.iter().map(|&r| r as usize)).collect();
        let t = b.num_tokens();
        labels.extend(f32d.prefix_rows.iter().map(|r| format!("{id} prefix row {r}")));
        labels.extend((0..b.num_branches()).map(|k| format!("{id} answer {k}")));
        for (ti, name) in names.iter().enumerate() {
            let ours = e.read_tap(name, t).unwrap();
            let (truth, refb) = (&f32d.rows[ti], &bf16d.rows[ti]);
            for (k, &r) in rows.iter().enumerate() {
                let o = &ours[r * H..(r + 1) * H];
                let (tr, rb) = (&truth[k * H..(k + 1) * H], &refb[k * H..(k + 1) * H]);
                errs[ti][0].push(rel(o, tr));
                errs[ti][1].push(rel(rb, tr));
                errs[ti][2].push(rel(o, rb));
            }
        }
        println!("{id}: T={t}, {} rows checked per tap", rows.len());
    }
    let t = Tolerances::load_default().unwrap();
    let gate = HiddenGate::load(&t).unwrap();
    let mut failed = Vec::new();
    for (ti, name) in names.iter().enumerate() {
        let [ours, refb, cross] = &errs[ti];
        assert!(ours.len() >= gate.rows_min, "{name}: only {} rows", ours.len());
        let c = gate.compare(name, ours, refb).unwrap();
        check_measured(&t, name, &c.refe);
        let sc = Stats::of(cross).unwrap();
        let fails = gate.failures(&c);
        println!("{name:7} sysone vs fp32: {}\n        bf16 ref vs fp32: {}\n        sysone vs bf16 ref: {sc}\n        \
                  (i) p99 {:.3e} <= {} x {:.3e} = {:.3e}; max {:.3e} (ref {:.3e}, reported)\n        \
                  (ii) median ln(sysone/ref) {:+.4} <= {:.4}\n        \
                  (iii) rows > {}x worse {} vs > {}x better {} (design effect {}: {} vs {}): p = {:.3e} (alpha {})\n        {}",
                 c.cand, c.refe, c.cand.p99, gate.factor, c.refe.p99, gate.factor * c.refe.p99, c.cand.max,
                 c.refe.max, c.median_log_ratio, gate.paired_ratio.ln(), gate.sign_ratio, c.n_up, gate.sign_ratio,
                 c.n_down, c.design_effect, c.eff_up, c.eff_down, c.sign_p, gate.sign_alpha,
                 if fails.is_empty() { "PASS".into() } else { fails.join("; ") });
        // Where the sign test's rows come from, per request (output only; the verdict above is the gate's).
        let mut per_req: Vec<(&str, usize, usize, usize)> = Vec::new();
        for (k, l) in labels.iter().enumerate() {
            let id = l.split(' ').next().unwrap_or_default();
            if per_req.last().is_none_or(|r| r.0 != id) {
                per_req.push((id, 0, 0, 0));
            }
            let r = per_req.last_mut().unwrap();
            r.1 += 1;
            r.2 += usize::from(ours[k] > gate.sign_ratio * refb[k]);
            r.3 += usize::from(refb[k] > gate.sign_ratio * ours[k]);
        }
        let line: Vec<String> = per_req.iter().map(|(id, n, u, d)| format!("{id} {u}/{d} of {n}")).collect();
        println!("        rows > {}x worse / better per request: {}", gate.sign_ratio, line.join(", "));
        let mut worst: Vec<usize> = (0..ours.len()).collect();
        worst.sort_by(|&a, &b| ours[b].total_cmp(&ours[a]));
        for &k in worst.iter().take(3) {
            println!("        worst: {}: sysone {:.3e}, bf16 ref {:.3e}", labels[k], ours[k], refb[k]);
        }
        if !fails.is_empty() {
            failed.push(format!("{name}: {}", fails.join("; ")));
        }
    }
    assert!(failed.is_empty(), "hidden-state gate failures:\n  {}", failed.join("\n  "));
}

/// The gate's validation (`reference/hidden_dense.py gate`, `tests/golden/dev/model/hidden_gate_report.json`)
/// recomputed here from the dense dumps with [`HiddenGate`]: the bf16 null realizations (per-question path and three
/// hidden-basis permutations) vs the reference in both role orders, and the deterministic synthetic regressions on
/// top of `bf16_perm1` (every row's error scaled; 5% of rows scaled, strided or as one block) must reach the same
/// statistics and verdicts as the Python gate. No GPU.
#[test]
fn hidden_gate_matches_python() {
    let t = Tolerances::load_default().unwrap();
    let gate = HiddenGate::load(&t).unwrap();
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo().join("tests/golden/dev/model/hidden_gate_report.json")).unwrap(),
    )
    .unwrap();
    let g = &report["gate"];
    assert_eq!(
        (gate.factor, gate.paired_ratio, gate.sign_ratio, gate.sign_alpha, gate.rows_min as u64,
         HiddenGate::TAPS.map(|tap| g["design_effect"][tap].as_f64().unwrap())),
        (g["factor"].as_f64().unwrap(), g["paired_ratio"].as_f64().unwrap(), g["sign_ratio"].as_f64().unwrap(),
         g["sign_alpha"].as_f64().unwrap(), g["rows_min"].as_u64().unwrap(), gate.design_effect),
        "hidden_gate_report.json was generated with other [model.hidden] parameters"
    );
    assert!(report["all_ok"].as_bool().unwrap(), "the Python validation itself did not meet its expectations");
    let names = tap_names();
    // Per run and tap: errors vs fp32 of the bf16 served reference and of the bf16 null realizations.
    const RUNS: [&str; 5] = ["bf16_served", "bf16_perq", "bf16_perm1", "bf16_perm2", "bf16_perm3"];
    let mut errs: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); names.len()]; RUNS.len()];
    for id in goldens::hidden_dense_ids(&repo(), "fp32_served").unwrap() {
        let f = dense("fp32_served", &id);
        for (ri, run) in RUNS.iter().enumerate() {
            let d = dense(run, &id);
            for ti in 0..names.len() {
                for k in 0..f.rows[ti].len() / H {
                    let s = k * H..(k + 1) * H;
                    errs[ri][ti].push(rel(&d.rows[ti][s.clone()], &f.rows[ti][s]));
                }
            }
        }
    }
    let refe = &errs[0];
    let run_errs = |run: &str| &errs[RUNS.iter().position(|r| *r == run).unwrap_or_else(|| panic!("no run {run}"))];
    let base = run_errs("bf16_perm1");
    let n = refe[0].len();
    assert_eq!(n as u64, report["rows"].as_u64().unwrap());
    for (ti, name) in names.iter().enumerate() {
        check_measured(&t, name, &Stats::of(&refe[ti]).unwrap());
    }
    let strided: Vec<bool> = (0..n).map(|k| k % 20 == 7).collect();
    let block: Vec<bool> = (0..n).map(|k| k >= n - n.div_ceil(20)).collect();
    let (mut checked, mut fails) = (0, 0);
    for cand in report["candidates"].as_array().unwrap() {
        if !cand["rust_check"].as_bool().unwrap() {
            continue;
        }
        let name = cand["name"].as_str().unwrap();
        let (kind, arg) = name.split_once(':').unwrap();
        let x: f64 = arg.parse().unwrap_or(f64::NAN);
        let mut all_pass = true;
        for (ti, tap) in names.iter().enumerate() {
            let scaled = |mask: &dyn Fn(usize) -> bool| -> Vec<f64> {
                base[ti].iter().enumerate().map(|(k, e)| if mask(k) { x * e } else { *e }).collect()
            };
            let (c, r) = match kind {
                "null" => (run_errs(arg)[ti].clone(), refe[ti].clone()),
                "null_swapped" => (refe[ti].clone(), run_errs(arg)[ti].clone()),
                "scale_all" => (scaled(&|_| true), refe[ti].clone()),
                "rows5_strided" => (scaled(&|k| strided[k]), refe[ti].clone()),
                "rows5_block" => (scaled(&|k| block[k]), refe[ti].clone()),
                _ => panic!("unknown rust_check candidate {name}"),
            };
            let got = gate.compare(tap, &c, &r).unwrap();
            let want = &cand["taps"][tap.as_str()];
            let close = |a: f64, k: &str| {
                let b = want[k].as_f64().unwrap();
                assert!((a - b).abs() <= 1e-9 * b.abs().max(1e-12), "{name} {tap} {k}: rust {a} python {b}");
            };
            close(got.cand.p99, "cand_p99");
            close(got.refe.p99, "ref_p99");
            close(got.median_log_ratio, "median_log_ratio");
            close(got.sign_p, "sign_p");
            assert_eq!((got.n_up as u64, got.n_down as u64),
                       (want["n_up"].as_u64().unwrap(), want["n_down"].as_u64().unwrap()), "{name} {tap}");
            assert_eq!((got.eff_up as u64, got.eff_down as u64, got.design_effect),
                       (want["eff_up"].as_u64().unwrap(), want["eff_down"].as_u64().unwrap(),
                        want["design_effect"].as_f64().unwrap()), "{name} {tap}: effective counts");
            let mut f: Vec<String> =
                gate.failures(&got).iter().map(|s| s.split_whitespace().next().unwrap().to_string()).collect();
            f.sort();
            let mut w: Vec<String> =
                want["failed"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
            w.sort();
            assert_eq!(f, w, "{name} {tap}: failed criteria");
            all_pass &= f.is_empty();
        }
        assert_eq!(if all_pass { "pass" } else { "fail" }, cand["verdict"].as_str().unwrap(), "{name}");
        checked += 1;
        fails += usize::from(!all_pass);
    }
    assert!(checked >= 10 && fails > 0 && fails < checked, "{checked} candidates checked, {fails} failing");
    println!("{checked} validation candidates reproduced ({fails} failing as in the Python report)");
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
    let taps = Taps { layers: LAYERS.to_vec(), norm: true, ..Default::default() };
    let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(32768), warmup: false, cuda_graphs: false, taps,
                               gdn_backend: gdn_backend_from_env().unwrap(), ..Default::default() };
    let mut e = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
    eprintln!("engine_layers: GDN backend {}", e.gdn_backend().name());
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

/// A minimal safetensors writer for the diagnostic dump below (f32 and i64 tensors, string metadata).
fn write_safetensors(path: &std::path::Path, tensors: &[(String, &str, Vec<usize>, Vec<u8>)], meta: &[(&str, String)]) {
    let mut hdr = serde_json::Map::new();
    let md: serde_json::Map<String, serde_json::Value> =
        meta.iter().map(|(k, v)| (k.to_string(), serde_json::Value::String(v.clone()))).collect();
    hdr.insert("__metadata__".into(), md.into());
    let mut off = 0usize;
    for (name, dt, shape, bytes) in tensors {
        hdr.insert(name.clone(), serde_json::json!({"dtype": dt, "shape": shape, "data_offsets": [off, off + bytes.len()]}));
        off += bytes.len();
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(hdr)).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    use std::io::Write;
    f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&h).unwrap();
    for t in tensors {
        f.write_all(&t.3).unwrap();
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64_bytes(v: impl IntoIterator<Item = i64>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Diagnostic (ignored; deep-layer investigation A3): sysone's residual stream after EVERY layer, and the final
/// norm, on the dense row set of the `hidden_dense` requests, in the layout of `reference/layer_dump.py`
/// (`layer{i}/prefix`, `layer{i}/answer`, `norm/...`, f32 upcasts of the bf16 values), one file per request in
/// `SYSONE_TAP_OUT`. `SYSONE_TAP_IDS=a,b` limits the requests. For the ids in `SYSONE_TAP_FULL=a,b` the whole
/// packed `[T, 4096]` stream of every tap is written too (`full/layer{i}`, `full/norm`), with the token ids and
/// segment table, for teacher-forced single-layer comparisons (`reference/deep_layer.py`). With
/// `SYSONE_TAP_PACK_WITH=a,b` each request runs packed first in a batch with those requests (its rows keep their
/// indices; the step's bucket and GEMM shapes change). `SYSONE_TAP_SUB=1` also keeps each layer's mixer output, post-mixer
/// residual and MLP output (`mix{i}`, `mid{i}`, `mlp{i}`); `SYSONE_TAP_PERM=seed` runs the model in a permuted hidden
/// basis (another sysone realization, `EngineOptions::debug_permute_hidden`).
#[test]
#[ignore]
fn hidden_dump_all_layers() {
    let out = PathBuf::from(std::env::var_os("SYSONE_TAP_OUT").expect("set SYSONE_TAP_OUT"));
    let list = |k: &str| -> Option<Vec<String>> {
        std::env::var(k).ok().map(|v| v.split(',').filter(|s| !s.is_empty()).map(str::to_string).collect())
    };
    let only = list("SYSONE_TAP_IDS");
    let full = list("SYSONE_TAP_FULL").unwrap_or_default();
    let fe = Frontend::load(&model_dir(), Limits::default()).unwrap();
    let ids = goldens::hidden_dense_ids(&repo(), "fp32_served").unwrap();
    let cases = goldens::model_corpus(&repo()).unwrap();
    let mut plans = goldens::plan_corpus(&fe, &cases, |id| {
        ids.iter().any(|x| x == id) && only.as_ref().is_none_or(|o| o.iter().any(|x| x == id))
    })
    .unwrap();
    plans.sort_by(|a, b| a.case.id.cmp(&b.case.id));
    let pad_ids = list("SYSONE_TAP_PACK_WITH").unwrap_or_default();
    let pad = goldens::plan_corpus(&fe, &cases, |id| pad_ids.iter().any(|x| x == id)).unwrap();
    let layers: Vec<usize> = (0..32).collect();
    let sublayers = std::env::var_os("SYSONE_TAP_SUB").is_some();
    let taps = Taps { layers: layers.clone(), norm: true, sublayers };
    let debug_permute_hidden = std::env::var("SYSONE_TAP_PERM").ok().map(|s| s.parse().unwrap());
    let opts = EngineOptions { buckets: EngineOptions::buckets_up_to(32768), warmup: false, cuda_graphs: false, taps, debug_permute_hidden,
                               ..Default::default() };
    let mut e = GpuEngine::load(&model_dir(), &fe.slot_ids, opts).unwrap();
    let mut names: Vec<String> = layers.iter().map(|l| format!("layer{l}")).chain(["norm".to_string()]).collect();
    if sublayers {
        names.extend(layers.iter().flat_map(|l| ["mix", "mid", "mlp"].map(|k| format!("{k}{l}"))));
    }
    for p in &plans {
        let id = &p.case.id;
        // Optionally pack the request first in a batch with other requests (another bucket, other GEMM shapes:
        // another sysone realization of the same rows).
        let mut batch = vec![&p.plan];
        batch.extend(pad.iter().map(|q| &q.plan));
        let (b, _) = goldens::batch_of(&batch).unwrap();
        use sysone::contract::DecisionEngine;
        e.run(&b).unwrap();
        let t = b.num_tokens();
        let prefix_rows = goldens::dense_prefix_rows(p.plan.prefix_ids.len());
        let answer_rows: Vec<usize> =
            b.answer_rows.iter().take(p.plan.branches.len()).map(|&r| r as usize).collect();
        let with_full = full.iter().any(|x| x == id);
        let mut tensors = Vec::new();
        for name in &names {
            let all = e.read_tap(name, t).unwrap();
            let pick = |rows: &[usize]| -> Vec<u8> {
                f32_bytes(&rows.iter().flat_map(|&r| all[r * H..(r + 1) * H].iter().copied()).collect::<Vec<f32>>())
            };
            tensors.push((format!("{name}/prefix"), "F32", vec![prefix_rows.len(), H], pick(&prefix_rows)));
            tensors.push((format!("{name}/answer"), "F32", vec![answer_rows.len(), H], pick(&answer_rows)));
            if with_full {
                tensors.push((format!("full/{name}"), "F32", vec![t, H], f32_bytes(&all)));
            }
        }
        tensors.push(("prefix_rows".into(), "I64", vec![prefix_rows.len()], i64_bytes(prefix_rows.iter().map(|&r| r as i64))));
        tensors.push(("answer_rows".into(), "I64", vec![answer_rows.len()], i64_bytes(answer_rows.iter().map(|&r| r as i64))));
        tensors.push(("token_ids".into(), "I64", vec![t], i64_bytes(b.token_ids.iter().map(|&x| x as i64))));
        let segs = b.segments.iter().flat_map(|s| [s.start as i64, s.len as i64, s.parent as i64, s.pos0 as i64]);
        tensors.push(("segments".into(), "I64", vec![b.segments.len(), 4], i64_bytes(segs)));
        let path = out.join(format!("hidden_{id}.safetensors"));
        write_safetensors(&path, &tensors, &[("request", id.clone()), ("run", "sysone".into())]);
        println!("{id}: T={t}, {} prefix rows, {} answers -> {}", prefix_rows.len(), answer_rows.len(), path.display());
    }
}
