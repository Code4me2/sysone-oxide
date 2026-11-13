//! CPU f32 reference vs qev fp32 (OneJev + serve.patch) on real elephant-1 weights (DESIGN.md §6.1, §9).
//!
//! Ignored by default (a few minutes on 20 cores, ~16 GB of mapped weights). Record the reference first with
//! `reference/hf_check/run_qev.py`, then:
//!
//!     cargo test --release --test cpu_ref -- --ignored --nocapture
//!
//! Gates, fixed before the first measurement: packed vs alone bitwise identical; vs the qev fp32 sequential
//! path max |dp| <= 2e-3 and max |dlogit| <= 5e-2 (fp32-rounding level across 32 layers); top choice agrees on
//! every branch.

use std::path::Path;
use sysone::engine::cpu_ref::{CpuRefEngine, check, slot_ids_from_tokenizer};

const RECORDED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/hf_check/out/qev_fp32.json");

/// The checkpoint: `$SYSONE_MODEL_DIR`, default the GB10 path.
fn model() -> std::path::PathBuf {
    std::env::var_os("SYSONE_MODEL_DIR").map(Into::into).unwrap_or_else(|| "/home/velvet/Desktop/hf_models/elephant-1".into())
}

#[test]
#[ignore = "needs the elephant-1 checkpoint and minutes of CPU"]
fn cpu_ref_matches_qev_fp32() {
    let rec = check::Recorded::load(Path::new(RECORDED)).unwrap();
    let mut engine = CpuRefEngine::load(&model()).unwrap();
    let rep = check::compare(&mut engine, &rec, true).unwrap();
    rep.print();
    assert!(rep.packed_vs_alone_bitwise, "packed and alone differ: {}", rep.packed_vs_alone_logit);
    assert_eq!(rep.top_choice_agree, rep.branches, "top choice differs from qev");
    assert!(rep.prob_vs_seq.max <= 2e-3, "max |dp| vs qev sequential {} > 2e-3", rep.prob_vs_seq.max);
    assert!(rep.logit_vs_seq.max <= 5e-2, "max |dlogit| vs qev sequential {} > 5e-2", rep.logit_vs_seq.max);
}

#[test]
#[ignore = "needs the elephant-1 checkpoint"]
fn slot_ids_match_qev() {
    let rec = check::Recorded::load(Path::new(RECORDED)).unwrap();
    let ids = slot_ids_from_tokenizer(&model()).unwrap();
    assert_eq!(ids, rec.slot_ids);
}
