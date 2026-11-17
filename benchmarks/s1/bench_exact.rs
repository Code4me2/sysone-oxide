//! Qualification harness for bucket replay, legacy bucket eager and exact-token eager execution.
//!
//! `plan` is CPU-only and replays the frozen bulk-workload parser without reserializing `body_json`.
//! `run` emits raw slot logits and host-requested GEMM row dimensions; it never estimates hardware FLOPs.

use anyhow::{Context as _, Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use sysone::api::schema::SystemOneRequest;
use sysone::contract::{DecisionBatch, MAX_SLOTS};
use sysone::engine::config::ModelConfig;
use sysone::engine::gpu::{EngineOptions, ExecutionMode, GpuEngine};
use sysone::engine::platform::require_idle_gpu;
use sysone::frontend::{Frontend, Limits, Plan};
use sysone::kernels::gdn::GdnBackend;

const MAX_TOKENS: usize = 16_384;
const MAX_PREFIXES: usize = 32;
const WORKLOAD_SCHEMA: &str = "sysone-bulk-profile/v1";
const PLAN_SCHEMA: &str = "sysone-exact-plan/v1";
const RUN_SCHEMA: &str = "sysone-exact-forward/v1";

#[derive(Parser, Debug)]
#[command(about = "Exact-token eager qualification plan and raw forward harness")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// CPU-only: validate all frozen, boundary and isolation cases, and reject T16385.
    Plan {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        workload: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// GPU: run one arm in a fresh process and write raw JSONL records.
    Run {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        workload: PathBuf,
        #[arg(long, value_enum)]
        arm: Arm,
        #[arg(long, value_enum, default_value_t = Suite::Frozen)]
        suite: Suite,
        #[arg(long, default_value_t = 1)]
        repeats: usize,
        #[arg(long)]
        output: PathBuf,
        /// Non-authoritative development escape hatch. Qualification must never pass it.
        #[arg(long)]
        allow_shared_gpu: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Arm {
    LegacyGraph,
    LegacyEager,
    ExactEager,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::LegacyGraph => "legacy_graph",
            Self::LegacyEager => "legacy_eager",
            Self::ExactEager => "exact_eager",
        }
    }

    fn options(self) -> (ExecutionMode, bool) {
        match self {
            Self::LegacyGraph => (ExecutionMode::Bucketed, true),
            Self::LegacyEager => (ExecutionMode::Bucketed, false),
            Self::ExactEager => (ExecutionMode::ExactEager, false),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum Suite {
    Frozen,
    Boundary,
    Isolation,
    All,
}

#[derive(Clone)]
pub(crate) struct PlannedRecord {
    id: String,
    plan: Plan,
}

#[derive(Clone)]
pub(crate) struct Case {
    pub(crate) suite: &'static str,
    pub(crate) name: String,
    pub(crate) shape: Option<String>,
    pub(crate) mode: Option<&'static str>,
    pub(crate) relation: Option<String>,
    pub(crate) request_ids: Vec<String>,
    pub(crate) request_first_rows: Vec<usize>,
    pub(crate) batch: DecisionBatch,
}

pub(crate) fn load_workload(path: &Path) -> Result<Value> {
    let workload: Value = serde_json::from_slice(&fs::read(path).with_context(|| format!("reading {}", path.display()))?)?;
    ensure!(workload["schema"].as_str() == Some(WORKLOAD_SCHEMA), "unexpected workload schema");
    ensure!(workload["shapes"].as_object().is_some(), "workload has no shapes object");
    Ok(workload)
}

pub(crate) fn planned_shapes(fe: &Frontend, workload: &Value) -> Result<BTreeMap<String, Vec<PlannedRecord>>> {
    let mut out = BTreeMap::new();
    for (shape, value) in workload["shapes"].as_object().context("shapes")? {
        let mut records = Vec::new();
        for record in value["records"].as_array().with_context(|| format!("{shape}.records"))? {
            // Author key/question/option ordering is semantic. Parse the stored ordered string directly.
            let body = record["body_json"].as_str().with_context(|| format!("{shape} ordered body_json"))?;
            let request = SystemOneRequest::from_body(body.as_bytes()).map_err(|e| anyhow::anyhow!("{shape}: {e}"))?;
            let plan = fe.plan(&request, 1).map_err(|e| anyhow::anyhow!("{shape}: {e}"))?;
            ensure!(plan.total == record["input_tokens"].as_u64().context("input_tokens")? as usize,
                    "{shape}: author token count differs");
            ensure!(plan.prompt_sha256 == record["prompt_sha256"].as_str().context("prompt_sha256")?,
                    "{shape}: author prompt hash differs");
            ensure!(plan.branches.len() == record["branches"].as_u64().context("branches")? as usize,
                    "{shape}: author branch count differs");
            records.push(PlannedRecord { id: record["id"].as_str().context("record id")?.to_string(), plan });
        }
        ensure!(records.len() >= 9, "{shape}: too few frozen records");
        out.insert(shape.clone(), records);
    }
    Ok(out)
}

fn pack(records: &[PlannedRecord], single: bool) -> Result<(DecisionBatch, Vec<String>, Vec<usize>)> {
    let mut batch = DecisionBatch::default();
    let mut ids = Vec::new();
    let mut first = Vec::new();
    // Preserve the frozen profile's selection: records 0..8 were author/Rust cross-check inputs; forwarding starts at 8.
    for record in records.iter().skip(8) {
        if batch.num_tokens() + record.plan.total > MAX_TOKENS || (single && !ids.is_empty()) {
            break;
        }
        first.push(record.plan.push_into(&mut batch)?);
        ids.push(record.id.clone());
    }
    ensure!(!ids.is_empty() && ids.len() <= MAX_PREFIXES, "frozen request set does not fit");
    batch.validate()?;
    Ok((batch, ids, first))
}

fn frozen_cases(shapes: &BTreeMap<String, Vec<PlannedRecord>>) -> Result<Vec<Case>> {
    let mut out = Vec::new();
    for (shape, records) in shapes {
        let (single, ids, first) = pack(records, true)?;
        out.push(Case {
            suite: "frozen",
            name: format!("{shape}/{}", "single"),
            shape: Some(shape.clone()),
            mode: Some("single"),
            relation: None,
            request_ids: ids,
            request_first_rows: first,
            batch: single,
        });
        let (packed, ids, first) = pack(records, false)?;
        if ids.len() > 1 {
            out.push(Case {
                suite: "frozen",
                name: format!("{shape}/{}", "packed"),
                shape: Some(shape.clone()),
                mode: Some("packed"),
                relation: None,
                request_ids: ids,
                request_first_rows: first,
                batch: packed,
            });
        }
    }
    ensure!(out.len() == 16, "frozen workload produced {} nonduplicate layouts, expected 16", out.len());
    Ok(out)
}

fn deterministic_ids(seed: u64, n: usize, vocab: usize) -> Vec<u32> {
    let span = vocab.saturating_sub(1024).max(1);
    let mut x = seed | 1;
    (0..n).map(|_| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (1024 + x as usize % span).min(vocab.saturating_sub(1)) as u32
    }).collect()
}

fn synthetic_case(name: &str, prefix: usize, suffixes: &[usize], vocab: usize, seed: u64) -> Result<Case> {
    let p = deterministic_ids(seed, prefix, vocab);
    let owned: Vec<Vec<u32>> = suffixes.iter().enumerate()
        .map(|(i, &n)| deterministic_ids(seed.wrapping_add(i as u64 + 1), n, vocab))
        .collect();
    let refs: Vec<(&[u32], u16)> = owned.iter().map(|s| (s.as_slice(), 4)).collect();
    let mut batch = DecisionBatch::default();
    batch.push_request(&p, &refs)?;
    batch.validate()?;
    Ok(Case {
        suite: "boundary",
        name: name.into(),
        shape: None,
        mode: None,
        relation: Some("synthetic-real-T-boundary".into()),
        request_ids: vec![name.into()],
        request_first_rows: vec![0],
        batch,
    })
}

fn boundary_cases(vocab: usize) -> Result<Vec<Case>> {
    let mut out = Vec::new();
    for (i, total) in [63usize, 64, 65, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
        8191, 8192, 8193, 16383, 16384].into_iter().enumerate() {
        out.push(synthetic_case(&format!("T{total}"), total - 1, &[1], vocab, 0x5eed + i as u64)?);
    }
    // Many suffixes share one prefix and cross both the 64-row kernel chunk and 256-row bucket boundaries.
    out.push(synthetic_case("many-suffixes-T257-Q64", 193, &vec![1; 64], vocab, 0xabc0)?);
    Ok(out)
}

fn clone_sequence(base: &Case, name: &str, relation: &str) -> Case {
    let mut c = base.clone();
    c.suite = "isolation";
    c.name = name.into();
    c.relation = Some(relation.into());
    c
}

fn reordered_case(name: &str, relation: &str, records: &[&PlannedRecord]) -> Result<Case> {
    let mut batch = DecisionBatch::default();
    let mut ids = Vec::new();
    let mut first = Vec::new();
    for record in records {
        first.push(record.plan.push_into(&mut batch)?);
        ids.push(record.id.clone());
    }
    ensure!(batch.num_tokens() <= MAX_TOKENS, "reordered mixed case exceeds token limit");
    batch.validate()?;
    Ok(Case {
        suite: "isolation",
        name: name.into(),
        shape: None,
        mode: None,
        relation: Some(relation.into()),
        request_ids: ids,
        request_first_rows: first,
        batch,
    })
}

fn isolation_cases(shapes: &BTreeMap<String, Vec<PlannedRecord>>, frozen: &[Case]) -> Result<Vec<Case>> {
    let find = |name: &str| frozen.iter().find(|c| c.name == name).with_context(|| format!("missing frozen case {name}"));
    let long = find("w8000-q8/single")?;
    let short = find("w1000-q1/single")?;
    let many = find("w1000-q8/packed")?;
    let few = find("w4000-q1/single")?;
    let mut out = vec![
        clone_sequence(long, "long-short-long/0-long", "long-short-long"),
        clone_sequence(short, "long-short-long/1-short", "long-short-long"),
        clone_sequence(long, "long-short-long/2-long", "long-short-long"),
        clone_sequence(many, "many-few-many/0-many", "many-few-many"),
        clone_sequence(few, "many-few-many/1-few", "many-few-many"),
        clone_sequence(many, "many-few-many/2-many", "many-few-many"),
    ];
    let a = shapes["w1000-q1"].get(8).context("w1000-q1 record 8")?;
    let b = shapes["w1000-q8"].get(8).context("w1000-q8 record 8")?;
    let c = shapes["w1000-q24"].get(8).context("w1000-q24 record 8")?;
    out.push(reordered_case("mixed-reordered/abc", "mixed-reordered", &[a, b, c])?);
    out.push(reordered_case("mixed-reordered/cab", "mixed-reordered", &[c, a, b])?);
    Ok(out)
}

pub(crate) fn capacity_for(batch: &DecisionBatch) -> Result<sysone::engine::graph::Bucket> {
    let plan = sysone::engine::graph::BucketPlan::new(&EngineOptions::buckets_up_to(MAX_TOKENS), 2048, 2048)?;
    plan.pick(batch.num_tokens(), batch.num_branches(), batch.segments.len()).context("no capacity bucket")
}

/// Atomically reserve a fresh output before loading metadata or touching CUDA. Existing files and symlinks
/// are rejected; a later failure may leave an empty/partial new file, never overwrite previous evidence.
fn new_output(path: &Path) -> Result<BufWriter<File>> {
    if let Some(parent) = path.parent() { fs::create_dir_all(parent)?; }
    let file = File::options().write(true).create_new(true).open(path)
        .with_context(|| format!("creating new output {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn plan_command(model: &Path, workload_path: &Path, output: &Path) -> Result<()> {
    let mut writer = new_output(output)?;
    let fe = Frontend::load(model, Limits { max_branch_tokens: MAX_TOKENS, max_request_tokens: MAX_TOKENS })?;
    let workload = load_workload(workload_path)?;
    let shapes = planned_shapes(&fe, &workload)?;
    let cfg = ModelConfig::load(model)?;
    let cases = selected_cases(Suite::All, &shapes, cfg.text_config.vocab_size)?;
    let illegal = synthetic_case("T16385", 16384, &[1], cfg.text_config.vocab_size, 1)?;
    ensure!(capacity_for(&illegal.batch).is_err(), "T16385 must be refused by CPU admission");
    let layouts: Vec<Value> = cases.iter().enumerate().map(|(sequence, case)| {
        let b = capacity_for(&case.batch).unwrap();
        json!({
            "sequence": sequence,
            "suite": case.suite,
            "name": case.name,
            "shape": case.shape,
            "mode": case.mode,
            "relation": case.relation,
            "requests": case.request_ids.len(),
            "request_ids": case.request_ids,
            "request_first_rows": case.request_first_rows,
            "request_branch_counts": branch_counts(case),
            "answer_rows": case.batch.answer_rows,
            "answer_slots": case.batch.answer_slots,
            "tokens": case.batch.num_tokens(),
            "answers": case.batch.num_branches(),
            "segments": case.batch.segments.len(),
            "prefixes": case.batch.segments.iter().filter(|s| s.is_prefix()).count(),
            "capacity": {"tokens": b.tokens, "answers": b.answers, "segments": b.segments},
            "padding_tokens": b.tokens - case.batch.num_tokens(),
        })
    }).collect();
    let doc = json!({
        "schema": PLAN_SCHEMA,
        "workload_schema": WORKLOAD_SCHEMA,
        "layout_count": 16,
        "case_count": layouts.len(),
        "illegal_T16385_cpu_admission_rejected": true,
        "layouts": layouts,
    });
    serde_json::to_writer_pretty(&mut writer, &doc)?;
    writer.flush()?;
    println!("EXACT_PLAN_PASS: 16 nonduplicate layouts, {} total cases; T16385 CPU admission rejected", cases.len());
    Ok(())
}

pub(crate) fn selected_cases(suite: Suite, shapes: &BTreeMap<String, Vec<PlannedRecord>>, vocab: usize) -> Result<Vec<Case>> {
    let frozen = frozen_cases(shapes)?;
    Ok(match suite {
        Suite::Frozen => frozen,
        Suite::Boundary => boundary_cases(vocab)?,
        Suite::Isolation => isolation_cases(shapes, &frozen)?,
        Suite::All => {
            let mut all = frozen.clone();
            all.extend(boundary_cases(vocab)?);
            all.extend(isolation_cases(shapes, &frozen)?);
            all
        }
    })
}

pub(crate) fn branch_counts(case: &Case) -> Vec<usize> {
    case.request_first_rows.iter().enumerate().map(|(i, start)| {
        case.request_first_rows.get(i + 1).copied().unwrap_or(case.batch.num_branches()) - start
    }).collect()
}

fn raw_logits(batch: &DecisionBatch, logits: &sysone::contract::SlotLogits) -> Vec<Vec<f32>> {
    (0..batch.num_branches()).map(|row| {
        let n = batch.answer_slots[row] as usize;
        logits.row(row, n.min(MAX_SLOTS)).to_vec()
    }).collect()
}

/// Flush the raw diagnostics before rejecting nonfinite active logits; JSON numbers alone lose their bits.
fn write_run_record(writer: &mut impl Write, mut record: Value, logits: &[Vec<f32>]) -> Result<()> {
    let bits: Vec<Vec<u32>> = logits.iter().map(|row| row.iter().map(|v| v.to_bits()).collect()).collect();
    record["logits"] = json!(logits);
    record["logits_bits"] = json!(bits);
    writeln!(writer, "{record}")?;
    writer.flush()?;
    ensure!(logits.iter().flatten().all(|v| v.is_finite()), "nonfinite active logits (bits preserved)");
    Ok(())
}

fn run_command(model: &Path, workload_path: &Path, arm: Arm, suite: Suite, repeats: usize, output: &Path,
               allow_shared_gpu: bool) -> Result<()> {
    ensure!(repeats > 0, "--repeats must be positive");
    let mut writer = new_output(output)?;
    let idle_note = require_idle_gpu(allow_shared_gpu)?;
    let cfg = ModelConfig::load(model)?;
    let fe = Frontend::load(model, Limits { max_branch_tokens: MAX_TOKENS, max_request_tokens: MAX_TOKENS })?;
    let workload = load_workload(workload_path)?;
    let shapes = planned_shapes(&fe, &workload)?;
    let cases = selected_cases(suite, &shapes, cfg.text_config.vocab_size)?;
    let (execution_mode, cuda_graphs) = arm.options();
    let opts = EngineOptions {
        buckets: EngineOptions::buckets_up_to(MAX_TOKENS),
        max_prefixes: MAX_PREFIXES,
        cuda_graphs,
        execution_mode,
        warmup: false,
        gdn_backend: Some(GdnBackend::Tile),
        ..Default::default()
    };
    let mut engine = GpuEngine::load(model, &fe.slot_ids, opts)?;
    ensure!(engine.gdn_backend() == GdnBackend::Tile, "qualification harness requires Tile GDN");
    writeln!(writer, "{}", json!({
        "schema": RUN_SCHEMA,
        "record_type": "engine",
        "arm": arm.name(),
        "suite": format!("{suite:?}").to_lowercase(),
        "execution_mode": execution_mode.name(cuda_graphs),
        "cuda_graphs": cuda_graphs,
        "gdn_backend": engine.gdn_backend().name(),
        "arch": engine.arch,
        "idle_note": idle_note,
        "buckets": EngineOptions::buckets_up_to(MAX_TOKENS),
        "max_prefixes": MAX_PREFIXES,
        "case_count": cases.len(),
        "repeats": repeats,
        "raw_logits": true,
        "gemm_rows_are_host_requested_dimensions_not_hardware_flops": true,
    }))?;

    // Prewarm every concrete case before preserving the measured suite order. Graph mode captures each capacity
    // class here; exact mode JITs arbitrary logical T without ever capturing.
    for case in &cases {
        let bucket = engine.pick_bucket(&case.batch, 0).context("prewarm bucket")?;
        engine.step(&case.batch, bucket, arm == Arm::LegacyGraph)?;
        ensure!(!engine.last.replayed, "prewarm unexpectedly replayed");
    }

    let mut records = 0usize;
    for repeat in 0..repeats {
        for (sequence, case) in cases.iter().enumerate() {
            let bucket = engine.pick_bucket(&case.batch, 0).context("run bucket")?;
            let started = Instant::now();
            let logits = engine.step(&case.batch, bucket, false)?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let expected_replay = arm == Arm::LegacyGraph;
            ensure!(engine.last.replayed == expected_replay,
                    "{} replay={} expected {expected_replay}", case.name, engine.last.replayed);
            ensure!(engine.last.bucket_tokens == bucket.tokens && engine.last.bucket_answers == bucket.answers
                    && engine.last.bucket_segments == bucket.segments,
                    "{} capacity dimensions differ", case.name);
            ensure!(engine.last.logical_tokens == case.batch.num_tokens()
                    && engine.last.logical_answers == case.batch.num_branches(),
                    "{} actual logical dimensions differ", case.name);
            if arm == Arm::ExactEager {
                ensure!(engine.last.execution_tokens == case.batch.num_tokens()
                        && engine.last.execution_answers == case.batch.num_branches(),
                        "{} exact execution dimensions differ", case.name);
            } else {
                ensure!(engine.last.execution_tokens == bucket.tokens && engine.last.execution_answers == bucket.answers,
                        "{} legacy execution dimensions differ", case.name);
            }
            let gemm_rows: Vec<Value> = engine.last_gemm_requests()?.into_iter().map(|g| json!({
                "op_index": g.op_index, "label": g.label, "row_kind": g.row_kind, "rows": g.rows,
            })).collect();
            let record = json!({
                "schema": RUN_SCHEMA,
                "record_type": "run",
                "arm": arm.name(),
                "suite": case.suite,
                "name": case.name,
                "shape": case.shape,
                "mode": case.mode,
                "relation": case.relation,
                "repeat": repeat,
                "sequence": sequence,
                "request_ids": case.request_ids,
                "request_first_rows": case.request_first_rows,
                "request_branch_counts": branch_counts(case),
                "logical": {
                    "tokens": case.batch.num_tokens(),
                    "answers": case.batch.num_branches(),
                    "segments": case.batch.segments.len(),
                    "prefixes": case.batch.segments.iter().filter(|s| s.is_prefix()).count(),
                },
                "capacity": {"tokens": bucket.tokens, "answers": bucket.answers, "segments": bucket.segments},
                "step": {
                    "execution_mode": engine.last.execution_mode.name(cuda_graphs),
                    "capacity_tokens": engine.last.bucket_tokens,
                    "capacity_answers": engine.last.bucket_answers,
                    "capacity_segments": engine.last.bucket_segments,
                    "logical_tokens": engine.last.logical_tokens,
                    "logical_answers": engine.last.logical_answers,
                    "logical_segments": engine.last.logical_segments,
                    "execution_tokens": engine.last.execution_tokens,
                    "execution_answers": engine.last.execution_answers,
                    "execution_segments": engine.last.execution_segments,
                    "replayed": engine.last.replayed,
                    "steps": engine.last.steps,
                    "elapsed_ms": elapsed_ms,
                },
                "answer_rows": case.batch.answer_rows,
                "answer_slots": case.batch.answer_slots,
                "gemm_rows": gemm_rows,
            });
            write_run_record(&mut writer, record, &raw_logits(&case.batch, &logits))?;
            records += 1;
            println!("{} {} repeat {repeat}: T={} Q={} capacity={} replay={} {:.2} ms", arm.name(), case.name,
                     case.batch.num_tokens(), case.batch.num_branches(), bucket.tokens, engine.last.replayed, elapsed_ms);
        }
    }
    println!("EXACT_FORWARD_COMPLETE: arm={} records={records}", arm.name());
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Plan { model, workload, output } => plan_command(&model, &workload, &output),
        Command::Run { model, workload, arm, suite, repeats, output, allow_shared_gpu } => {
            run_command(&model, &workload, arm, suite, repeats, &output, allow_shared_gpu)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct OutputFixture(PathBuf);
    impl OutputFixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!("bench-exact-output-{}-{}",
                std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for OutputFixture {
        fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
    }

    fn assert_commands_reject_output(output: &Path) {
        let missing_model = output.parent().unwrap().join("nonexistent-model");
        let missing_workload = output.parent().unwrap().join("nonexistent-workload");
        let plan_err = plan_command(&missing_model, &missing_workload, output).unwrap_err();
        assert!(plan_err.to_string().starts_with("creating new output "), "{plan_err:#}");
        let run_err = run_command(&missing_model, &missing_workload, Arm::ExactEager, Suite::Frozen,
            1, output, false).unwrap_err();
        assert!(run_err.to_string().starts_with("creating new output "), "{run_err:#}");
    }

    #[test]
    fn output_existing_content_is_untouched_for_helper_plan_and_run() {
        let fixture = OutputFixture::new();
        let path = fixture.0.join("existing.json");
        let original = b"preserved qualification output";
        fs::write(&path, original).unwrap();
        assert!(new_output(&path).is_err());
        assert_commands_reject_output(&path);
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn output_helper_reserves_and_writes_new_file_only_once() {
        let fixture = OutputFixture::new();
        let path = fixture.0.join("nested").join("result.json");
        let mut writer = new_output(&path).unwrap();
        assert!(new_output(&path).is_err());
        writer.write_all(b"new output").unwrap();
        writer.flush().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new output");
    }

    #[cfg(unix)]
    #[test]
    fn output_symlinks_are_rejected_without_touching_targets() {
        let fixture = OutputFixture::new();
        let target = fixture.0.join("target.json");
        let link = fixture.0.join("link.json");
        fs::write(&target, b"preserved target").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(new_output(&link).is_err());
        assert_commands_reject_output(&link);
        assert!(fs::symlink_metadata(&link).unwrap().is_symlink());
        assert_eq!(fs::read(&target).unwrap(), b"preserved target");
        fs::remove_file(&target).unwrap();
        assert!(new_output(&link).is_err());
        assert_commands_reject_output(&link);
        assert!(fs::symlink_metadata(&link).unwrap().is_symlink());
        assert!(!target.exists());
    }

    fn serialization_metadata() -> Value {
        json!({"schema": RUN_SCHEMA, "record_type": "run", "name": "T63", "sequence": 0})
    }

    fn assert_nonfinite_serialization(bits: u32) {
        let logits = vec![vec![1.25, f32::from_bits(bits), -0.0], vec![-3.5]];
        let mut writer = BufWriter::new(Vec::new());
        let error = write_run_record(&mut writer, serialization_metadata(), &logits).unwrap_err();
        assert_eq!(error.to_string(), "nonfinite active logits (bits preserved)");
        // Inspect the underlying sink before drop: even a rejected record must already be flushed.
        assert!(writer.buffer().is_empty());
        let bytes = writer.get_ref();
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1);
        let record: Value = serde_json::from_slice(bytes).unwrap();
        let mut expected = serialization_metadata();
        expected["logits"] = json!([[1.25, null, -0.0], [-3.5]]);
        expected["logits_bits"] = json!([[1.25f32.to_bits(), bits, (-0.0f32).to_bits()], [(-3.5f32).to_bits()]]);
        assert_eq!(record, expected);
        assert!(!String::from_utf8_lossy(bytes).contains("EXACT_FORWARD_COMPLETE"));
    }

    #[test]
    fn serialization_nan_preserves_payload_and_fails_after_flush() {
        assert_nonfinite_serialization(0x7fc1_2345);
    }

    #[test]
    fn serialization_positive_infinity_preserves_bits_and_fails_after_flush() {
        assert_nonfinite_serialization(f32::INFINITY.to_bits());
    }

    #[test]
    fn serialization_negative_infinity_preserves_bits_and_fails_after_flush() {
        assert_nonfinite_serialization(f32::NEG_INFINITY.to_bits());
    }

    #[test]
    fn serialization_finite_values_and_signed_zero_are_preserved() {
        let logits = vec![vec![1.25, -0.0, 0.0], vec![-3.5, f32::MIN_POSITIVE, f32::from_bits(1)]];
        let mut writer = BufWriter::new(Vec::new());
        write_run_record(&mut writer, serialization_metadata(), &logits).unwrap();
        assert!(writer.buffer().is_empty());
        let record: Value = serde_json::from_slice(writer.get_ref()).unwrap();
        // The existing JSON value representation is unchanged; the bit rows are additive diagnostics.
        assert_eq!(record["logits"], json!(logits));
        assert_eq!(record["schema"], RUN_SCHEMA);
        assert_eq!(record["name"], "T63");
        assert_eq!(record["sequence"], 0);
        for (r, row) in logits.iter().enumerate() {
            for (s, value) in row.iter().enumerate() {
                assert_eq!(record["logits_bits"][r][s].as_u64().unwrap(), value.to_bits() as u64);
                assert_eq!((record["logits"][r][s].as_f64().unwrap() as f32).to_bits(), value.to_bits());
            }
        }
    }

    #[test]
    fn synthetic_boundaries_have_exact_requested_lengths() {
        let cases = boundary_cases(200_000).unwrap();
        let totals: Vec<usize> = cases.iter().map(|c| c.batch.num_tokens()).collect();
        assert_eq!(&totals[..9], &[63, 64, 65, 255, 256, 257, 511, 512, 513]);
        assert_eq!(&totals[9..23], &[1023, 1024, 1025, 2047, 2048, 2049,
            4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384]);
        assert_eq!(cases[23].batch.num_tokens(), 257);
        assert_eq!(cases[23].batch.num_branches(), 64);
        assert_eq!(cases[23].batch.segments.iter().filter(|s| s.is_prefix()).count(), 1);
        assert!(cases.iter().all(|c| capacity_for(&c.batch).is_ok()));
        let illegal = synthetic_case("T16385", 16384, &[1], 200_000, 1).unwrap();
        assert!(capacity_for(&illegal.batch).is_err());
    }
}
