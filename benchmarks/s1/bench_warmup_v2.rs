//! Bounded warmup qualification. Plan is CPU-only; run/profile require independent review and main.
//! Reuse the candidate copy of the reviewed exact planner, never mutate or import Python into old E/D.
#[allow(dead_code)]
#[path = "bench_exact.rs"]
mod exact;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand, ValueEnum};
use cutile::tile_kernel::jit_compile_count;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::{self, File}, io::{BufWriter, Write}, path::{Path, PathBuf}, time::Instant};
use sysone::{contract::DecisionBatch, engine::{config::ModelConfig, goldens,
    gpu::{EngineOptions, ExecutionMode, GpuEngine}, platform::require_idle_gpu, warmup},
    frontend::{Frontend, Limits}, kernels::gdn::GdnBackend};

#[derive(Parser)]
#[command(about = "Fixed bounded exact warmup CPU manifest / raw GPU qualification / instrumented eager profile")]
struct Args { #[command(subcommand)] command: Command }
#[derive(Subcommand)]
enum Command {
    Plan { #[arg(long)] model: PathBuf, #[arg(long)] workload: PathBuf, #[arg(long)] output: PathBuf },
    Run { #[arg(long)] model: PathBuf, #[arg(long)] workload: PathBuf,
        #[arg(long, value_enum)] arm: Arm, #[arg(long)] output: PathBuf },
    Profile { #[arg(long)] model: PathBuf, #[arg(long)] workload: PathBuf, #[arg(long)] output: PathBuf },
}
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Arm { OldExact, Bounded }
impl Arm { fn name(self) -> &'static str { if self == Self::Bounded { "bounded" } else { "old_exact" } } }
fn options(arm: Arm) -> EngineOptions {
    EngineOptions { buckets: EngineOptions::buckets_up_to(16384), max_prefixes: 32,
        execution_mode: ExecutionMode::ExactEager, cuda_graphs: false, warmup: true,
        bounded_exact_warmup: arm == Arm::Bounded, gdn_backend: Some(GdnBackend::Tile), ..Default::default() }
}
fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn output(path: &Path) -> Result<BufWriter<File>> {
    if let Some(p) = path.parent() { fs::create_dir_all(p)?; }
    Ok(BufWriter::new(File::options().write(true).create_new(true).open(path)?))
}
fn emit(w: &mut impl Write, value: &Value) -> Result<()> { writeln!(w, "{value}")?; w.flush()?; Ok(()) }
fn ns(epoch: Instant) -> u64 { epoch.elapsed().as_nanos().try_into().unwrap() }

fn cases(model: &Path, workload: &Path) -> Result<(Frontend, Vec<exact::Case>)> {
    let fe = Frontend::load(model, Limits { max_branch_tokens: 16384, max_request_tokens: 16384 })?;
    let cfg = ModelConfig::load(model)?;
    let w = exact::load_workload(workload)?;
    let shapes = exact::planned_shapes(&fe, &w)?;
    let mut cases = exact::selected_cases(exact::Suite::All, &shapes, cfg.text_config.vocab_size)?;
    ensure!(cases.len() == 48, "original 48-case closure");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let gold = goldens::golden_run(repo, "bf16_batched")?;
    let corpus = goldens::model_corpus(repo)?;
    let development = goldens::plan_corpus(&fe, &corpus, |id| gold.get(id).is_some_and(|g| g.status == 200))?;
    ensure!(development.len() == 47, "47 development contexts required, not held-out release evidence");
    for p in development {
        let mut batch = DecisionBatch::default();
        let first = p.plan.push_into(&mut batch)?;
        batch.validate()?;
        cases.push(exact::Case { suite: "development", name: format!("development/{}",p.case.id),
            shape: None, mode: Some("single"), relation: None, request_ids: vec![p.case.id],
            request_first_rows: vec![first], batch });
    }
    for c in &cases {
        exact::capacity_for(&c.batch)?;
        ensure!(c.batch.segments.iter().filter(|s| s.is_prefix()).count() <= 32, "prefix capacity");
    }
    Ok((fe, cases))
}

fn layout(c: &exact::Case) -> Result<Value> {
    let b = exact::capacity_for(&c.batch)?;
    Ok(json!({"suite":c.suite,"name":c.name,"shape":c.shape,"mode":c.mode,"relation":c.relation,
        "request_ids":c.request_ids,"request_first_rows":c.request_first_rows,
        "request_branch_counts":exact::branch_counts(c),
        "logical":{"tokens":c.batch.num_tokens(),"answers":c.batch.num_branches(),
            "segments":c.batch.segments.len(),"prefixes":c.batch.segments.iter().filter(|s|s.is_prefix()).count()},
        "capacity":{"tokens":b.tokens,"answers":b.answers,"segments":b.segments},
        "token_ids":c.batch.token_ids,"segments":c.batch.segments.iter().map(|s|json!({
            "start":s.start,"len":s.len,"parent":s.parent,"pos0":s.pos0})).collect::<Vec<_>>(),
        "answer_rows":c.batch.answer_rows,"answer_slots":c.batch.answer_slots}))
}
fn profile_cases(cases: &[exact::Case]) -> Result<Vec<&exact::Case>> {
    let mut out = Vec::new();
    for words in [1000,4000,8000,12000] { for questions in [8,24] {
        let shape = format!("w{words}-q{questions}");
        let matching: Vec<_> = cases.iter().filter(|c|c.suite == "frozen" && c.shape.as_deref() == Some(&shape)).collect();
        let c = matching.iter().copied().find(|c|c.mode == Some("packed"))
            .or_else(||matching.first().copied()).context("missing representative frozen shape")?;
        out.push(c);
    }}
    ensure!(out.len() == 8, "eight representative frozen layouts");
    Ok(out)
}
fn plan(model: &Path, workload: &Path, path: &Path) -> Result<()> {
    ensure!(jit_compile_count() == 0, "CPU planning must not install kernels");
    let (_, cases) = cases(model, workload)?;
    options(Arm::Bounded).validate_bounded_exact_warmup()?;
    let layouts = cases.iter().map(layout).collect::<Result<Vec<_>>>()?;
    let profiles = profile_cases(&cases)?.iter().map(|c|c.name.clone()).collect::<Vec<_>>();
    let cap = sysone::engine::graph::BucketPlan::new(&EngineOptions::buckets_up_to(16384),2048,2048)?;
    ensure!(cap.pick(16385,1,2).is_none(), "T16385 must be rejected");
    emit(&mut output(path)?, &json!({"schema":"sysone-warmup-plan/v2","workload_sha256":sha(&fs::read(workload)?),
        "warmup_policy":warmup::POLICY,"warmup_plan":warmup::plan(),"startup_calls":51,
        "extra_calls":50,"wall_bound_ms":90000,"case_count":layouts.len(),"layouts":layouts,
        "profile_layouts":profiles,"profile_samples_per_layout":3,
        "profile_kind":"instrumented_eager_not_exact_http_decomposition","T16385_rejected":true}))?;
    ensure!(jit_compile_count() == 0, "CPU manifest installed a kernel");
    println!("WARMUP_CPU_PLAN_PASS: 48 original +47 development cases; 51 startup calls; 8x3 profiles");
    Ok(())
}

fn run(model: &Path, workload: &Path, arm: Arm, profile: bool, path: &Path) -> Result<()> {
    let epoch = Instant::now();
    ensure!(jit_compile_count() == 0, "fresh process required");
    ensure!(std::env::var("CUTILE_JIT_TIMING").as_deref() == Ok("1"), "CUTILE_JIT_TIMING=1 required");
    let (fe, cases) = cases(model, workload)?;
    require_idle_gpu(false)?;
    let mut w = output(path)?;
    let before = jit_compile_count();
    let begin_ns = ns(epoch);
    eprintln!("WARMUP_V2_STARTUP_BEGIN {}",json!({"arm":arm.name(),"begin_ns":begin_ns,"jit_before":before}));
    let loaded = GpuEngine::load(model, &fe.slot_ids, options(arm));
    let end_ns = ns(epoch);
    let after = jit_compile_count();
    eprintln!("WARMUP_V2_STARTUP_END {}",json!({"arm":arm.name(),"end_ns":end_ns,"jit_after":after,"ok":loaded.is_ok()}));
    let mut engine = loaded?;
    ensure!(engine.arch == "sm_120" && engine.gdn_backend() == GdnBackend::Tile, "qualified Tile architecture");
    emit(&mut w,&json!({"schema":"sysone-warmup-run/v2","record_type":"engine","arm":arm.name(),
        "profile":profile,"begin_ns":begin_ns,"end_ns":end_ns,"jit_before":before,"jit_after":after,
        "startup_ms":(end_ns-begin_ns) as f64 / 1e6,"report":engine.warmup_report(),
        "arch":engine.arch,"captured_buckets":engine.captured_bucket_count(),
        "workload_sha256":sha(&fs::read(workload)?),"arena_bytes":engine.arena_bytes()}))?;
    ensure!(engine.captured_bucket_count() == 0, "exact startup captured graphs");
    if arm == Arm::Bounded {
        let r = engine.warmup_report().context("missing bounded report")?;
        ensure!(r.complete && r.calls.len() == 51 && r.validation_calls == 25
            && r.validation_new_installations == 0, "bounded startup proof");
    }
    let selected = if profile { profile_cases(&cases)? } else { cases.iter().collect() };
    for (sequence,c) in selected.iter().enumerate() {
        let b = engine.pick_bucket(&c.batch,0).context("case bucket")?;
        for sample in 0..if profile {3} else {1} {
            let before = jit_compile_count();
            let begin_ns = ns(epoch);
            eprintln!("WARMUP_V2_STEP_BEGIN {}",json!({"sequence":sequence,"sample":sample,"name":c.name,
                "begin_ns":begin_ns,"jit_before":before}));
            let result = if profile {
                engine.profile(&c.batch,b).map(|(ops,total)| (None,Some((ops,total))))
            } else { engine.step(&c.batch,b,false).map(|l|(Some(l),None)) };
            let end_ns = ns(epoch);
            let after = jit_compile_count();
            eprintln!("WARMUP_V2_STEP_END {}",json!({"sequence":sequence,"sample":sample,"end_ns":end_ns,
                "jit_after":after,"ok":result.is_ok()}));
            let (logits, prof) = result?;
            // profile() does not update last; direct traces use last's actual shape, profile trace is emitted
            // from a CPU executor-independent production graph walk using the manifest's T/Q dimensions.
            let trace = if !profile { engine.last_gemm_requests()?.into_iter().map(|g|json!({
                "op_index":g.op_index,"label":g.label,"row_kind":g.row_kind,"rows":g.rows})).collect::<Vec<_>>() }
                else { profile_gemm_trace(&engine,&c.batch) };
            let raw = logits.as_ref().map(|l|(0..c.batch.num_branches()).map(|r|
                l.row(r,c.batch.answer_slots[r] as usize).to_vec()).collect::<Vec<_>>());
            let bits = raw.as_ref().map(|r|r.iter().map(|row|row.iter().map(|v|v.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>());
            emit(&mut w,&json!({"schema":"sysone-warmup-run/v2","record_type":if profile {"profile"} else {"step"},
                "arm":arm.name(),"sequence":sequence,"sample":sample,"layout":layout(c)?,
                "begin_ns":begin_ns,"end_ns":end_ns,"wall_ms":(end_ns-begin_ns) as f64/1e6,
                "jit_before":before,"jit_after":after,"jit_delta":after-before,
                "captured_buckets":engine.captured_bucket_count(),"gemm_rows":trace,
                "execution":{"tokens":c.batch.num_tokens(),"answers":c.batch.num_branches(),"segments":c.batch.segments.len()},
                "replayed":if profile {false} else {engine.last.replayed},"logits":raw,"logits_bits":bits,
                "profile_kind":if profile {Some("instrumented_eager_not_exact_http_decomposition")} else {None},
                "profile_ops":prof.as_ref().map(|p|&p.0),"profile_total_ms":prof.as_ref().map(|p|p.1)}))?;
            // Preserve raw data before any qualification failure. Never prewarm the concrete cases/retry.
            ensure!(trace.len() == 137, "full 137-GEMM trace required");
            ensure!(raw.as_ref().is_none_or(|r|r.iter().flatten().all(|v|v.is_finite())), "nonfinite active logits (bits preserved)");
            ensure!(engine.captured_bucket_count() == 0, "exact case captured a graph");
            if !profile {
                ensure!(!engine.last.replayed && engine.last.steps == 1
                    && engine.last.execution_tokens == c.batch.num_tokens()
                    && engine.last.execution_answers == c.batch.num_branches()
                    && engine.last.execution_segments == c.batch.segments.len(), "exact StepInfo drift");
            }
            ensure!(arm != Arm::Bounded || after == before, "post-ready cuTile L1 installation: {} sample {sample}",c.name);
        }
    }
    emit(&mut w,&json!({"record_type":"complete","case_count":selected.len(),
        "samples_per_case":if profile {3} else {1},"jit_at_ready":after,"jit_final":jit_compile_count(),
        "qualification_requires_independent_fidelity_and_native_log_reconciliation":true}))?;
    Ok(())
}
fn profile_gemm_trace(e: &GpuEngine, b: &DecisionBatch) -> Vec<Value> {
    use sysone::engine::graph::{GraphOp, WeightRef};
    e.fwd.graph.ops.iter().enumerate().filter_map(|(op_index,op)| match op {
        GraphOp::Gemm {w,..} => {
            let answers = *w == WeightRef::SlotHead;
            Some(json!({"op_index":op_index,"label":op.profile_label(),
                "row_kind":if answers {"answers"} else {"tokens"},
                "rows":if answers {b.num_branches()} else {b.num_tokens()}}))
        }, _ => None,
    }).collect()
}
fn main() -> Result<()> {
    match Args::parse().command {
        Command::Plan {model,workload,output} => plan(&model,&workload,&output),
        Command::Run {model,workload,arm,output} => run(&model,&workload,arm,false,&output),
        Command::Profile {model,workload,output} => run(&model,&workload,Arm::Bounded,true,&output),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_cache_and_suite_flags_are_rejected_not_silently_unused() {
        for flag in ["--cache-class", "--suite", "--retries", "--allow-shared-gpu"] {
            assert!(Args::try_parse_from(["bench_warmup_v2","run","--model","m","--workload","w",
                "--arm","bounded","--output","o",flag,"x"]).is_err());
        }
    }
    #[test]
    fn arm_options_only_change_warmup_policy() {
        assert!(!options(Arm::OldExact).bounded_exact_warmup);
        let o = options(Arm::Bounded); o.validate_bounded_exact_warmup().unwrap();
        assert!(o.warmup && !o.cuda_graphs && o.max_prefixes == 32);
    }
    #[test]
    fn output_is_create_new() {
        let path = std::env::temp_dir().join(format!("sysone-warmup-no-overwrite-{}",std::process::id()));
        let _f = output(&path).unwrap();
        assert!(output(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
