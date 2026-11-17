//! Explicitly ignored GPU prerequisite. Never part of CPU execution. Run once in its own fresh process
//! only after independent review + main, under the owned GPU2 lock/container/startup deadline.
use cutile::tile_kernel::jit_compile_count;
use std::path::PathBuf;
use sysone::{contract::DecisionEngine, engine::{gpu::{EngineOptions, ExecutionMode, GpuEngine}, warmup},
    frontend::{Frontend, Limits}, kernels::gdn::GdnBackend};

#[test]
#[ignore = "GPU2 only; independent review and main readiness approval required"]
fn bounded_first_startup_51_calls_no_capture_validation_and_repeat_misses() {
    assert_eq!(jit_compile_count(), 0, "must be the first GPU test in a fresh process");
    let model = PathBuf::from(std::env::var("SYSONE_MODEL_DIR").expect("model path"));
    let fe = Frontend::load(&model, Limits { max_branch_tokens:16384, max_request_tokens:16384 }).unwrap();
    let opts = EngineOptions { buckets:EngineOptions::buckets_up_to(16384), max_prefixes:32,
        warmup:true, bounded_exact_warmup:true, execution_mode:ExecutionMode::ExactEager,
        cuda_graphs:false, gdn_backend:Some(GdnBackend::Tile), ..Default::default() };
    let mut e = GpuEngine::load(&model,&fe.slot_ids,opts).unwrap();
    let r = e.warmup_report().unwrap();
    assert!(r.complete && r.warmup_ms < 90000.);
    assert_eq!((r.calls.len(), r.standard_calls, r.coverage_calls, r.validation_calls), (51,1,25,25));
    assert_eq!(r.validation_new_installations, 0);
    assert_eq!(e.captured_bucket_count(), 0);
    let before = jit_compile_count();
    for layout in warmup::plan() {
        let b = layout.batch().unwrap();
        let logits = e.run(&b).unwrap();
        assert!(logits.data.iter().all(|x|x.is_finite()));
        assert!(!e.last.replayed && e.last.steps == 1);
        assert_eq!((e.last.execution_tokens,e.last.execution_answers,e.last.execution_segments),
            (layout.tokens,layout.answers,layout.segments));
        assert_eq!(e.last_gemm_requests().unwrap().len(),137);
        assert_eq!(jit_compile_count(),before);
        assert_eq!(e.captured_bucket_count(),0);
    }
}
