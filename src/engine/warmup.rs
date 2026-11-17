//! Fixed specialization-coverage warmup. No workload-dependent/adaptive shapes or request fallback.
use crate::contract::DecisionBatch;
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::time::Duration;

pub const POLICY: &str = "bounded_exact_v2";
pub const WALL_LIMIT: Duration = Duration::from_secs(90);
pub const TOKENS: [usize; 5] = [65, 66, 68, 72, 80];
pub const ANSWERS: [usize; 5] = [1, 2, 4, 8, 16];

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct WarmupLayout {
    pub phase: &'static str,
    pub tokens: usize,
    pub answers: usize,
    pub segments: usize,
    pub prefixes: usize,
    pub prefix_tokens: usize,
    pub suffix_tokens: usize,
    pub readout_slots: u16,
}

impl WarmupLayout {
    pub fn batch(&self) -> Result<DecisionBatch> {
        let ids: Vec<u32> = (0..self.tokens).map(|i| 1000 + i as u32).collect();
        let suffixes: Vec<(&[u32], u16)> = ids[self.prefix_tokens..].chunks_exact(self.suffix_tokens)
            .map(|s| (s, self.readout_slots)).collect();
        let mut batch = DecisionBatch::default();
        batch.push_request(&ids[..self.prefix_tokens], &suffixes)?;
        batch.validate()?;
        ensure!(batch.num_tokens() == self.tokens && batch.num_branches() == self.answers
            && batch.segments.len() == self.segments, "warmup layout mismatch");
        Ok(batch)
    }
}

/// Unchanged standard batch, then T-major coverage, then the identical lattice for validation.
pub fn plan() -> Vec<WarmupLayout> {
    let mut out = vec![WarmupLayout { phase: "standard", tokens: 100, answers: 1, segments: 2,
        prefixes: 1, prefix_tokens: 60, suffix_tokens: 40, readout_slots: 2 }];
    for phase in ["coverage", "validation"] {
        for tokens in TOKENS {
            for answers in ANSWERS {
                out.push(WarmupLayout { phase, tokens, answers, segments: answers + 1, prefixes: 1,
                    prefix_tokens: tokens - answers, suffix_tokens: 1, readout_slots: 2 });
            }
        }
    }
    out
}

#[derive(Clone, Debug, Serialize)]
pub struct WarmupCall {
    pub sequence: usize,
    pub layout: WarmupLayout,
    pub begin_ms: f64,
    pub end_ms: f64,
    pub jit_before: u64,
    pub jit_after: u64,
    pub ok: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct WarmupReport {
    pub policy: &'static str,
    pub plan: Vec<WarmupLayout>,
    pub wall_limit_ms: u64,
    pub calls: Vec<WarmupCall>,
    pub standard_calls: usize,
    pub coverage_calls: usize,
    pub validation_calls: usize,
    pub validation_new_installations: u64,
    pub jit_before: u64,
    pub jit_after: u64,
    pub warmup_ms: f64,
    /// Entire GpuEngine::load, including loading and warmup, excluding frontend/HTTP startup.
    pub startup_ms: f64,
    pub complete: bool,
}

impl Default for WarmupReport {
    fn default() -> Self {
        Self { policy: POLICY, plan: plan(), wall_limit_ms: WALL_LIMIT.as_millis() as u64,
            calls: vec![], standard_calls: 0, coverage_calls: 0, validation_calls: 0,
            validation_new_installations: 0, jit_before: 0, jit_after: 0, warmup_ms: 0.,
            startup_ms: 0., complete: false }
    }
}

/// Between-call deadline only: the owned outer startup deadline is responsible for a hung CUDA call.
/// Injecting the synchronized step/counter/monotonic clock also permits real CPU state-machine tests.
pub(crate) fn execute(report: &mut WarmupReport,
    mut step: impl FnMut(&DecisionBatch) -> Result<()>,
    mut count: impl FnMut() -> u64,
    mut elapsed: impl FnMut() -> Duration,
) -> Result<()> {
    report.jit_before = count();
    report.jit_after = report.jit_before;
    for (sequence, layout) in plan().into_iter().enumerate() {
        ensure!(elapsed() < WALL_LIMIT, "bounded exact warmup exceeded 90s before call {sequence}");
        let batch = layout.batch()?;
        let before = count();
        let begin_ms = elapsed().as_secs_f64() * 1000.;
        eprintln!("SYSONE_WARMUP_CALL_BEGIN {}", serde_json::json!({"sequence":sequence,
            "layout":layout,"begin_ms":begin_ms,"jit_before":before}));
        let result = step(&batch);
        let end = elapsed();
        let after = count();
        let call = WarmupCall { sequence, layout: layout.clone(), begin_ms,
            end_ms: end.as_secs_f64() * 1000., jit_before: before, jit_after: after, ok: result.is_ok() };
        eprintln!("SYSONE_WARMUP_CALL_END {}", serde_json::to_string(&call)?);
        report.calls.push(call);
        report.jit_after = after;
        report.warmup_ms = end.as_secs_f64() * 1000.;
        match layout.phase {
            "standard" => report.standard_calls += 1,
            "coverage" => report.coverage_calls += 1,
            "validation" => report.validation_calls += 1,
            _ => unreachable!(),
        }
        result.with_context(|| format!("bounded exact warmup call {sequence} ({})", layout.phase))?;
        let delta = after.checked_sub(before).context("cuTile L1 counter moved backwards")?;
        if layout.phase == "validation" {
            report.validation_new_installations += delta;
            ensure!(delta == 0, "bounded exact warmup validation call {sequence} installed {delta} new cuTile L1 kernels");
        }
        ensure!(end < WALL_LIMIT, "bounded exact warmup exceeded 90s after call {sequence}");
    }
    report.complete = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn fixed_51_call_plan_is_legal_and_identical_between_passes() {
        let p = plan();
        assert_eq!(p.len(), 51);
        let b = p[0].batch().unwrap();
        assert_eq!(b.token_ids, (1000..1100).collect::<Vec<_>>());
        assert_eq!((b.segments[0].len, b.segments[1].len), (60, 40));
        for (i, layout) in p.iter().enumerate() {
            let b = layout.batch().unwrap();
            assert!(b.answer_slots.iter().all(|&n| n == 2));
            if i > 0 {
                assert_eq!(layout.tokens, TOKENS[((i - 1) % 25) / 5]);
                assert_eq!(layout.answers, ANSWERS[(i - 1) % 5]);
                assert!(b.segments[1..].iter().all(|s| s.len == 1 && s.parent == 0));
                assert_eq!(b.segments[0].len as usize, layout.tokens - layout.answers);
            }
        }
        for i in 1..26 { assert_eq!(p[i].batch().unwrap(), p[i + 25].batch().unwrap()); }
    }

    #[test]
    fn exactly_51_calls_no_adaptive_extra_steps() {
        let calls = Cell::new(0);
        let jit = Cell::new(0);
        let mut r = WarmupReport::default();
        execute(&mut r, |_| { calls.set(calls.get()+1); if calls.get() <= 26 { jit.set(jit.get()+1); } Ok(()) },
            || jit.get(), || Duration::from_millis(calls.get())).unwrap();
        assert!(r.complete);
        assert_eq!((r.standard_calls, r.coverage_calls, r.validation_calls), (1,25,25));
        assert_eq!((r.calls.len(), r.validation_new_installations, r.jit_after), (51,0,26));
    }

    #[test]
    fn validation_miss_stops_without_retry_or_readiness() {
        let calls = Cell::new(0);
        let mut r = WarmupReport::default();
        let err = execute(&mut r, |_| { calls.set(calls.get()+1); Ok(()) }, || calls.get(), || Duration::ZERO).unwrap_err();
        assert!(err.to_string().contains("installed 1"));
        assert_eq!(calls.get(), 27);
        assert!(!r.complete);
        assert_eq!(r.validation_new_installations, 1);
    }

    #[test]
    fn wall_bound_is_checked_after_and_before_steps_and_hangs_need_outer_guard() {
        let calls = Cell::new(0);
        let mut r = WarmupReport::default();
        assert!(execute(&mut r, |_| { calls.set(1); Ok(()) }, || 0,
            || if calls.get() == 0 { Duration::ZERO } else { WALL_LIMIT }).is_err());
        assert_eq!(r.calls.len(), 1);
        assert!(!r.complete);
        let mut r = WarmupReport::default();
        assert!(execute(&mut r, |_| panic!("deadline must stop before step"), || 0, || WALL_LIMIT).is_err());
        assert!(r.calls.is_empty());
    }

    #[test]
    fn execution_error_preserves_call_and_never_continues() {
        let mut r = WarmupReport::default();
        assert!(execute(&mut r, |_| anyhow::bail!("fake CUDA error"), || 0, || Duration::ZERO).is_err());
        assert_eq!(r.calls.len(), 1);
        assert!(!r.calls[0].ok);
        assert!(!r.complete);
    }
}
