//! Private upload ownership state, shared by the CUDA path and CPU fault tests.
//! Arm before submission; only a successful drain releases owners. Submission/drain failure poisons reuse.
//! Tests use operation traces and five allocation/stream DropSpies: success/reuse, each copy failure,
//! drain/combined errors, retry rejection, constructor failure and submission/drain unwinding.

use anyhow::{Result, ensure};
use std::fmt;

pub(super) struct UploadState<L> {
    poisoned: bool,
    lease: Option<L>,
}

impl<L> Default for UploadState<L> {
    fn default() -> Self {
        Self { poisoned: false, lease: None }
    }
}

/// Keep both actual errors, including their chains, when submission and its drain fail.
#[derive(Debug)]
pub(super) struct SubmissionAndDrainError {
    pub submission: anyhow::Error,
    pub drain: anyhow::Error,
}

impl fmt::Display for SubmissionAndDrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GDN table submission failed: {:#}; completion drain also failed: {:#}",
            self.submission, self.drain)
    }
}

impl std::error::Error for SubmissionAndDrainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.submission.as_ref())
    }
}

impl<L> UploadState<L> {
    pub fn ensure_ready(&self) -> Result<()> {
        ensure!(!self.poisoned && self.lease.is_none(), "GDN table upload is poisoned; replace the plan");
        Ok(())
    }

    /// `prepare` may mutate staging only while Ready, and must return ALL transfer owners.
    /// Copy/drain callbacks must use that lease; `Ok` from drain must establish completion.
    pub fn upload(
        &mut self,
        prepare: impl FnOnce() -> Result<L>,
        first: impl FnOnce(&L) -> Result<()>,
        second: impl FnOnce(&L) -> Result<()>,
        drain: impl FnOnce(&L) -> Result<()>,
    ) -> Result<()> {
        self.ensure_ready()?;
        let lease = prepare()?;
        self.lease = Some(lease);
        self.poisoned = true; // Armed BEFORE any possibly submitted copy, including unwinding.
        let lease = self.lease.as_ref().unwrap();
        let submission = first(lease).and_then(|()| second(lease));
        // Always drain on submission Err as well; an Err from drain is NOT completion.
        let completion = drain(lease);
        if completion.is_ok() {
            drop(self.lease.take());
        }
        match (submission, completion) {
            (Ok(()), Ok(())) => {
                self.poisoned = false;
                Ok(())
            }
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(submission), Err(drain)) => Err(SubmissionAndDrainError { submission, drain }.into()),
        }
    }
}

impl<L> Drop for UploadState<L> {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            // Completion is unproved (driver error or unwind). Quarantine actual owners, not
            // pointers/borrows: freeing either host OR destination, stream/context is unsafe.
            // Exceptional, bounded to one lease per failed plan; no normal-path leak or retry.
            std::mem::forget(lease);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, panic::{AssertUnwindSafe, catch_unwind}, rc::Rc, sync::{Arc, Weak}};

    type Trace = Rc<RefCell<Vec<&'static str>>>;
    struct DropSpy { name: &'static str, drops: Trace }
    impl Drop for DropSpy {
        fn drop(&mut self) { self.drops.borrow_mut().push(self.name); }
    }
    struct Host { _spy: Arc<DropSpy>, value: usize }
    struct Lease {
        hosts: [Arc<Host>; 2],
        dests: [Arc<DropSpy>; 2],
        stream: Arc<DropSpy>,
    }
    struct Plan {
        hosts: [Arc<Host>; 2],
        dests: [Arc<DropSpy>; 2],
        stream: Arc<DropSpy>,
        trace: Trace,
        state: UploadState<Lease>, // Like GdnPlan: ordinary owners drop before the lease state.
    }
    struct Probe {
        hosts: [Weak<DropSpy>; 2],
        dests: [Weak<DropSpy>; 2],
        stream: Weak<DropSpy>,
        drops: Trace,
        trace: Trace,
    }
    #[derive(Clone, Copy)]
    enum Fault { None, First, Second, Sync, FirstAndSync, SecondAndSync, PanicAfterFirst, PanicDrain }

    impl Plan {
        fn new() -> (Self, Probe) {
            let drops = Rc::new(RefCell::new(vec![]));
            let trace = Rc::new(RefCell::new(vec![]));
            let spy = |name| DropSpy { name, drops: drops.clone() };
            let hosts = [Arc::new(Host { _spy: Arc::new(spy("host0")), value: 0 }),
                Arc::new(Host { _spy: Arc::new(spy("host1")), value: 0 })];
            let dests = [Arc::new(spy("dest0")), Arc::new(spy("dest1"))];
            let stream = Arc::new(spy("stream"));
            // Weak probes target the nested spy, not Host: Arc::get_mut also rejects weak Host aliases.
            let probe = Probe { hosts: hosts.each_ref().map(|h| Arc::downgrade(&h._spy)),
                dests: dests.each_ref().map(Arc::downgrade), stream: Arc::downgrade(&stream),
                drops, trace: trace.clone() };
            (Self { state: UploadState::default(), hosts, dests, stream, trace }, probe)
        }

        fn upload(&mut self, fault: Fault) -> Result<()> {
            // Same production helper: readiness gates staging, and callbacks see owned resources.
            self.state.upload(
                || {
                    self.trace.borrow_mut().push("stage");
                    for host in &mut self.hosts {
                        Arc::get_mut(host).expect("staging must be unique").value += 1;
                    }
                    Ok(Lease { hosts: self.hosts.clone(), dests: self.dests.clone(), stream: self.stream.clone() })
                },
                |lease| {
                    self.trace.borrow_mut().push("first");
                    assert_eq!(lease.hosts[0].value, lease.hosts[1].value);
                    assert_eq!(lease.dests[0].name, "dest0");
                    assert_eq!(lease.stream.name, "stream");
                    if matches!(fault, Fault::First | Fault::FirstAndSync) { anyhow::bail!("first copy"); }
                    Ok(())
                },
                |lease| {
                    self.trace.borrow_mut().push("second");
                    assert_eq!(lease.dests[1].name, "dest1");
                    if matches!(fault, Fault::PanicAfterFirst) { panic!("after first queued"); }
                    if matches!(fault, Fault::Second | Fault::SecondAndSync | Fault::PanicDrain) { anyhow::bail!("second copy"); }
                    Ok(())
                },
                |_| {
                    self.trace.borrow_mut().push("drain");
                    if matches!(fault, Fault::PanicDrain) { panic!("drain unwind"); }
                    if matches!(fault, Fault::Sync | Fault::FirstAndSync | Fault::SecondAndSync) { anyhow::bail!("drain error"); }
                    Ok(())
                },
            )
        }
    }

    impl Probe {
        fn all_retained(&self) {
            assert!(self.hosts.iter().all(|p| p.upgrade().is_some()));
            assert!(self.dests.iter().all(|p| p.upgrade().is_some()));
            assert!(self.stream.upgrade().is_some());
            assert!(self.drops.borrow().is_empty());
        }
        fn all_released(&self) {
            assert!(self.hosts.iter().all(|p| p.upgrade().is_none()));
            assert!(self.dests.iter().all(|p| p.upgrade().is_none()));
            assert!(self.stream.upgrade().is_none());
            let mut drops = self.drops.borrow().clone(); drops.sort();
            assert_eq!(drops, ["dest0", "dest1", "host0", "host1", "stream"]);
        }
    }

    // Two copies then one drain, lease released each time; unchanged source owners can be reused.
    #[test]
    fn success_order_reuse_and_normal_release() {
        let (mut plan, probe) = Plan::new();
        let host_addresses = plan.hosts.each_ref().map(Arc::as_ptr);
        let dest_addresses = plan.dests.each_ref().map(Arc::as_ptr);
        for _ in 0..2 {
            plan.upload(Fault::None).unwrap();
            assert_eq!(plan.hosts.each_ref().map(Arc::as_ptr), host_addresses);
            assert_eq!(plan.dests.each_ref().map(Arc::as_ptr), dest_addresses);
            plan.state.ensure_ready().unwrap();
            assert!(plan.state.lease.is_none());
            assert!(plan.hosts.iter_mut().all(|p| Arc::get_mut(p).is_some()));
        }
        assert_eq!(*probe.trace.borrow(), ["stage", "first", "second", "drain", "stage", "first", "second", "drain"]);
        assert_eq!(plan.hosts[0].value, 2);
        drop(plan); probe.all_released();
    }

    // Even the first enqueue error drains; confirmed completion releases the lease, not poison.
    #[test]
    fn first_copy_error_drains_and_releases() {
        let (mut plan, probe) = Plan::new();
        assert_eq!(plan.upload(Fault::First).unwrap_err().to_string(), "first copy");
        assert_eq!(*probe.trace.borrow(), ["stage", "first", "drain"]);
        assert!(plan.state.ensure_ready().is_err());
        assert!(plan.state.lease.is_none());
        drop(plan); probe.all_released();
    }

    // First is queued before second fails; positive drain allows normal destruction, never reuse.
    #[test]
    fn second_copy_error_after_first_queued_drains() {
        let (mut plan, probe) = Plan::new();
        assert_eq!(plan.upload(Fault::Second).unwrap_err().to_string(), "second copy");
        assert_eq!(*probe.trace.borrow(), ["stage", "first", "second", "drain"]);
        assert!(plan.state.ensure_ready().is_err());
        drop(plan); probe.all_released();
    }

    // Sync failure is not completion: every host/destination/stream owner survives plan destruction.
    #[test]
    fn synchronization_error_quarantines_all_owners() {
        let (mut plan, probe) = Plan::new();
        assert_eq!(plan.upload(Fault::Sync).unwrap_err().to_string(), "drain error");
        assert!(plan.state.lease.is_some());
        drop(plan); probe.all_retained();
    }

    // Both first- and second-enqueue errors retain their original error plus the actual drain error.
    #[test]
    fn combined_submission_and_drain_errors_are_both_retained() {
        for (fault, primary) in [(Fault::FirstAndSync, "first copy"), (Fault::SecondAndSync, "second copy")] {
            let (mut plan, probe) = Plan::new();
            let error = plan.upload(fault).unwrap_err();
            let combined = error.downcast_ref::<SubmissionAndDrainError>().unwrap();
            assert_eq!(combined.submission.to_string(), primary);
            assert_eq!(combined.drain.to_string(), "drain error");
            assert!(error.to_string().contains(primary) && error.to_string().contains("drain error"));
            drop(plan); probe.all_retained();
        }
    }

    // Poison rejects before prepare/any callback, for both confirmed and uncertain completion.
    #[test]
    fn retry_is_rejected_before_host_mutation_or_callbacks() {
        for fault in [Fault::First, Fault::Second, Fault::Sync, Fault::SecondAndSync] {
            let (mut plan, probe) = Plan::new();
            assert!(plan.upload(fault).is_err());
            let before = probe.trace.borrow().clone();
            let values = plan.hosts.each_ref().map(|h| h.value);
            assert!(plan.upload(Fault::None).unwrap_err().to_string().contains("poisoned"));
            assert_eq!(*probe.trace.borrow(), before);
            assert_eq!(plan.hosts.each_ref().map(|h| h.value), values);
        }
    }

    // Constructor `?` destroys a partial plan: completed drains free normally, uncertain ones retain ALL.
    #[test]
    fn constructor_failure_drop_obeys_completion() {
        fn construct(mut plan: Plan, fault: Fault) -> Result<Plan> { plan.upload(fault)?; Ok(plan) }
        for fault in [Fault::Second, Fault::SecondAndSync] {
            let (plan, probe) = Plan::new();
            assert!(construct(plan, fault).is_err());
            if matches!(fault, Fault::Second) { probe.all_released(); } else { probe.all_retained(); }
        }
    }

    // Catching an unwind leaves a live plan poisoned/leased; Drop must also be safe during unwinding.
    #[test]
    fn panic_after_first_submission_poison_and_drop() {
        let (mut plan, probe) = Plan::new();
        assert!(catch_unwind(AssertUnwindSafe(|| plan.upload(Fault::PanicAfterFirst))).is_err());
        let before = probe.trace.borrow().clone();
        assert_eq!(before, ["stage", "first", "second"]);
        assert!(plan.upload(Fault::None).is_err());
        assert_eq!(*probe.trace.borrow(), before);
        drop(plan); probe.all_retained();

        let (mut plan, probe) = Plan::new();
        assert!(catch_unwind(AssertUnwindSafe(move || plan.upload(Fault::PanicAfterFirst))).is_err());
        probe.all_retained();
    }

    // A panic inside the error-path drain cannot drop owners or make staging reusable.
    #[test]
    fn drain_unwind_retains_all_owners() {
        let (mut plan, probe) = Plan::new();
        assert!(catch_unwind(AssertUnwindSafe(|| plan.upload(Fault::PanicDrain))).is_err());
        assert_eq!(*probe.trace.borrow(), ["stage", "first", "second", "drain"]);
        assert!(plan.upload(Fault::None).is_err());
        drop(plan); probe.all_retained();
    }

    // A preflight error runs no copy/drain and leaves the unarmed state reusable.
    #[test]
    fn preparation_error_does_not_arm_or_submit() {
        let mut state = UploadState::<()>::default();
        assert!(state.upload(|| anyhow::bail!("preflight"), |_| panic!("first"),
            |_| panic!("second"), |_| panic!("drain")).is_err());
        state.ensure_ready().unwrap();
    }
}
