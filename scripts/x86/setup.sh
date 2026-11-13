#!/usr/bin/env bash
# x86 setup check (docs/x86/TASKS.md T0, first half): prerequisites, build, and the tests that need neither a GPU
# nor the checkpoint. Prints a summary; writes nothing under benchmarks/results (run_tests.sh does that).
#
#   scripts/x86/setup.sh
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$X86_ROOT"
status=0
step() { echo; echo "== $*"; }

step "doctor (core prerequisites)"
scripts/doctor.sh --core || { echo "x86 setup: fix the doctor's FIX lines first"; exit 1; }

step "GPUs"
for i in $(x86_gpu_indices); do
    cc=$(x86_gpu_field "$i" compute_cap)
    echo "GPU $i: $(x86_gpu_field "$i" name), $(x86_sm "$cc"), $(x86_gpu_field "$i" memory.total) MiB, slug $(x86_card_slug "$(x86_gpu_field "$i" name)")"
done

step "build (cargo build --release --all-targets)"
cargo build --release --all-targets 2>&1 | grep -E '^(warning: SIMT|error|warning: unused)|Finished' || true
[ "${PIPESTATUS[0]}" = 0 ] || { echo "x86 setup: build failed (rerun cargo build --release --all-targets for the full log)"; exit 1; }

step "GPU-free tests"
for t in "--lib" "--test render_golden_shape" "--test scripts"; do
    # shellcheck disable=SC2086
    if out=$(cargo test --release $t 2>&1); then echo "ok      cargo test --release $t: $(grep -E '^test result' <<< "$out" | tail -1)"
    else echo "FAILED  cargo test --release $t"; grep -E '^(test .*FAILED|failures:|---- )' <<< "$out" | head -20; status=1; fi
done

echo
[ "$status" = 0 ] && echo "x86 setup: OK. Next: scripts/x86/run_tests.sh, then scripts/x86/run_bench.sh (docs/x86/AGENT.md)" ||
    echo "x86 setup: FAILED (see above)"
exit "$status"
