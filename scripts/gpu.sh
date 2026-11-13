#!/usr/bin/env bash
# Run a command that uses the GPU while holding the shared GPU lock, so concurrent jobs (benchmarks,
# reference/golden runs, GPU tests) never overlap. Every GPU job goes through this, not only timing runs:
# in Phase 1 unlocked reference jobs kept the GPU busy and invalidated the benchmarks.
#   scripts/gpu.sh cargo test --release --test kernels_gdn
#   scripts/gpu.sh ./target/release/bench_gdn          (benchmarks also refuse to start on a shared GPU)
set -euo pipefail
export SYSONE_GPU_LOCKED=1
exec flock "${SYSONE_GPU_LOCK:-/tmp/sysone-gpu.lock}" "$@"
