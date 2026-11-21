# x86 results

This directory holds small sanitized x86 results. For the completed optimized-qev serving comparison, see
[the user-facing performance guide](../../../docs/PERFORMANCE.md) and
[`qev-serving-optimized/summary.json`](qev-serving-optimized/summary.json). That file is a curated summary of an
accepted comparison, **not standard runner raw output or a fresh run**. It retains both observations and
full-precision means; full private raw evidence and the original launcher are not included here.

## Standard runner layout

The x86 runners (`scripts/x86/run_tests.sh`, `run_bench.sh`, `run_bulk.sh`) write sanitized test/benchmark outputs
for review under the original [operating rules](../../../docs/x86/AGENT.md). The normal layout is:

```
<card-slug>/<date>/gpu<N>-tests/        run_tests.sh: one .txt per test target (cargo test stdout)
<card-slug>/<date>/gpu<N>-single/       run_bench.sh --mode single: one .txt per benchmark, profile.json (hwprobe)
<card-slug>/<date>/gpu<N>-concurrent/   run_bench.sh --mode concurrent: the jobs run on all GPUs at once
host/<date>/cpu-tests/                  run_tests.sh: GPU-free tests
```

Every standard test/benchmark run directory has `summary.tsv` (job, exit code or `NOT-RUN`, seconds or the reason)
and `metadata.json` (GPU name, compute capability, memory, driver and CUDA versions, power limit, clocks, sysone git sha, cuda-rust
revision: nothing else about the machine). A second run on the same day gets a `-2` suffix. Numbers from `--quick`
runs are smoke tests only and are marked so in `metadata.json`.
