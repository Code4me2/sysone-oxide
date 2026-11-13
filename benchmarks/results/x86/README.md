# x86 results

Sanitised test and benchmark results from the x86 host (2x RTX PRO 6000 Blackwell, 1x RTX 5090; docs/x86/AGENT.md),
written by `scripts/x86/run_tests.sh` and `scripts/x86/run_bench.sh` and committed through PRs that passed
`scripts/x86/check_outgoing.sh`.

```
<card-slug>/<date>/gpu<N>-tests/        run_tests.sh: one .txt per test target (cargo test stdout)
<card-slug>/<date>/gpu<N>-single/       run_bench.sh --mode single: one .txt per benchmark, profile.json (hwprobe)
<card-slug>/<date>/gpu<N>-concurrent/   run_bench.sh --mode concurrent: the jobs run on all GPUs at once
host/<date>/cpu-tests/                  run_tests.sh: GPU-free tests
```

Every run directory has `summary.tsv` (job, exit code or `NOT-RUN`, seconds or the reason) and `metadata.json`
(GPU name, compute capability, memory, driver and CUDA versions, power limit, clocks, sysone git sha, cuda-rust
revision: nothing else about the machine). A second run on the same day gets a `-2` suffix. Numbers from `--quick`
runs are smoke tests only and are marked so in `metadata.json`.
