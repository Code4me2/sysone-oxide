# Opt-in exact eager execution and bounded warmup

This is a scoped BF16/Tile implementation, not a new default or deployment recommendation.
See [architecture](../ARCHITECTURE.md) and [design](../DESIGN.md) for the unchanged request semantics.

## Rebase provenance (not GPU requalification)

The observations below are historical, from the pre-rebase heuristic-GEMM, untrimmed baseline. They are **not
measurements or numerical requalification of this rebased tree**. The rebase inherits upstream L1/N1/C1/S2:
BF16 GEMMs remain pinned by default (one algorithm per weight shape, checked per T), last-layer trimming remains
on, and answer capacity is `min(ceil(T / 8), max_answers)` by default. Exact mode applies logical T to all-token
ops and logical Q to the trimmed queries, attention, MLP and readout; it does not disable trimming or change the
pinned algorithm to accommodate dynamic rows. Unsupported per-T descriptors fail, without heuristic fallback.

Capacity planning uses the same answer-density options as the engine. Thus T257/Q64 selects capacity T512/Q64,
and the fixed warmup lattice needs at least T128/Q16 with default density (T100 has only Q13). Coverage itself
is unchanged: 51 calls, no adaptive shrinking. Benchmark metadata records the inherited choices; profile GEMM
rows come from current graph specs, not a historical assumption that only the readout uses Q. CPU checks cannot
qualify these GPU interactions. A standalone production full-model boundary/isolation numerical regression suite
remains a gap; fresh GPU fidelity, graph/default-path and startup qualification are required before promotion.
No new performance claim is made here.

## Configuration and API

The historical qualification used these server flags (plus the model argument):

```text
--engine bf16 --gdn-backend tile --exact-token-eager --bounded-exact-warmup
--max-branch-tokens 16384 --max-request-tokens 16384 --max-step-tokens 16384
--token-budget 16384 --max-prefixes 32 --cpu-workers 4
```

The 16,384-token limit includes the **rendered prefix and question/options**, not just the document body.
These sizes are explicit qualification settings, not changed CLI defaults. FIFO admission, packing, submission
and answer ordering, four CPU workers, full bidirectional prefix/own-suffix attention, and causal GDN semantics
are preserved. BF16 storage, FP32 GEMM accumulation/scoring, and the prohibition on reduced-precision GEMM
reductions are unchanged; this is not quantization, generation, or tensor parallelism.

- `--exact-token-eager` requests actual logical token/answer rows over the existing capacity-backed arena.
  Allocation buckets and fixed-capacity tables remain; exact mode never captures or replays CUDA graphs.
  It requires BF16 and explicit `--gdn-backend tile`; combining it with `--no-cuda-graphs` is rejected.
- `--no-cuda-graphs` alone remains legacy **bucket-sized** eager execution, not exact execution.
- `--bounded-exact-warmup` additionally requires exact mode. Without it, the legacy standard warmup remains;
  the bounded flag defaults to false. Unflagged BF16 execution remains bucket-graph execution.
- The fixed policy performs 51 startup calls: the standard 100-token call, then 25 coverage and 25 identical
  validation calls. Coverage is T={65,66,68,72,80} × Q={1,2,4,8,16}, one prefix, two readout slots.
  Validation must install zero new cuTile kernels. Errors, validation misses, or deadline violations fail startup,
  without adaptive shapes or retries. The 90-second bound is checked **between calls**; a hung device call still
  requires an external startup deadline. Coverage is not a guarantee for every future shape or architecture.

Library callers use `EngineOptions::execution_mode = ExecutionMode::ExactEager`, `cuda_graphs = false`, and
optionally `bounded_exact_warmup = true` with `warmup = true` and explicit Tile. Unbounded library exact mode may keep
`gdn_backend = None`: it is checked after architecture resolution, before weight loading; explicit SIMT is
rejected on CPU. Both explicitly requested NVFP4 weights and an auto-detected NVFP4 checkpoint are rejected
before device/model allocation. Bounded validation requires the full
lattice/standard capacity, at least Q16/S17/P1, and no debug taps or hidden permutation. `StepShape` separates
logical dimensions from allocation capacity; `StepInfo` reports both. Engine metrics identify execution, graph,
backend, GEMM, trimming and warmup policies; `GpuEngine::warmup_report()` exposes the startup report.

`bench_exact` and `bench_warmup_v2` retain CPU planning and GPU qualification/profile subcommands. GPU work
requires separate authorization and the normal GPU lock discipline. Raw qualification inputs/results are not
included here. Portable unit tests cover CLI rejection/defaults, shape spans, boundary admission and the fixed
warmup state machine; GPU graph/startup tests are separate and are not established by a CPU build.

## Historical observations and limits (pre-rebase only)

For the first 1K-**word**, 8-question, concurrency-64 finite burst on RTX 5090, two-start means were
3.9315 → 8.0651 documents/s (about 2.05×). This moves JIT work before readiness, **not** a 2× steady-state gain:
readiness rose from 18.13 to 38.15 seconds, the five later uniform-cell means were flat/slightly lower
(0.04–0.18% regressions), and cold whole-job runtime was slightly longer.

The primary mixed workload was 512 documents, 64 each at 1K/4K/8K/12K body words × 8/24 questions,
8,192 decisions and 3,775,760 logical input tokens. Full drain at concurrency 16, including the tail:

| Separately observed single card | Documents/s | Logical input tokens/s |
|---|---:|---:|
| RTX 5090 | 1.511954 | 11,149.955 |
| RTX PRO 6000 Blackwell Max-Q, observation A | 1.538433 | 11,345.220 |
| RTX PRO 6000 Blackwell Max-Q, observation B | 1.615096 | 11,910.578 |

Both PRO cards had a documented 300 W power limit; they were measured serially, not as simultaneous replicas
or a combined fleet. Their observed gains over the matched RTX 5090 cell were about 1.8% and 6.8%.
For that RTX 5090 mix, concurrency 32/64 was slightly slower (1.502188/1.500645 documents/s), with more queueing.
These are individual observations, not ABBA significance, a statistical global optimum, or guaranteed per-length
capacity. Logical input TPS is **not generation speed**; words are not tokens.

Fixed numerical qualification gates passed within their stated scope, not bit-identically across every packing
layout. Permitted near-tie choice flips and direct/matrix/individual-operation regressions remain part of the
observations: the two PRO runs had respectively 27/95 and 21/95 direct cases, and 15/96 and 12/96 matrix windows,
slower than RTX 5090. Synthetic/development consistency is not held-out human quality. An earlier incomplete
warmup run failed its deadline; a separately authorized complete run superseded it without erasing that failure.
The known scheduler test `sched::pool::tests::new_work_is_bounded_and_finishing_work_is_not` remains
**FAILED / NOT RERUN**. Targeted qualification and CPU passes do not make the full Rust suite green.

The GPU observations above refer to the qualified candidate before packaging hardening: diagnostic-only
privacy edits, earlier option/backend rejection, create-only benchmark outputs, and fail-closed nonfinite-logit
bit diagnostics. The packaged source was
rebuilt and CPU-tested against the pinned dependency, without another GPU run. The committed toy-graph and
ignored startup tests are not a self-contained full-model exact-versus-legacy numerical regression suite;
production boundary/isolation comparisons came from the separate qualification harness.

No default promotion, persistent deployment, publication, or additional GPU work is implied.
