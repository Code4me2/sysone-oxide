# Exact eager execution and bounded warmup

Opt-in BF16/Tile execution; not a new default or deployment recommendation. For a first server, use
[Getting started](../GETTING_STARTED.md); for the completed post-fix optimized-qev serving comparison, use
[Performance](../PERFORMANCE.md). Request semantics remain those in [architecture](../ARCHITECTURE.md) and
[design](../DESIGN.md).

## Modes and configuration

| BF16 server choice | Execution |
|---|---|
| Neither flag (default) | Bucket-sized CUDA graphs |
| `--no-cuda-graphs` | Bucket-sized eager |
| `--exact-token-eager --gdn-backend tile` | Logical token/answer rows, eager only |

Exact mode keeps capacity-backed arenas and fixed-capacity tables, but never captures or replays graphs.
It requires BF16/Tile, rejects NVFP4 (including auto-detected checkpoints), and cannot be combined with
`--no-cuda-graphs`. Add `--bounded-exact-warmup` to opt into the policy below; otherwise standard warmup remains.
Both new flags default to false. The server's engine default remains `mock`, not BF16.

Library callers select `ExecutionMode::ExactEager` with `cuda_graphs = false`. Bounded warmup additionally needs
`warmup = true`, `bounded_exact_warmup = true`, explicit Tile, no debug taps/permutation, and capacity for the
whole plan (at least Q16/S17/P1). Without bounded warmup, an automatic backend is allowed only if architecture
resolution selects Tile. `StepShape` separates logical dimensions from capacity; `StepInfo` reports both.
Engine metrics report the execution, backend, GEMM, trimming and warmup choices;
`GpuEngine::warmup_report()` exposes startup results. The targeted check uses explicit 16,384-token limits
for rendered prefix plus question/options, not document body alone; these are not changed CLI defaults.

Pinned BF16 GEMMs and last-layer trimming remain on by default. Exact mode uses logical T for all-token work
and logical Q for trimmed queries, attention, MLP and readout; it does not disable trimming or switch GEMM
algorithms. Unsupported per-T descriptors fail without heuristic fallback. Default answer capacity is
`min(ceil(T / 8), max_answers)`, so T257/Q64 needs a T512/Q64 bucket. BF16 storage, FP32 accumulation/scoring,
attention/GDN semantics, admission, packing and answer ordering are unchanged; this is not quantization or generation.

## Warmup policy

The fixed plan is **51 calls**: one standard T100 call (60-token prefix, 40-token suffix), then 25 coverage and
25 identical validation calls. The lattice is T={65,66,68,72,80} × Q={1,2,4,8,16}, T-major, with one prefix,
one-token suffixes and two readout slots. Default answer density needs at least a T128/Q16 bucket.

Validation must install zero new cuTile kernels. Execution errors, validation misses and deadline violations
fail startup without retries or adaptive shrinking. The 90-second bound is checked **between calls**;
a hung device call needs an external startup deadline. This lattice does not cover every future shape/device.

## Targeted GPU validation and limits

The qualified production/test source passed `tests/engine_exact_gpu.rs` on **one RTX PRO 6000 Blackwell Max-Q**,
with pinned BF16 GEMMs, Tile and trimming. Exact eager ran first in a fresh process, followed by bucket eager
and bucket graph: 18 cases per mode, 72 run records including graph capture/replay, and 60 comparisons.
Cases cover T63/64/65/255/256/257/16384, Q1/2/16/64, interleaved restoration and independently remapped AB/BA
requests. Observed logit/probability max and p99 differences and confident-choice flips were all zero;
restorations were bit-identical. This is not a universal cross-mode bit-identity guarantee.

That targeted GPU run preceded the subsequent upload-error ownership fix. At that stage, the fix passed an
all-target build and 28 focused CPU tests, including 10 fault-injection cases, without rerunning the targeted
state-isolation GPU test. Later post-fix CUDA/model execution is covered by the completed HTTP serving comparison
below; it is not a rerun of this synthetic test. Failed uploads poison the plan; if completion is uncertain, one
lease retains its transfer resources for process lifetime rather than freeing or reusing them. Successful uploads
release that lease after synchronization.

Exact startup completed 1+25+25 calls with zero validation or checked post-ready installations and no captures.
The unchanged gates are logit max/p99 0.2842/0.1201, probability max/p99 0.04241/0.01846, confident-choice margin
0.02. This is synthetic correctness/state-isolation/JIT coverage, not throughput, HTTP capacity, held-out human
quality or another-device qualification. The test's default CPU case checks planning only; its ignored GPU case
needs the public model, a fresh process and a real held GPU lock. `engine_warmup_v2` is a historical restricted
startup test, not this qualification. `bench_exact` and `bench_warmup_v2` also provide planning/profile tools;
GPU work requires separate authorization.

Historical pre-rebase throughput observations used heuristic GEMMs and an untrimmed baseline. Moving JIT before
readiness improved an initial burst, not steady-state throughput; those rates do not measure this rebased tree.
Detailed experiments/provenance remain in PR history and private evidence. Those historical observations must
not be substituted for the completed post-fix comparison below.

## Completed post-fix HTTP serving comparison

The later accepted comparison at **fbddc43** exercised the public BF16 model after the ownership fix on one
NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120), with Tile, pinned GEMMs and trimming.
Against the fastest verified optimized qev backend in the completed tests (real FLA GDN and successful CUDA
Triton launches), exact eager with bounded warmup delivered **12–68% higher decisions/s in 8/8 cells**.
Bucketed graphs were mixed: **0.65–1.58× qev, four wins and four losses**. This supersedes the old statement that
there was no rebased serving throughput, not the narrower pre-fix state-isolation test's provenance.

See [Performance](../PERFORMANCE.md) and its linked curated summary for all eight rates, matched settings,
two-start finite-burst method and tolerance agreement (not bit identity). This is not sustained capacity,
held-out quality, full error-conformance or another-device qualification, nor a new test run for these docs.
No globally green Rust suite is claimed: `sched::pool::tests::new_work_is_bounded_and_finishing_work_is_not`
remains **FAILED / NOT RERUN**. No default promotion, deployment recommendation or additional GPU work is implied.
