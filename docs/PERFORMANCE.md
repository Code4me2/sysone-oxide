# Performance: completed optimized-qev serving comparison

**Sysone's opt-in exact eager mode delivered 12–68% higher decision throughput than optimized qev in all eight
tested cells (up to 1.68×).** This is a completed, tolerance-checked HTTP serving comparison on one GPU, not a
new run, a global qev tuning optimum or a change of defaults.

**Default bucketed BF16 graphs were mixed: 0.65–1.58× qev, four wins and four losses.** The HTTP server defaults
to **mock**, not BF16; neither default should be read as the exact-mode headline.

Numerical source: [curated accepted-comparison summary](../benchmarks/results/x86/qev-serving-optimized/summary.json).
It preserves full-precision arithmetic means and both observations per arm. The
[retained published report](https://github.com/Code4me2/sysone-oxide/pull/10#issuecomment-6996569961) supplies the
comparison's audit context. This JSON is a sanitized summary, **not raw logs**. Full private raw evidence is not
in this repository, and the original comparison launcher was not committed as a verbatim reproduction script.

## What was compared

- **One GPU:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, sm_120.
- **Same public BF16 model:** `baselabs/elephant-1` at revision
  `b3f8a93462fb1051a9de031e94a7ff380e254072`.
- **Sysone:** `fbddc43f3c52b6671abdbd08a099e354bbcc4753`, BF16 engine, Tile GDN, pinned GEMMs and last-layer
  trimming; bucketed CUDA graphs versus exact-token eager with bounded exact warmup.
- **qev:** OneJev `81ce62f1597c91e46767d4d02d6ac2e18534fe94` plus the **unchanged model-card `serve.patch`**.
  The fastest verified qev backend in the **completed** tests used real **FLA GDN with actual successful linked
  CUDA Triton launches**, not fallback Torch timings. The author's auto-probe selected batched forks within
  each request. Backend verification/profiling was completed and profiling disabled before timing.
- **Matched request bodies and limits:** 1,000/4,000-word states, 1/8 questions per request, concurrency 1/16;
  both servers used branch and whole-request limits of 16,384 rendered tokens, with no truncation. These include
  the state, template, questions/options and any debias branches.
- **Sysone-specific capacity and scheduling:** max step and token budget 16,384, max prefixes 32, CPU workers 4,
  max wait 2,000 µs. These are sysone controls, not shared qev settings.

Stock qev's HTTP handler **serializes cross-request inference** while batching forks **within** each request.
That is the measured serving behavior, not an invented cross-request batcher or a deliberately crippled baseline.
Sysone can pack several requests into a step. These results do not rule out faster qev configurations elsewhere.

## All eight matched cells

Absolute rates are **decisions per second**, rounded to three decimals. A decision is one question answered:
Q8 means eight decisions per request/document. State length is **words, not tokens**. Logical input-token
throughput, if reported by a load generator, is not autoregressive generation speed.

| State words | Questions | Concurrency | qev decisions/s | Exact decisions/s | Bucket decisions/s | Exact / qev | Bucket / qev |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1 | 1 | 7.671 | 8.949 | 5.991 | 1.167× | 0.781× |
| 1000 | 1 | 16 | 7.677 | 11.974 | 9.324 | 1.560× | 1.214× |
| 1000 | 8 | 1 | 39.009 | 51.785 | 45.327 | 1.328× | 1.162× |
| 1000 | 8 | 16 | 39.121 | 65.607 | 61.622 | 1.677× | 1.575× |
| 4000 | 1 | 1 | 2.605 | 2.906 | 1.704 | 1.116× | 0.654× |
| 4000 | 1 | 16 | 2.598 | 3.065 | 2.321 | 1.180× | 0.893× |
| 4000 | 8 | 1 | 14.734 | 20.333 | 13.190 | 1.380× | 0.895× |
| 4000 | 8 | 16 | 14.622 | 21.729 | 18.662 | 1.486× | 1.276× |

Each rate is the **arithmetic mean of two fresh-process observations**. Each ratio divides those two means;
it is not the mean of paired ratios, a median, the best start or a pooled rate. Percent headlines round the
full-precision ratios, not the displayed rounded rates. Exact wins 8/8; bucket wins 4/8 and loses 4/8.

## Measurement scope

Six fresh processes ran in this order: **qev → bucket → exact → exact → bucket → qev**.
Each process measured 16 requests per cell after **two excluded warmup requests** per cell.
The timing denominator spans submission through final completion/drain, without retries or dropped requests.

Across 48 process/cell combinations there were **1,146 POSTs = 768 timed + 96 warmup + 282 correctness**.
Correctness repeats used **47 independent development contexts**; repetitions are not additional independent
quality examples.

These are **small finite bursts with two starts per arm**. They do not establish statistical significance,
sustained capacity, a latency SLA, a global configuration optimum or hardware peak performance. Do not extrapolate
them to other GPUs, state lengths, question counts, precision formats or bulk throughput.
Reference startup included verification, profiling and source-check overhead; it is not an intrinsic startup-speed
comparison. No startup-speed headline is made here.

## Correctness and precision

Agreement is **within unchanged tolerances**, not bit-identical output or human-quality certification.
Across the accepted pairings:

| Absolute difference | Observed max | Observed p99 | Unchanged max gate | Unchanged p99 gate |
|---|---:|---:|---:|---:|
| Probability | 0.027066 | 0.009440 | 0.04241 | 0.01846 |
| Slot logit | 0.1337204 | 0.0778408 | 0.2842 | 0.1201 |

There was one permitted near-tie choice change per pairing: live reference margin **0.000691**, below the
unchanged **0.02** margin gate. This supports tolerance agreement on those development contexts, not identical
choices in every possible case. Five historical error cases were excluded: this is not full error-conformance,
held-out/human quality or multimodal qualification.

The path retains **BF16 storage/backbone with FP32 accumulation/scoring** (CPU answer assembly uses f64).
This is not a global “no TF32” claim: native Tile GDN `Fp32Split3` implements products using **three TF32 products
with FP32 accumulation**. Torch/cuBLAS TF32 flags being off does not disable that native kernel math.
See [the architecture precision table](ARCHITECTURE.md#5-gpu-ops-of-one-forward) and
[`src/kernels/gdn/tile.rs`](../src/kernels/gdn/tile.rs).

Separate synthetic exact-mode state-isolation checks are described in [the x86 guide](x86/README.md).
The known scheduler case `sched::pool::tests::new_work_is_bounded_and_finishing_work_is_not` remains
**FAILED / NOT RERUN**; this comparison does not establish a globally green Rust suite.

## Sysone configuration

After [setup](GETTING_STARTED.md#setup-and-build), from the repository root, this selects the measured exact
arm's settings (choose an authorized GPU). These are **explicit choices**, not new defaults or a deployment
recommendation:

```bash
MODEL_DIR=../models/elephant-1
scripts/gpu.sh --gpu 0 ./target/release/sysone \
  --model "$MODEL_DIR" --name elephant-1 --listen 127.0.0.1:8000 \
  --engine bf16 --max-branch-tokens 16384 --max-request-tokens 16384 \
  --max-step-tokens 16384 --token-budget 16384 --max-prefixes 32 \
  --cpu-workers 4 --gdn-backend tile --gemm pinned \
  --exact-token-eager --bounded-exact-warmup
```

For the bucket arm, stop that server and rerun with **only** `--exact-token-eager --bounded-exact-warmup` removed.
Pinned GEMMs and last-layer trimming stay enabled. Do not add `--no-cuda-graphs` for that arm: it would select
bucket eager instead of bucket graphs. `--no-cuda-graphs` is also mutually exclusive with exact eager.
Exact is BF16/Tile only, not NVFP4. [Warmup and residual risks](x86/README.md) remain opt-in qualifications.
The server has no built-in authentication/TLS; keep the loopback binding unless you provide protected remote access.

## Evaluate your own workload

1. Freeze the public model/source revisions, question definitions, rendered-token limits, debias, hardware,
   execution flags and client concurrency. Count **questions** for decisions/s; also keep document/request rates.
2. Check response correctness before timing, including near-ties and errors. `/health` alone does not distinguish
   mock from real inference and is not a numerical test. Keep existing gates unchanged.
3. On an authorized idle GPU under its lock, compare bucket and exact in separate fresh server processes. Define
   warmups and a submission-to-final-drain timing interval up front; keep all status counts and report failures.
4. Use enough independent representative inputs and repeated starts to assess variability. For sustained capacity,
   run a longer workload separately rather than promoting the finite-burst headline to a steady-state claim.

For example, the existing **CPU HTTP client** can send a JSONL job corpus once against the locked server above.
Run this from another terminal; it is not another GPU engine and must not try to take the running server's lock:

```bash
./target/release/loadgen --addr 127.0.0.1:8000 --concurrency 16 \
  --corpus jobs.jsonl --responses responses.jsonl
```

`jobs.jsonl` uses the [bulk job format](BULK.md#usage). This is a workload measurement entry point, **not the
original accepted-comparison launcher**: corpus mode sends each job once and does not automatically exclude
warmups. Arrange separate warmup requests/process repetitions and correctness review for your experiment.
If evaluating qev yourself, independently verify its actual FLA/Triton execution and selected fork mode, match
requests/limits and disclose cross-request scheduling; no qev reproduction script is supplied here.
Keep raw logs local and sanitize any results you choose to share.
