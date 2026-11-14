# x86 tasks

The initial work list for the x86 agent ([AGENT.md](AGENT.md)). Each task becomes one GitHub issue labelled `x86`
on `Code4me2/sysone-oxide`, one branch `x86/<issue#>-<slug>` and one PR that says `Closes #<n>`. Do them in order:
each assumes the previous ones are merged. Throughout: every GPU job through `scripts/gpu.sh` (the runners do it),
`scripts/x86/check_outgoing.sh` before every push, no gate or tolerance changes, failures reported with numbers.

`$M` below is the elephant-1 bf16 checkpoint directory (`hf download baselabs/elephant-1`).
Card slugs: the runners derive them from the GPU name (e.g. `rtx-pro-6000-blackwell-...`, `geforce-rtx-5090`).

## T0: setup and smoke

Goal: a clean build and the data-free tests on every card.

1. Setup from scratch per AGENT.md §3: both checkouts side by side, cuda-rust at `CUDA_RUST_REV`.
2. `scripts/x86/setup.sh` (doctor `--core`, build, GPU-free tests).
3. `scripts/x86/run_tests.sh --model $M` (all GPUs, one at a time).

Acceptance:
- `setup.sh` ends with `x86 setup: OK`; `scripts/doctor.sh --core` reports cuda-rust at the pinned revision.
- Results committed: `benchmarks/results/x86/host/<date>/cpu-tests/` and `.../<card-slug>/<date>/gpu<N>-tests/` for
  N = 0, 1, 2, each with `metadata.json` and `summary.tsv`.
- PR table: per GPU and test target, pass/fail (and the failing test names). Expected NOT RUN: `engine_layers`,
  `kernels_nvfp4_layer0` (GB10 reference data), `cpu_ref` (slow), NVFP4 loader cases (no private checkpoint).
- Every failure is described: test name, assertion message in your own words (no paths), measured vs limit. No
  retry-until-green: if a test is flaky, say how often it failed out of how many runs.
- `platform_gpu` passes on every GPU (it proves each job ran on the GPU it locked).

## T1: hwprobe and platform profiles per card

Goal: the per-machine platform profile (PLATFORM.md P4) for each card type, and the raw GEMM/bandwidth rates.

1. `scripts/x86/run_bench.sh --model $M --mode single` covers it (hwprobe is its first job), or alone:
   `scripts/gpu.sh --gpu N ./target/release/hwprobe --profile-out <results dir>/profile.json`.
2. Commit `profiles/sm_120-<card-slug>.json` for the RTX PRO 6000 and the RTX 5090 (the runner writes them from the
   lowest-numbered GPU of each type) and each GPU's hwprobe output.

Acceptance:
- Two profiles (`sm_120-rtx-pro-6000-...json`, `sm_120-geforce-rtx-5090.json`), written by hwprobe (its save runs
  `PlatformProfile::check`, the same check the engine applies on load).
- PR table per GPU: memory bandwidth (GB/s), best bf16 GEMM TFLOP/s (cuBLAS, **fp32 accumulate**), GEMM-only
  forward tok/s at 1k/4k/16k, weight floor ms, crossover tokens.
- **Note on the GeForce 5090:** GeForce cards may run bf16 tensor-core MMA with fp32 accumulate at a fraction of
  the fp16-accumulate rate, unlike the RTX PRO 6000. State the measured fp32-accumulate bf16 rate of the 5090 vs the
  RTX PRO 6000 and their ratio, and how it compares to each card's published dense bf16 rate. sysone's precision
  policy requires fp32 accumulation (DESIGN §6.6): do not switch accumulators to get a number.
- The two RTX PRO 6000 agree within a few percent (if not, say which and by how much: thermal, power limit?).

## T2: kernel benchmarks per card, and SIMT on sm_120

Goal: the kernel-level picture on sm_120, including the first run of the cuda-oxide SIMT GDN kernels on sm_120
(`kernels-simt/artifacts/gdn_simt.sm_120.ptx`, built on GB10 with ptxas for sm_120 and never executed).

1. Correctness first, per GPU: `run_tests.sh` already ran `kernels_simt` and `kernels_gdn` (both backends);
   confirm `kernels_simt` printed `SIMT artifact sm_120: ok` and both pass.
2. `scripts/x86/run_bench.sh --model $M --mode single` (bench_gdn `--backend tile` and `--backend simt`,
   bench_attn, bench_nvfp4, tile_gemm_probe), or reuse T1's run if it is from the same commit.

Acceptance:
- `kernels_simt` and `kernels_gdn` pass on all three GPUs (or the exact failure is reported). If the SIMT kernels
  fail to load or run on sm_120, report the error text from sysone (not the log file) and stop there: the artifact
  is rebuilt on the GB10 side.
- PR table per card: GDN mixer ms per layer at the four bench_gdn shapes, tile vs simt and the speed-up, like the
  table in docs/G1.md "Results"; seg_attn TFLOP/s at 1k/4k/16k; NVFP4 vs bf16 GEMM rates at the MLP shapes;
  tile_gemm_probe bf16/NVFP4 rates.
- One paragraph: what is bandwidth-bound vs compute-bound on these cards compared to GB10 (G1.md, PHASE2A.md).

## T3: bench_forward bf16 per card, single GPU and all three concurrently

Goal: the number the user cares about: packed-forward throughput (tokens/s) per card and for the whole box.

1. Single GPU, one at a time: `scripts/x86/run_bench.sh --model $M --mode single` (bench_forward at 1k/4k/16k).
   On the 96 GB cards also `--forward-tokens 1024,4096,16384,32768,65536` (bigger steps may raise throughput; the
   5090 may run out of memory at large buckets: report the largest that fits).
2. All three concurrently: `scripts/x86/run_bench.sh --model $M --mode concurrent` (bench_forward on every GPU at
   once, each under its own lock).
3. Server level (throughput for bulk tagging): per GPU, start the server on that GPU and drive it with loadgen:
   ```bash
   scripts/gpu.sh --gpu N ./target/release/sysone --model $M --engine bf16 --listen 127.0.0.1:800N &
   ./target/release/loadgen --addr 127.0.0.1:800N --concurrency 256 --state-tokens 1000 --questions 8 --duration-s 120
   ```
   Also concurrency 64, and states of 4000 tokens. Save loadgen's JSON line per run as a `.json` file in a results
   directory with a `metadata.json` (copy the one from the same day's single run and adjust `kind`/`note`), stop the
   server, run check_outgoing.sh.

Acceptance:
- PR table: per card and bucket, replay ms, packed tok/s and the per-op class split (GEMM / GDN / norm+pointwise /
  attention); single vs concurrent tok/s per GPU and the box total (sum of the concurrent runs); the ratio to the
  GEMM-only estimate from hwprobe (DESIGN §10 target: >= 75% at T >= 4k).
- Server table: decisions/s, requests/s, prompt tok/s, p50/p99 latency per GPU, concurrency and state length.
- Any OOM or failure reported with the bucket and the card.

## T4: NVFP4 on sm_120

Goal: the NVFP4 kernels on sm_120.

- **If the user has provided the private NVFP4 checkpoint on this machine** (never fetch it otherwise):
  `scripts/x86/run_tests.sh --model $M --nvfp4 <dir>` (adds the NVFP4 loader cases and `engine_e2e_nvfp4`: the
  nvfp4 engine against the NVFP4 goldens, its batch invariance, graph replay, fused == standalone quantize) and
  bench_nvfp4 per card. Then the nvfp4 engine (docs/N1.md) per card:
  `scripts/gpu.sh --gpu N ./target/release/bench_forward --model <dir> --tokens 1024,4096,16384` (and
  `--nvfp4-unfused`), and the server: `scripts/gpu.sh --gpu N ./target/release/sysone --model <dir> --engine nvfp4
  --listen 127.0.0.1:800N` driven by loadgen as in T3.
- **Otherwise** kernel-level synthetic tests only: `kernels_nvfp4` minus the layer-0 case (already in `run_tests.sh`:
  bit-exact quantization vs compressed-tensors vectors and the CPU port, small GEMMs vs f64) and `bench_nvfp4`
  (synthetic weights at the MLP shapes).

Acceptance:
- `kernels_nvfp4` (synthetic cases) and `kernels_nvfp4_fused` pass on all three GPUs, or the failing cases are
  reported.
- PR table: NVFP4 GEMM TFLOP/s and quantize GB/s per card vs cuBLAS bf16 at the same shapes (bench_nvfp4 prints
  both); whether FP4 beats bf16 at 1k/4k/16k on each card.
- With the checkpoint: `engine_e2e_nvfp4` pass/fail with its printed dp numbers; a bench_forward table nvfp4 vs
  bf16 per card and bucket (replay ms, tok/s, per-class split) and the loadgen server table for `--engine nvfp4`.
- Nothing about the checkpoint beyond pass/fail, gate numbers and timings.

## T5: vLLM baseline for elephant-1 on one RTX PRO 6000

Goal: a speed baseline (DESIGN §10, baseline 2). Speed only: vLLM runs the model causally, so its probabilities
are not comparable.

1. Docker with the NVIDIA container toolkit; an official vLLM image that supports elephant-1's architecture (check
   `config.json` `architectures` against the vLLM release notes). Record the image tag and digest.
2. Run it on one RTX PRO 6000 under that GPU's lock:
   `scripts/gpu.sh --gpu N docker run --rm --gpus device=N ...` with automatic prefix caching on.
3. Method (DESIGN §10): one completion request per question, `max_tokens=1`, logprobs restricted to the answer slot
   token ids (`allowed_token_ids` or an equivalent), the same synthetic workload as T3's loadgen runs (state length,
   questions per request, concurrency). A small client script is a code PR under `benchmarks/vllm/` (Python, run
   with `python -I`).

Acceptance:
- PR table: decisions/s and prompt tok/s for vLLM vs sysone (T3 server numbers) on the same card and workload,
  with concurrency and state length; the vLLM version, image digest and flags used.
- No model or image is pushed anywhere; the client script contains no host-specific paths.

## T6: authoritative x86 reference goldens (later, needs coordination)

Goal: the authoritative golden tier (DESIGN §9.1): the reference (qev + serve.patch) run in the pinned image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` on x86, to compare with the GB10 dev goldens.

Do not start this from TASKS.md: it needs the reference sources and a data-transfer plan agreed in its issue (which
files, how large, where they go). The issue will give the exact steps.

## T7: bulk throughput on all three GPUs

Goal: the user's real workload, bulk tagging, on the whole box: `sysone bulk` (docs/BULK.md) on each card alone and
on all three cards at once from one shared queue, against the HTTP server driven by `loadgen` on the same corpus.

1. Tests first: `scripts/x86/run_tests.sh --model $M` now includes `bulk` (CPU, with the script tests' fake
   nvidia-smi: resume, packing determinism, load balancing, gpu.sh locks) and `bulk_gpu` per GPU (bulk vs the
   server on that GPU; it sizes itself to the card: 16,384-token steps and no two-engine part below 64 / 80 GiB).
2. Smoke: `scripts/x86/run_bulk.sh --model $M --mode both --server --quick` (200 documents; not for reporting).
3. The run: `scripts/x86/run_bulk.sh --model $M --mode both --server`. It writes the corpus (2,000 synthetic
   documents, states 300..8,000 tokens, 4..16 questions; local, under `target/x86-bulk/`), the packing simulation
   (`host/<date>/bulk-corpus/`), then per GPU, one at a time: `sysone bulk` on that GPU, the server on that GPU with
   `loadgen --corpus` at concurrency 256, and `sysone bulk compare` of the two result files
   (`<card-slug>/<date>/gpu<N>-bulk/`); then `sysone bulk` on all GPUs at once (`host/<date>/bulk-all/`) and its
   comparison with each single-GPU run.
4. Step size on one RTX PRO 6000 (single GPU, after step 3): the bulk run with `--max-step-tokens 16384` and with
   `65536` (`scripts/gpu.sh --gpu N ./target/release/sysone bulk --model $M --input target/x86-bulk/corpus-2000.jsonl
   --output target/x86-bulk/step-<T>.jsonl --gpus N --require-idle --max-step-tokens <T>`); report the summary lines.

Acceptance:
- Every bulk run writes 2,000 results with status 200, and its summary has `failed` 0, `unrunnable` 0, `bad_lines` 0
  and `aborted` null; every worker reports the buckets it chose (the 5090's top bucket comes from its memory).
- `compare-server-vs-bulk.txt` per GPU and `compare-gpu<N>-vs-all.txt`: `only_a` = `only_b` = 0,
  `status_mismatches` = 0, `other_mismatches` = 0, `dp_max` <= 0.01822 (`[batch_invariance].dp_max`: the two sides
  pack the requests differently), any `choice_flips` only with `margin_b` <= 0.02. Report the numbers, never rerun
  until they pass.
- PR table per GPU: bulk docs/s, decisions/s, packed tok/s, padding fraction, busy fraction, top bucket, load s;
  server (loadgen.json): requests/s, decisions/s, input tok/s, `server_padding_frac`, `server_engine_busy_frac`,
  p50/p99 latency; the ratio bulk / server in packed tok/s.
- All-GPU table: total docs/s and packed tok/s, per GPU token share, steps and busy fraction; the sum of the three
  single-GPU packed tok/s and the all-GPU total as a fraction of it (scaling); whether the 5090 took a share in line
  with its single-GPU speed.
- Step-size table (item 4): packed tok/s and padding at 16,384 / 32,768 (step 3) / 65,536 on the RTX PRO 6000.
- Anything that failed (OOM, a worker that stopped, a lock wait) is reported with the GPU and the summary's `error`
  field, in your own words.

## T8: profile and NVFP4 speed on sm_120 (F1)

Goal: repeat F1's profile (docs/PROFILE.md) on each card and check that F1's NVFP4 changes (docs/F1.md: the fused
gate/up + SiLU·mul + quantize FP4 GEMM, occupancy hints on the norms and SiLU·mul) are correct and pay on sm_120.
GB10 tuned them; nothing has run on sm_120 yet. Needs the NVFP4 checkpoint on this machine for the engine parts
(never fetch it otherwise); the kernel parts run without it.

1. Tests: `scripts/x86/run_tests.sh --model $M` now runs `kernels_nvfp4_f1` per GPU (synthetic weights: the fused
   GEMM, every tile shape and occupancy hint the engine uses, byte-identical to the unfused kernels). With the
   checkpoint, `--nvfp4 <dir>` runs `engine_e2e_nvfp4`, which now includes `f1_launches_equal_n1_launches` (the
   engine with F1's launches vs N1's, bit for bit, eager and replayed); the bf16 `engine_e2e` runs it too.
2. Microbenchmarks per card (no checkpoint): `scripts/gpu.sh --gpu N ./target/release/bench_nvfp4_mlp --tokens
   1024,4096,16384,32768 --norms --gemm-opts o1,o2,w4,w8 --swiglu 128x128g8,128x128g8/o2,128x128g8/o1,64x128g8`
   and `bench_nvfp4 --sweep --no-quant --tokens 4096,16384` (tile choice per M; on GB10 128x128x128 grouped-M 8
   won at every M).
3. Forward per card: `bench_forward --model <nvfp4 dir> --tokens 1024,4096,16384,32768` three ways: default (F1),
   `--nvfp4-separate-silu --n1-tuning` (N1's launches in the same build), and the bf16 checkpoint.
4. Profile per card, as docs/PROFILE.md "Method" (exact commands there): Nsight Systems on bench_forward (try
   `--trace=cuda --cuda-graph-trace=node` first: on a discrete GPU the hardware trace may give per-node replay
   timing, which GB10 could not; fall back to `--trace=cuda-sw` and `scripts/profile/nsys_forward.py`), then
   Nsight Compute on the top kernels at 16k with the sections and raw metrics listed there (on sm_120 `dram__*`
   metrics exist: add `dram__throughput.avg.pct_of_peak_sustained_elapsed`). Keep reports and traces local; commit
   only the script summaries.

Acceptance:
- `kernels_nvfp4_f1` passes on all three GPUs (0 mismatches everywhere), or the failing case is reported with its
  printed mismatch counts. With the checkpoint: `engine_e2e_nvfp4` 5/5 incl. `f1_launches_equal_n1_launches`
  (84/84 steps bit-identical), and bf16 `engine_e2e` 5/5.
- PR table per card: bench_nvfp4_mlp (N1 gate/up + `silu_mul_nvfp4` vs fused GEMM with and without its occupancy
  hint; norms by occupancy); bench_forward replay ms and packed tok/s at 1k/4k/16k/32k for F1, N1 launches and
  bf16; whether each F1 choice (fused GEMM, `swiglu_occupancy` 2, norm occupancy 2, SiLU occupancy 3) is faster,
  equal or slower than N1's on that card. A choice that is slower on sm_120 is reported, not changed (the GB10 side
  makes the per-arch decision).
- Profile summary per card (one table like PROFILE.md (a) and (b)): kernel time per class, replay bubbles, step
  boundary, and for the FP4 GEMMs / fused GEMM / cuBLAS GEMMs / norms / scan: SM %, memory %, tensor-pipe %,
  occupancy and the limiter. Whether PDL would recover ≥ 2% of the forward there (PROFILE.md (c)).
- Nothing about the checkpoint beyond pass/fail, gate numbers and timings.
