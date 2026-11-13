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
  `scripts/x86/run_tests.sh --model $M --nvfp4 <dir>` (adds the NVFP4 loader cases) and bench_nvfp4 per card.
- **Otherwise** kernel-level synthetic tests only: `kernels_nvfp4` minus the layer-0 case (already in `run_tests.sh`:
  bit-exact quantization vs compressed-tensors vectors and the CPU port, small GEMMs vs f64) and `bench_nvfp4`
  (synthetic weights at the MLP shapes).

Acceptance:
- `kernels_nvfp4` (synthetic cases) pass on all three GPUs, or the failing cases are reported.
- PR table: NVFP4 GEMM TFLOP/s and quantize GB/s per card vs cuBLAS bf16 at the same shapes (bench_nvfp4 prints
  both); whether FP4 beats bf16 at 1k/4k/16k on each card.
- Nothing about the checkpoint beyond pass/fail and timings.

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
