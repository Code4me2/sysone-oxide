# Getting started

Run the public BF16 elephant-1 checkpoint as a text decision service. sysone accepts qev-style state/questions
and returns probabilities; it does not provide chat completions, generated prose, images or audio.
See [Performance](PERFORMANCE.md) for the completed comparison, not a promise for every workload.

## Setup and build

### Prerequisites

- Linux; the documented x86 setup targets Ubuntu 24.04. [Platform notes](PLATFORM.md) also cover DGX Spark.
- NVIDIA GPU, R580+ driver, **CUDA toolkit ≥13.3 with `tileiras`** (the prerequisite package set uses CUDA 13.4).
  Set `CUDA_TOOLKIT_PATH` in your shell if the toolkit is not at the expected `/usr/local/cuda-13` location.
- clang and libclang development/resource headers, CMake, a C/C++ toolchain, pkg-config, Git, Bash and `flock`
  (util-linux). Install prerequisites through your normal administrator-approved process.
- rustup with the host toolchain in [`rust-toolchain.toml`](../rust-toolchain.toml) (currently 1.98.0).
  Building may fetch that toolchain and Cargo dependencies if they are not already available.
- Hugging Face CLI (`hf`) for the public model download, and curl for the example HTTP requests.

**CPU/mock runtime does not mean a CUDA-free build**: this crate still builds CUDA-dependent crates.
The checked-in SIMT artifacts normally avoid rebuilding cuda-oxide kernels; their separate compiler toolchain is
only needed for kernel development. Docker and the Python/qev reference stack are not serving prerequisites.

### Clone the separate repositories and pin cuda-rust

Run these commands in a new working directory. Access to the sysone repository may require permission already
arranged by its owner; the public checkpoint does not require private-model access.

```bash
mkdir sysone-work
cd sysone-work
git clone https://github.com/Code4me2/sysone-oxide.git sysone
git clone https://github.com/NVIDIA/cuda-rust.git cuda-rust
cd sysone
CUDA_RUST_PIN=$(awk 'NF && $1 !~ /^#/ {print $1; exit}' CUDA_RUST_REV)
git -C ../cuda-rust checkout "$CUDA_RUST_PIN"
```

The layout is `sysone-work/{sysone,cuda-rust}`. `cuda-rust` is a **separate adjacent repository**, not a submodule
or a Cargo registry dependency: [`Cargo.toml`](../Cargo.toml) uses `../cuda-rust/...` paths. Keep it clean at the
exact revision in [`CUDA_RUST_REV`](../CUDA_RUST_REV); do not substitute its latest main or patch it locally.

### Download the public tested checkpoint and build

From the sysone root, pin the model to the tested public revision:

```bash
MODEL_DIR=../models/elephant-1
hf download baselabs/elephant-1 \
  --revision b3f8a93462fb1051a9de031e94a7ff380e254072 --local-dir "$MODEL_DIR"
cargo build --release --bins
```

Allow about 19 GB for the checkpoint download, plus build/cache space. The model directory must contain
`config.json`, `readout.json`, `model.safetensors`, `tokenizer.json` and `chat_template.jinja`; keep the other downloaded metadata
alongside them. No `serve.patch` application is needed for sysone; that patch belongs to the qev reference.

`scripts/doctor.sh --core` diagnoses build/runtime prerequisites, including the pinned cuda-rust checkout,
CUDA tooling and NVIDIA driver. It also probes Docker and HF authentication but treats them as optional in core
mode; neither is required for public BF16 serving. It queries the environment rather than checking files alone.

## First real server and request

The HTTP server **defaults to `--engine mock`**, which returns fake logits. Always select `--engine bf16` for
this checkpoint. `--name elephant-1` pins the API model name rather than deriving it from a resolved path/symlink.

These 16K limits, 32 prefixes and four CPU workers are **explicit example choices**, not changed defaults.
The command also matches the comparison's bucket arm settings; it does not select exact eager.
Run from the sysone root, choosing a GPU you are authorized to use:

```bash
scripts/gpu.sh --gpu 0 ./target/release/sysone \
  --model "$MODEL_DIR" --name elephant-1 --listen 127.0.0.1:8000 \
  --engine bf16 --max-branch-tokens 16384 --max-request-tokens 16384 \
  --max-step-tokens 16384 --token-budget 16384 --max-prefixes 32 \
  --cpu-workers 4 --gdn-backend tile --gemm pinned
```

Wait for the serving message. Initial loading/JIT/warmup can take time. In a second terminal:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/health
curl --fail-with-body -sS 'http://127.0.0.1:8000/v1/systemone?debias=1' \
  -H 'content-type: application/json' -d '{
    "model": "elephant-1",
    "state": "The invoice is 40 days overdue.",
    "questions": {
      "late": {"type": "noul", "instructions": "Is the invoice late?"},
      "route": {"type": "choice", "criteria": {"billing": "Billing team", "support": "Technical support"}},
      "urgency": {"type": "score", "criteria": ["low", "medium", "high"]}
    }
  }'
```

`/health` is **service metadata, not proof of real inference or numerical correctness**; mock looks healthy too.
Check the startup engine choice and `/metrics` engine information, then validate answers on your own cases.
The request above illustrates the API, not a numerical acceptance test or a fixed expected answer.

## API essentials

- `state` can be a text string or JSON value. `questions` is a nonempty object keyed by your question ids.
- `noul` uses `instructions` and optional `criteria` with `true`/`false` descriptions. Its answer has `noul`,
  the true probability (not a boolean).
- `choice` has a `criteria` object of named options (1–255). Answers include `choice`, `probabilities`,
  and `confidence`. `score` has an ordered `criteria` array (1–10 levels); answers include `score`, `legend`,
  `probabilities` and `confidence`. Both also accept `instructions`.
- Responses contain `model`, `answers`, `usage` and `qev` metadata/timings. `usage.output_tokens` counts decisions,
  not generated prose; `usage.input_tokens` counts the packed rendered request.
- Accepted model names include the served name and aliases `jev-latest`, `jev-preview`, `qev-latest`.
- `?debias=k` averages cyclic option-order shifts; more shifts can add branches and token cost. The fallback is
  `--debias`, then `QEV_DEBIAS`, then 1. For controlled experiments pass `?debias=1` explicitly.
- `?debug=1` adds diagnostics. Any nonempty debug value, **even `debug=0`**, enables it. GPU engines do not
  provide `allowed_token_mass`. Do not expose diagnostics to untrusted clients.
- `GET /v1/models` lists model metadata; `GET /metrics` provides Prometheus text.
- `POST /v1/systemone:batch` is a sysone extension: a JSON array of request bodies, ordered per-item results.
  HTTP 200 can contain failed items; inspect each item's `status` and `body`.

Compatibility is scoped to the supported text API and tested cases, not universal error/multimodal conformance.
[Architecture](ARCHITECTURE.md#42-api-srcapi) describes intentional differences such as overload responses.

## Execution modes

| Choice | Behavior |
|---|---|
| `--engine mock` (HTTP default) | Real frontend/scheduler, deterministic fake logits, no inference GPU |
| `--engine bf16`, neither eager flag | Bucket-sized CUDA graphs; pinned GEMMs and last-layer trimming default on |
| BF16 + `--no-cuda-graphs` | Bucket-sized eager execution, still padded to capacity |
| BF16 + `--gdn-backend tile --exact-token-eager` | Logical token/answer rows, eager only; explicit opt-in |

To try the measured exact arm, stop the first server and rerun its full command with
**`--exact-token-eager --bounded-exact-warmup`** appended. Do not add `--no-cuda-graphs`: it is mutually exclusive
with exact. Exact requires BF16 and explicit Tile; it is not an NVFP4 mode. The separate NVFP4 engine requires a
matching checkpoint and is not needed for this public BF16 guide or included in the comparison.
No default promotion or deployment recommendation is implied.

Bounded exact warmup makes 51 fixed calls. Its 90-second bound is checked **between calls**, not by a device-call
watchdog; use an external startup deadline for a potentially hung call. It does not precompile every possible
future shape. [Exact eager guide](x86/README.md) details the plan, failure policy and residual risks.

For a no-inference frontend check, the same model's tokenizer/config/template are still needed:

```bash
./target/release/sysone --model "$MODEL_DIR" --name elephant-1 \
  --listen 127.0.0.1:8000 --engine mock
```

## Token limits and memory

Limits apply **after rendering and tokenization**, including the system/chat template, question instructions,
option descriptions and any debias shifts. Words, bytes and raw state tokens alone are not the limit quantity.

- `--max-branch-tokens`: shared prefix + longest question suffix. Public BF16 profile default: **32,768**.
- `--max-request-tokens`: prefix once + **all** suffixes. Public BF16 profile default: **65,536**.
- `--token-budget`: batch-closing target, default **16,384**; a larger admitted request runs alone.
- `--max-step-tokens`: must cover **both** token budget and max-request-tokens because requests are never split
  across scheduler steps. Its default is the larger of those two, not 16K.

There is no silent truncation. More questions, longer option text and debias can exceed the whole-request limit
with a state that easily fits one branch. Keep headroom and inspect `usage.input_tokens` on accepted samples.
The quickstart deliberately lowers both request limits to 16K and sizes the engine to match.

The download size is not a VRAM requirement. Weights, activation arenas, GDN scratch/state, fixed-capacity
metadata, CUDA libraries and graph/JIT resources all need room. Smaller step capacity can reduce allocations;
exact eager still retains capacity-backed arenas. `--max-prefixes` allocates 2 MiB of GDN state per prefix in
addition to other scratch. Reduce batch/step capacity and request limits together if memory is tight; do not
infer fit on a smaller GPU from the single-card comparison. Unified-memory GB10 also competes with host memory.
[Platform notes](PLATFORM.md) and [bulk memory sizing](BULK.md#bucket-ladder-for-bulk) explain the distinction.

## Bulk document tagging

Use bulk when you want offline results rather than HTTP latency. The default BF16 **process coordinator handles
per-GPU locks internally**; do **not** wrap these commands in `gpu.sh`. Unlike the HTTP CLI, bulk defaults to BF16.
It does not expose the HTTP exact-eager/warmup flags.

Create `questions.json` and `documents.jsonl`:

```json
{"questions":{"late":{"type":"noul","instructions":"Is the invoice late?"}}}
```

```jsonl
{"id":"invoice-1","state":"The invoice is 40 days overdue."}
{"id":"invoice-2","state":"The invoice is due next month."}
```

```bash
./target/release/sysone bulk --model "$MODEL_DIR" --name elephant-1 \
  --engine bf16 --gdn-backend tile --gpus 0 --questions questions.json --docs documents.jsonl \
  --output results.jsonl --summary bulk-summary.json \
  --max-branch-tokens 16384 --max-request-tokens 16384 --max-step-tokens 16384
```

Alternatively, put full request bodies in `jobs.jsonl`, one `{"id":...,"body":{...}}` per line, and replace
`--questions ... --docs ...` with `--input jobs.jsonl`. A directory of UTF-8 text files also works with `--docs`.
Select several distinct GPUs with e.g. `--gpus 0,1`; each gets its own worker process, not tensor parallelism.

Results append in completion order, keyed by id with `status` and `body`. Rerun with unchanged inputs/settings to
resume; already written ids (including error responses) are skipped. Use a fresh output for changed questions,
models or settings. Inspect error counts and process exit status, not just the presence of an output file.
[Bulk guide](BULK.md) covers job formats, crash recovery, simulation and result comparison.

## Operational safety

- All direct GPU server, benchmark and test commands must use `scripts/gpu.sh --gpu N`. One job per GPU; wait for
  the lock. Do not override `SYSONE_GPU_LOCK` on a multi-GPU host, steal locks or kill someone else's job.
- Bulk process workers and the x86 runner scripts acquire their own locks; do not wrap their coordinators.
  Keep `scripts/` with the checkout so bulk can find its worker wrapper.
- Published benchmark results require an otherwise idle selected GPU. Do not use `--allow-shared-gpu` to report
  numbers, or alter clocks/power limits to force a result.
- Examples explicitly bind **127.0.0.1:8000**; the actual CLI default is `0.0.0.0:8000`. There is no built-in
  authentication/TLS. For remote access, put TLS, authentication, access restrictions, body limits and timeouts
  in a separately managed reverse proxy. Do not expose `/metrics` or diagnostics publicly.
- Queues and body budgets are bounded, but are not total process-memory limits. HTTP bodies default to at most
  256 MiB each and 1 GiB in flight; parsed plans/responses have additional overhead. Size access controls and
  request limits for your workload.

## Testing layers

These commands target a prepared development environment. CPU execution still requires the build prerequisites;
the known failure below means the suite is not globally green.

```bash
# CPU-only unit tests; no checkpoint (includes the known failing scheduler case below).
cargo test --release --lib
# CPU-only golden shape/script checks; no inference GPU.
cargo test --release --test render_golden_shape --test scripts
# CPU frontend tests need the public checkpoint's tokenizer/template/config.
SYSONE_MODEL_DIR="$MODEL_DIR" cargo test --release --test frontend_golden --test frontend_assemble
# A GPU engine check: only on an authorized idle GPU, with public weights.
SYSONE_MODEL_DIR="$MODEL_DIR" scripts/gpu.sh --gpu 0 cargo test --release \
  --test engine_e2e -- --nocapture --test-threads=1
```

**Known failure:** `sched::pool::tests::new_work_is_bounded_and_finishing_work_is_not` remains
**FAILED / NOT RERUN**. Do not skip it and relabel the suite green, or change a tolerance to obtain a pass.

Kernel/engine tests, live HTTP tests, batch invariance and external-reference comparisons are separate layers.
Some tests require private checkpoints, generated NVFP4 data or large hidden-state dumps not in the repository;
some fail on missing fixtures while others skip. Neither behavior establishes coverage of the absent data.
The targeted exact GPU test has a CPU planning case and a separately ignored GPU case with additional real-lock
requirements; do not interpret the default CPU case as GPU qualification.

The dependency-heavy [reference harness](../reference/README.md) regenerates goldens for developers. It has
machine-specific defaults and inconsistent overrides; it is not a portable serving installer and is **not needed
to serve public BF16**. See [Architecture testing](ARCHITECTURE.md#6-correctness-and-testing) for fixture layers
and [the x86 guide](x86/README.md) for preserved qualification outcomes.

## Troubleshooting

| Symptom | Check |
|---|---|
| Healthy but implausible answers | Confirm `--engine bf16`, startup logs and engine metrics; mock also reports healthy. |
| Missing CUDA crate / bindgen / `tileiras` error | Adjacent clean pinned cuda-rust, CUDA toolkit path/version, clang resource headers; mock builds need these too. |
| Wrong model name / 404 | Pass `--name elephant-1`; use that name or an accepted alias in the body. |
| 422 length error | Count the fully rendered branches, not just document text; shorten input/questions or deliberately resize all relevant limits. |
| Startup rejects max-step / exact flags | Step must cover budget and request limit; exact needs BF16 + Tile and cannot combine with `--no-cuda-graphs`. |
| OOM or slow first requests | Lower capacity/limits together, reserve VRAM headroom and allow JIT; bounded warmup does not cover every shape. |
| 503 overload | Reduce client concurrency or inspect queue/body overload metrics; do not just raise every queue limit. |
| 413 | Body too large or `:batch` exceeds admission capacity; split the call. |
| 500 | Inspect local server errors, engine/checkpoint match and query values (`debias` must be an integer); sanitize logs before sharing. |
| Command waiting before launch | GPU lock may be held; wait rather than bypassing it. |

For performance evaluation, start with [a workload-specific measurement plan](PERFORMANCE.md#evaluate-your-own-workload),
not the assumption that exact eager or any default wins on every shape/device.
