# sysone

A Rust/CUDA inference server for **System One text decisions** with
[`baselabs/elephant-1`](https://huggingface.co/baselabs/elephant-1), built on NVIDIA's
[cuda-rust](https://github.com/NVIDIA/cuda-rust).

Give it a document or JSON state and named questions. It returns qev-compatible decision JSON:

- **`noul`** — a true/false probability;
- **`choice`** — a selected option and its probability distribution;
- **`score`** — a score over ordered criteria, with probabilities.

This is **not a general chat, text-generation or multimodal server**. It scores answer slots rather than
streaming generated text. Shared document prefixes and question branches are packed into a single forward
pass; multiple requests can share a step without sharing their state.

**Start here:** [Getting started](docs/GETTING_STARTED.md) · [Performance](docs/PERFORMANCE.md) ·
[All documentation](docs/README.md)

## Performance against optimized qev

Sysone can run faster by batching work across requests, avoiding padded computation in exact eager mode, and
trimming selected final-layer operations to answer rows while retaining keys/values for every token. The
measurements below evaluate the combined server configuration, not each optimization's contribution in isolation.

**Up to 68% higher decision throughput**, with **12–68% gains across all 8 tested cells**, using sysone's
**opt-in exact eager BF16/Tile mode** against the fastest verified qev backend in the completed tests.
The baseline used real FLA GatedDeltaNet and verified successful CUDA Triton launches, not fallback Torch timings.
This is not a claim of globally optimal qev tuning.

| Sysone mode | Throughput relative to optimized qev | Tested cells won |
|---|---:|---:|
| Exact eager + bounded warmup (opt-in) | **1.12–1.68×** | **8 / 8** |
| Bucketed CUDA graphs (BF16 execution default) | **0.65–1.58×** | **4 / 8** |

The HTTP server itself defaults to **mock**, not BF16. Neither default configuration implies the exact-mode gains.

The comparison used the same public BF16 checkpoint, matched requests and 16K rendered-token limits on **one
NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120)**: 1,000/4,000-word states, 1/8 questions,
concurrency 1/16. At 1,000 words, eight questions and concurrency 16, exact averaged **65.6 decisions/s versus
qev's 39.1**. Rates count one decision per question, not generation tokens/s.
Two short fresh-process observations per arm are not sustained-capacity or statistical-significance evidence.
Outputs agreed within unchanged tolerances, not bit for bit; near-tie choices can differ.

[All eight absolute rates, configuration, correctness and limitations](docs/PERFORMANCE.md) ·
[Curated numerical summary](benchmarks/results/x86/qev-serving-optimized/summary.json)

## Set up and build

You need Linux, a supported NVIDIA GPU with an R580+ driver, **CUDA toolkit ≥13.3 including `tileiras`**,
clang/libclang headers, C/C++ build tools, and rustup. The host Rust toolchain is pinned in
[`rust-toolchain.toml`](rust-toolchain.toml).
Even the CPU/mock build uses CUDA dependencies; this is not a CUDA-free build.

Keep two separate repositories side by side: this checkout and `../cuda-rust`, at the exact revision in
[`CUDA_RUST_REV`](CUDA_RUST_REV). The CUDA crates are local path dependencies, not downloaded by Cargo from Git.
You also need the public elephant-1 checkpoint files, about 19 GB of download plus runtime memory headroom.

Follow [the clone, pin and public model download instructions](docs/GETTING_STARTED.md#setup-and-build), then,
from the sysone repository root:

```bash
cargo build --release --bins
```

No Python reference environment, golden regeneration or private checkpoint is needed to serve public BF16.
[Platform prerequisites and caveats](docs/PLATFORM.md) cover the GPU/toolkit differences.

## Launch and ask a question

**Warning: omitting `--engine bf16` starts the deterministic mock engine with fake logits.** A healthy-looking
`/health` response does not prove real inference or numerical correctness.

After the setup above, use the downloaded model directory. These are **explicit 16K example limits**, not the
checkpoint defaults. This launch uses bucketed graphs, **not** the exact-mode headline configuration.

```bash
MODEL_DIR=../models/elephant-1
scripts/gpu.sh --gpu 0 ./target/release/sysone \
  --model "$MODEL_DIR" --name elephant-1 --listen 127.0.0.1:8000 \
  --engine bf16 --max-branch-tokens 16384 --max-request-tokens 16384 \
  --max-step-tokens 16384 --token-budget 16384 --max-prefixes 32 \
  --cpu-workers 4 --gdn-backend tile --gemm pinned
```

To select the measured exact mode instead, stop the server and append
**`--exact-token-eager --bounded-exact-warmup`** to that command. This is BF16/Tile-only opt-in, not a change of
defaults or a deployment recommendation. Do not combine it with `--no-cuda-graphs` or NVFP4.
See [execution modes and warmup limits](docs/x86/README.md).

In another terminal:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/health
curl --fail-with-body -sS 'http://127.0.0.1:8000/v1/systemone?debias=1' \
  -H 'content-type: application/json' -d '{
    "model": "elephant-1",
    "state": "The invoice is 40 days overdue.",
    "questions": {"late": {"type": "noul", "instructions": "Is the invoice late?"}}
  }'
```

Read `answers.late.noul` for the true probability; `usage` and `qev` contain token counts and timing metadata.
Questions/options, templates and any debias branches all count toward the rendered-token limits, not just the
raw document. Requests above a limit are rejected, not silently truncated.

The server has **no built-in authentication or TLS**. Keep it bound to loopback; remote access needs a separately
managed reverse proxy with TLS, authentication and access controls. Every GPU server, test or benchmark command
must hold its GPU's lock through `scripts/gpu.sh --gpu N`. Do not bypass another job's lock.

## Tag many documents

For offline work, use **`sysone bulk`**: resumable JSONL input/output, window packing and one worker process per
selected GPU. The process coordinator starts workers under their own GPU locks; **do not wrap it in `gpu.sh`**.

```bash
# jobs.jsonl: one {"id": ..., "body": {"state": ..., "questions": ...}} per line
./target/release/sysone bulk --model "$MODEL_DIR" --name elephant-1 \
  --engine bf16 --gdn-backend tile --gpus 0 --input jobs.jsonl --output results.jsonl \
  --max-branch-tokens 16384 --max-request-tokens 16384 --max-step-tokens 16384
```

Or reuse one question set with `--questions questions.json --docs documents.jsonl` instead of `--input`.
Results are keyed by id, not input position. Reuse the same inputs/settings when resuming; use a new output file
for a different experiment. Bulk has its own CLI and is not the measured exact HTTP mode.
[Formats, question sets and resume behavior](docs/BULK.md).

## Where next?

| Guide | Read it for |
|---|---|
| [Getting started](docs/GETTING_STARTED.md) | Full setup, API essentials, limits, testing and troubleshooting |
| [Performance](docs/PERFORMANCE.md) | Completed optimized-qev comparison and what it does not establish |
| [Bulk tagging](docs/BULK.md) | Offline jobs and shared question sets |
| [Architecture](docs/ARCHITECTURE.md) | Request lifecycle, modules and testing layers |
| [Platforms](docs/PLATFORM.md) | GPU differences and prerequisites |
| [Documentation index](docs/README.md) | Developer references and preserved historical reports |

Known limits include text-only input, no cross-request prefix cache, and no GPU `allowed_token_mass` diagnostics.
The Rust suite is **not globally green**: the known scheduler-pool case remains **FAILED / NOT RERUN**;
[testing details](docs/GETTING_STARTED.md#testing-layers) distinguish missing fixtures from failures.
