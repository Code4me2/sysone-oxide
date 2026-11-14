# sysone

An inference server for System One decision models (`baselabs/elephant-1`), written in Rust on
[cuda-rust](../cuda-rust). It is a drop-in replacement for OneJev's `qev` server: same `POST /v1/systemone` request
and response JSON, same error bodies, and probabilities that match the reference within measured tolerances. It gets
its speed by packing many requests into one forward pass: each request's shared prefix (system prompt + state)
appears once, followed by one short suffix per question branch.

Forked from grout. grout's Qwen3 engine is kept read-only under [legacy/grout](legacy/grout) as porting reference.

## Status

Phase 2a is done ([docs/PHASE2A.md](docs/PHASE2A.md) has the numbers):

- The server, frontend and scheduler are complete for text requests and match qev on the golden corpus.
- The bf16 GPU engine (`--engine bf16`) runs the full packed forward with T-bucket CUDA graphs. It passes the
  end-to-end probability gate and the batch-invariance gate. The per-layer hidden-state gate still fails on the last
  layers; a new definition of that gate is pending.
- NVFP4 kernels and loading exist, but there is no NVFP4 forward yet. `allowed_token_mass` (`?debug=1`) is not
  computed by the GPU engine. No prefix cache, no images.
- A faster GatedDeltaNet scan on cuda-oxide SIMT kernels (G1, [docs/G1.md](docs/G1.md)) runs in the engine and passes
  every accuracy gate (E2, [docs/E2.md](docs/E2.md)); it is the default on sm_121 (`--gdn-backend auto`), cuTile
  elsewhere. The forward is 18-22% faster with it.
- Bulk tagging (B1, [docs/BULK.md](docs/BULK.md)): `sysone bulk` answers a JSONL file of requests with the server's
  answers, packed for throughput (full steps from a window of requests) over one engine per GPU, resumable.

## Prerequisites

- `../cuda-rust` checked out next to this repository (the crates are path dependencies), at the revision in
  `CUDA_RUST_REV` (upstream https://github.com/NVIDIA/cuda-rust).
- An NVIDIA GPU with an R580+ driver, CUDA toolkit ≥ 13.3 with `tileiras`, clang/libclang headers, rustup (the
  toolchain is pinned in `rust-toolchain.toml`). [docs/PLATFORM.md](docs/PLATFORM.md) lists them per platform.
- For the model-dependent tests and for serving: the elephant-1 checkpoint directory (`config.json`,
  `model.safetensors`, `tokenizer.json`, `chat_template.jinja`). Tests read it from `$SYSONE_MODEL_DIR`.

`scripts/doctor.sh` checks every prerequisite and prints the fix for anything missing.

## Build, test, run

```bash
cargo build --release --all-targets

# Unit tests: CPU only, no checkpoint needed.
cargo test --release --lib

# Everything else. Any command that uses the GPU runs under the shared GPU lock.
# Not all green today: engine_layers needs the hidden-state dumps under ../sysone-ref and its gate (a) is being
# redefined (docs/PHASE2A.md); kernels_nvfp4's layer-0 test fails unless reference/nvfp4/data was generated.
# See docs/ARCHITECTURE.md §6.3 for each test's data prerequisites.
scripts/gpu.sh cargo test --release --all-targets

# Serve elephant-1 on the GPU (default listen address 0.0.0.0:8000).
# --engine defaults to mock: without --engine bf16 the server looks healthy but returns fake logits.
./target/release/sysone --model /path/to/elephant-1 --engine bf16
# GDN delta rule: --gdn-backend auto (default: simt on sm_121, tile elsewhere) | tile | simt; the engine tests
# and bench_forward read SYSONE_GDN_BACKEND.

# Serve with the deterministic mock engine (real frontend and scheduler, fake logits; no GPU needed).
./target/release/sysone --model /path/to/elephant-1 --engine mock

# Load test a running server (or send every job of a bulk job file once: --corpus FILE).
./target/release/loadgen --addr 127.0.0.1:8000 --concurrency 64 --state-tokens 1000 --questions 8 --duration-s 60

# Bulk tagging: a JSONL of {"id", "body"} jobs in, one result line per id out, on every GPU (one worker process per
# GPU, each under its gpu.sh lock); rerun the same command to resume. docs/BULK.md.
./target/release/sysone bulk gen --docs 2000 --output corpus.jsonl
./target/release/sysone bulk --model /path/to/elephant-1 --input corpus.jsonl --output results.jsonl

# Benchmarks refuse to start while another process uses their GPU. On a multi-GPU host pick the GPU with
# scripts/gpu.sh --gpu N (one lock per GPU); the x86 runners in scripts/x86/ do this per card (docs/x86/AGENT.md).
scripts/gpu.sh ./target/release/bench_forward --model /path/to/elephant-1 --tokens 1024,4096,16384
scripts/gpu.sh ./target/release/hwprobe
```

`./target/release/sysone --help` lists every server flag (token budget, max wait, queue sizes, CPU workers,
`--pin-cores`, length limits, bucket sizing). Requests:

```bash
curl -s localhost:8000/health
curl -s 'localhost:8000/v1/systemone?debias=1' -H 'content-type: application/json' -d '{
  "state": "The invoice is 40 days overdue.",
  "questions": {"late": {"type": "noul", "instructions": "Is the invoice late?"}}
}'
```

See [docs/ARCHITECTURE.md §6](docs/ARCHITECTURE.md#6-correctness-and-testing) for which tests need which data, and
§7 for every benchmark binary.

## Repository map

| Path | Contents |
|---|---|
| `src/api/` | HTTP server (axum), request/response schema with qev-identical errors |
| `src/frontend/` | Rendering, chat template, tokenization, prefix split, branch planning, answer assembly (CPU) |
| `src/sched/` | Admission queue, token-budget batcher, GPU worker thread, CPU pool, metrics |
| `src/bulk/` | `sysone bulk`: job/result files, packing, coordinator, per-GPU worker processes, pipe protocol |
| `src/contract.rs` | `DecisionBatch` (the packed tree) and the `DecisionEngine` trait |
| `src/engine/` | GPU engine: loader, weights, StepGraph forward, T-bucket CUDA graphs, cuBLAS GEMMs; CPU reference (`cpu_ref/`) |
| `src/kernels/` | cuTile kernels: norms, pointwise, attention, GatedDeltaNet, readout, NVFP4; loader of the SIMT kernels (`simt.rs`) |
| `kernels-simt/` | cuda-oxide SIMT kernels of the GDN delta rule (own toolchain, Rust 1.99.1) and their checked-in PTX; `kernels-simt/build.sh <arch>` rebuilds it (needs cargo-oxide; the main build does not) |
| `src/tolerances.rs`, `tests/tolerances.toml` | Test thresholds derived from measured reference noise |
| `src/main.rs` | The `sysone` server binary (and `sysone bulk ...`) |
| `tests/` | Integration tests and goldens (`tests/golden/dev/`) |
| `benchmarks/s1/` | Benchmark and probe binaries, load generator |
| `reference/` | Python harness that runs the reference (qev + serve.patch) to produce goldens and the eval set |
| `scripts/` | `gpu.sh` (per-GPU lock; `--gpu N` on multi-GPU hosts), `doctor.sh` (prerequisites), `x86/` (runners and the pre-push leak check for the x86 host) |
| `CUDA_RUST_REV` | The cuda-rust commit sysone is built against (checked by `doctor.sh`) |
| `benchmarks/results/x86/` | Sanitised results from the x86 host (the rest of `benchmarks/results/` is git-ignored) |
| `profiles/` | Platform profiles written by `hwprobe --profile-out` |
| `legacy/grout/` | grout's original engine, read-only reference |

## Documentation

| Document | Read it for |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the code works: request lifecycle, packed tree, buckets and graphs, module map, GPU op table, tests, glossary |
| [docs/DESIGN.md](docs/DESIGN.md) | Why: model facts, reference semantics, execution model, correctness strategy, phases |
| [docs/PLATFORM.md](docs/PLATFORM.md) | GB10 vs discrete x86 GPUs, prerequisites |
| [docs/x86/AGENT.md](docs/x86/AGENT.md), [docs/x86/TASKS.md](docs/x86/TASKS.md) | The x86 host (2x RTX PRO 6000 + RTX 5090, sm_120): operating manual for the agent there (setup, GPU discipline, data hygiene, GitHub protocol) and its task list |
| [docs/PHASE0.md](docs/PHASE0.md), [docs/PHASE1.md](docs/PHASE1.md), [docs/PHASE2A.md](docs/PHASE2A.md) | Phase reports and measurements |
| [docs/G1.md](docs/G1.md) | The GDN delta rule in cuda-oxide: results, design, accuracy, the SIMT kernel build and artifact policy |
| [docs/BULK.md](docs/BULK.md) | Bulk tagging: `sysone bulk` usage, packing for throughput, multi-GPU design, numbers |
| [docs/E2.md](docs/E2.md) | The SIMT GDN backend in the engine: selection, every gate on both backends, forward speed |
| [reference/README.md](reference/README.md) | The reference harness and golden formats |
| [reference/eval/README.md](reference/eval/README.md) | The labeled eval set and the statistical gate |
| `cargo doc --no-deps --open` | Per-module API docs |
