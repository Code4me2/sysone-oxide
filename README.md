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
- A faster GatedDeltaNet scan on cuda-oxide SIMT kernels (G1) is in progress.

## Prerequisites

- `../cuda-rust` checked out next to this repository (the crates are path dependencies).
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

# Serve with the deterministic mock engine (real frontend and scheduler, fake logits; no GPU needed).
./target/release/sysone --model /path/to/elephant-1 --engine mock

# Load test a running server.
./target/release/loadgen --addr 127.0.0.1:8000 --concurrency 64 --state-tokens 1000 --questions 8 --duration-s 60

# Benchmarks refuse to start while another process uses the GPU.
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
| `src/contract.rs` | `DecisionBatch` (the packed tree) and the `DecisionEngine` trait |
| `src/engine/` | GPU engine: loader, weights, StepGraph forward, T-bucket CUDA graphs, cuBLAS GEMMs; CPU reference (`cpu_ref/`) |
| `src/kernels/` | cuTile kernels: norms, pointwise, attention, GatedDeltaNet, readout, NVFP4 |
| `src/tolerances.rs`, `tests/tolerances.toml` | Test thresholds derived from measured reference noise |
| `src/main.rs` | The `sysone` server binary |
| `tests/` | Integration tests and goldens (`tests/golden/dev/`) |
| `benchmarks/s1/` | Benchmark and probe binaries, load generator |
| `reference/` | Python harness that runs the reference (qev + serve.patch) to produce goldens and the eval set |
| `scripts/` | `gpu.sh` (GPU lock), `doctor.sh` (prerequisites) |
| `profiles/` | Platform profiles written by `hwprobe --profile-out` |
| `legacy/grout/` | grout's original engine, read-only reference |

## Documentation

| Document | Read it for |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the code works: request lifecycle, packed tree, buckets and graphs, module map, GPU op table, tests, glossary |
| [docs/DESIGN.md](docs/DESIGN.md) | Why: model facts, reference semantics, execution model, correctness strategy, phases |
| [docs/PLATFORM.md](docs/PLATFORM.md) | GB10 vs discrete x86 GPUs, prerequisites |
| [docs/PHASE0.md](docs/PHASE0.md), [docs/PHASE1.md](docs/PHASE1.md), [docs/PHASE2A.md](docs/PHASE2A.md) | Phase reports and measurements |
| [reference/README.md](reference/README.md) | The reference harness and golden formats |
| [reference/eval/README.md](reference/eval/README.md) | The labeled eval set and the statistical gate |
| `cargo doc --no-deps --open` | Per-module API docs |
