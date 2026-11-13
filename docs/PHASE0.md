# Phase 0 report (2026-11-13)

Box: NVIDIA GB10 (DGX Spark), sm_121, driver 580.178.04 (CUDA 13.0 driver API), Ubuntu 24.04 aarch64.

## Done

- **Fork restructured.** grout's Qwen3 decode engine, tests and benchmarks moved to `legacy/grout/`. They are not
  built; they are kept as porting reference for W3/W4. A new `sysone` crate depends on `../cuda-rust` by path:
  `cuda-core` and `cuda-async` 0.5.0, `cutile` and `cutile-kernels` 0.6.0. It also keeps `cudarc` 0.19 for the
  cuBLAS baseline.
- **Contract in code.** `src/contract.rs` defines `Segment`, `DecisionBatch` (with `push_request` and `validate`
  enforcing the §6.1 invariants), `SlotLogits`, the `DecisionEngine` trait, and a batch-invariant `MockEngine` for W5.
- **Config.** `src/engine/config.rs` parses `config.json` and refuses anything v1 cannot serve: a non-`slot_lm`
  readout, a causal full-attention mask, attention bias, tied embeddings and so on.
- **Module layout** per DESIGN §12, with owner notes. `cargo test --lib`: 5 passed.
- **`hwprobe`** (`benchmarks/s1/hwprobe.rs`) measures bandwidth and bf16 GEMM throughput.

## Environment findings

Status as of the end of Phase 0. `scripts/doctor.sh` re-checks all of this; `docs/PLATFORM.md` has the per-platform
prerequisites.

| Item | Finding | Resolution |
|---|---|---|
| Rust | Was not installed | rustup in `~/.cargo`; toolchain 1.98.0 from `rust-toolchain.toml` (matches cuda-rust) |
| CUDA toolkit | DGX OS ships 13.0; cuTile needs ≥ 13.2 (≥ 13.3 for FP4 MMA) | **done:** 13.4 + tileiras installed (`/usr/local/cuda-13` → 13.4) |
| Driver | R580 = CUDA 13.0 driver API | Supported as-is: Tile IR emits native sm_121 code (Spark tutorial). cudarc pinned to `cuda-13000` because it binds driver symbols eagerly; documented in `Cargo.toml` |
| bindgen headers | `libclang1-18` without clang resource headers, so `stddef.h` is missing and cuda-bindings fails to build (cuda-rust itself too) | Missing prerequisite, not a sysone issue. **user:** `sudo apt-get install -y clang libclang-dev cmake build-essential pkg-config`. The include-path workaround was removed |
| LLVM 21+ (SIMT, cuda-oxide) | Not packaged for Ubuntu 24.04 | Needed only when SIMT kernels start: `apt.llvm.org/llvm.sh 21` |
| Docker | User added to group `docker`; not active in the current login session | Log out and back in |
| NGC images | `nvcr.io/nvidia/pytorch:26.10-py3` and `nvcr.io/nvidia/vllm:26.10-py3` have arm64 manifests | W1 / W6 |
| Reference stack on aarch64 | No pinned `torch==2.8.0+cu128` aarch64 wheel; the NVFP4 checkpoint was evaluated with torch 2.13.0+cu132 | W1 dev tier (DESIGN §9.1) |

## Measurements (cuBLAS bf16, fp32 accumulate, random data)

- Device memory bandwidth (d2d copy): **243 GB/s**.
- elephant-1 GEMM weights: 13.84 GB, so the **weight-streaming floor is 57 ms per forward**.
- bf16 GEMM: **~92–101 TFLOP/s** at T ≥ 4k on the large projections, and ~85–105 at T = 1k. `in_proj_ab` (n=64)
  manages only 14–31 TFLOP/s, so fuse it into the `in_proj_qkv`/`in_proj_z` GEMM (n = 8192+4096+64).
- GEMM-only forward estimate (all 32 layers, distinct weights):

| T (packed tokens) | ms | TFLOP/s | tokens/s |
|---:|---:|---:|---:|
| 256 | 90.8 | 39.0 | 2,821 |
| 1,024 | 154.5 | 91.7 | 6,629 |
| 4,096 | 625.4 | 90.6 | 6,550 |
| 8,192 | 1,184.1 | 95.7 | 6,919 |
| 16,384 | 2,320.1 | 97.7 | 7,062 |

Rerun with `cargo run --release --bin hwprobe`.

## What this changes

1. **GB10 becomes compute-bound at roughly 1k tokens per forward.** The GEMM ceiling is about **7,000 tokens/s in
   bf16**. Attention, GDN and norms add on top, so a realistic bf16 end-to-end ceiling is about 5,500–6,500 tok/s.
   Batching across requests still matters, but only below about 1k packed tokens, where the 57 ms weight-streaming
   floor dominates. Above that, throughput is set by FLOPs per decision.
2. **The multiple over the reference depends on request shape.**
   - Short states at high concurrency: the reference pays 2 weight-bound forwards per request plus Python
     overhead, while we pack about 30 requests per forward. Expect several times the throughput.
   - Long states (≥ 4k tokens): both sides are compute-bound on the same GEMMs. Our edge is only the reference's
     waste: padding, fork copies, the full-vocab LM head, and sequential fallbacks. Expect roughly 1.2–1.5×.

   The flat "≥ 5× the reference" target is therefore replaced (DESIGN §10).
3. **vs vLLM:** per-question requests recompute or re-read the state for every question unless prefix caching
   works for this hybrid model. Our shared prefix gives up to Q× fewer FLOPs, so the ≥ 2× target stands.
4. **Precision is the big remaining lever on long states.** NVFP4 was moved up (DESIGN §1, §3.1), using the
   existing W4A4-MLP checkpoint. The MLP is ≈ 70% of GEMM FLOPs, so the end-to-end gain depends on the measured
   FP4/bf16 GEMM ratio on sm_121.

## Remaining Phase 0 items

- After the clang install: run a cutile bf16 GEMM bench and the NVFP4 example on sm_121, and add Tile bf16 and
  FP4 rates to `hwprobe`. That gives the Tile-vs-cuBLAS baseline and the measured FP4 speedup `r` (DESIGN §13 Q8).
- After re-login: W6 checks that the NGC vLLM image serves Qwen3.5.
