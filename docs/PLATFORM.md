# Platforms: DGX Spark (GB10) vs discrete x86 GPUs

sysone has development measurements on DGX Spark (GB10) and a completed public-BF16 Tile serving comparison on
one RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120). B200/H100 are design targets, **not tested by that
comparison**. Start with [Getting started](GETTING_STARTED.md), [Performance](PERFORMANCE.md) and the
[documentation index](README.md).

This file mixes platform design rationale with implemented behavior; it is not a qualification matrix. In
particular, the HTTP server takes budgets/buckets from CLI settings, not an automatic platform-profile policy;
bulk profiles supply a throughput curve, and cross-request prefix caching is not implemented. Treat proposed
per-architecture tuning/fallbacks below as design notes unless backed by a specific report. `scripts/doctor.sh`
checks operational prerequisites (including NVIDIA queries), not CPU-only static documentation correctness.

Sources: measurements in `docs/PHASE0.md`; cuda-rust's DGX Spark tutorial
(`cutile-rs/cutile-book/tutorials/12-dgx-spark-inference.md`); the NVFP4 tutorial (`11-nvfp4-inference.md`);
the cutile-examples requirement table (`cutile-examples/src/requirements.rs`).

## Differences that shape the design

| # | GB10 (DGX Spark) | Discrete x86 GPU | Design response |
|---|---|---|---|
| P1 | **One memory pool:** 128 GB LPDDR5X shared by CPU, OS and GPU. ~243 GB/s measured (273 peak). | Separate VRAM plus host DRAM: **GDDR on the RTX cards**, HBM on datacenter GPUs such as H100/B200; bandwidth/interconnect depend on the card. | Budget against the whole system on GB10: server RSS, tokenizer, page cache and GPU buffers compete. Current server budgets come from CLI settings; a cross-request prefix cache is not implemented. |
| P2 | **Page cache doubles weights.** Reading safetensors via mmap leaves a file-cache copy next to the device copy. Harmless for 16 GB; 2 × 54 GB for mammoth-1 bf16 would crowd 128 GB. | The page-cache copy lives in host RAM, separate from VRAM. | The loader streams each tensor through a reusable staging buffer and drops file pages behind it (`posix_fadvise(DONTNEED)`, or `O_DIRECT`). The same code is correct on x86. Detect integrated memory with `CU_DEVICE_ATTRIBUTE_INTEGRATED`. |
| P3 | **Host pointers are valid on the GPU** (ATS), but readback into fresh pageable memory runs at ~0.13 GB/s because page faults on the destination dominate. | Normal pinned/pageable split; PCIe readback at GB/s. | Device→host results (slot logits, metrics) go to persistent, pre-touched pinned buffers. Never compute into a fresh `Vec`. Don't rely on ATS for correctness (pointer-attribute checks misclassify such pointers). |
| P4 | **Compute/bandwidth crossover is low:** a 57 ms weight floor per forward; compute-bound above ~1k packed tokens; bf16 GEMM ~100 TFLOP/s. | B200: ~8 TB/s, ~2.2 PF bf16, crossover in the hundreds of tokens but a far higher ceiling. H100 differs again. | Measure each GPU separately with `hwprobe`; do not extrapolate GB10 rates to x86. Current HTTP budgets/buckets are CLI-controlled; bulk can use a profile's throughput curve. |
| P5 | **sm_121 (consumer-Blackwell family):** a smaller shared-memory budget per SM than sm_100, and no tcgen05/TMEM path. Tile IR 13.3+ supports block-scaled FP4/FP8 MMA here. PDL is not listed for sm_12x. | sm_100: tcgen05 and larger smem. sm_90: **no FP4** (block-scaled MMA unsupported). | Kernels are cuTile with per-arch tuning records (`tuning/sm_121`, `sm_100`, `sm_90`); never reuse sm_120 (RTX 5090) records. D=256 attention tiles are chosen per arch. NVFP4 needs a fallback on sm_90 (§NVFP4 below). PDL is optional. |
| P6 | **Shared power and thermal envelope** (~140 W SoC): CPU load lowers GPU clocks, and sustained throughput sits below burst. | The GPU has its own power budget; host CPU load barely affects it. | Measure sustained behavior separately with clocks recorded. The completed [serving comparison](PERFORMANCE.md) uses finite bursts, not a sustained-capacity qualification. |
| P7 | **CPU:** 20 Arm cores (performance + efficiency). The desktop session shares the GPU. | Many x86 cores; usually headless. | Pin the GPU worker and tokenizer pool to performance cores. Benchmark with the desktop idle; record it in run metadata. |
| P8 | **aarch64 software gaps:** no pinned `torch==2.8.0+cu128` wheel; fla/causal-conv1d build from source; DGX OS ships CUDA 13.0 and libclang without resource headers. | The reference image runs as-is (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`). | Two golden tiers (DESIGN §9.1): dev goldens on GB10 (NGC `pytorch:26.10-py3` arm64, or a venv with an aarch64 CUDA torch), authoritative goldens on x86 with the pinned image. Prerequisites are installed, never patched around. |
| P9 | **Driver R580** (CUDA 13.0 driver API) on DGX OS with toolkit 13.4. Tile IR emits native sm_121 code, so no newer driver is needed. | Varies; usually newer. | R580 is the minimum supported driver. Driver-API bindings (cudarc `cuda-13000`) target 13.0; the engine uses cuda-core for contexts and streams. |
| P10 | **Single GPU,** no MIG, no NVLink peers. | Possibly several GPUs. | v1 is one engine per GPU. Multi-GPU means one process per GPU; no tensor parallelism. `sysone bulk` does this: one worker process per GPU (started through `scripts/gpu.sh --gpu N`) fed from one queue, each card sized by its own memory (docs/BULK.md). GPU jobs lock and check only their own GPU (`scripts/gpu.sh --gpu N`, per-GPU lock files). |

## The x86 host

The original x86 workstation collaboration is documented in the unchanged [agent manual](x86/AGENT.md) and
[task list](x86/TASKS.md). These are historical host-specific operating/collaboration material, **not new-user
prerequisites**; dated never-run statements there are not the current BF16/Tile qualification status.

The tested x86 path is **BF16 + Tile**: the [exact-mode guide](x86/README.md) separates the pre-fix synthetic
state-isolation test from the completed post-fix optimized-qev HTTP comparison at fbddc43. The latter measured
one RTX PRO 6000 Blackwell Max-Q Workstation Edition, not every card in the workstation, H100/B200 or the sm_120
SIMT/NVFP4 paths. A checked-in SIMT artifact alone does not establish runtime qualification.

`sysone bulk` implements one worker process per selected GPU from a shared queue; there is no tensor parallelism.
Each process takes its own lock via `scripts/gpu.sh --gpu N`; `require_idle_gpu` checks only the selected GPU.
Keep cuda-rust at the clean revision in `CUDA_RUST_REV`. See [Bulk](BULK.md) for the coordinator and per-card sizing.

## Toolkit upgrades

cuda-bindings' and cutile-compiler's build scripts detect the toolkit once and re-run only when an environment
variable changes, not when `/usr/local/cuda-13` is repointed to a new version. After a toolkit upgrade, run
`cargo clean -p cutile-compiler -p cuda-bindings`; otherwise cutile keeps the old version's capabilities (seen
here: "toolkit is CUDA 13.0" warnings after installing 13.4).

## NVFP4 per platform

| Target | Path |
|---|---|
| sm_121 (GB10), sm_120 | Native block-scaled FP4 MMA (`mmaf_scaled`, Tile IR ≥ 13.3). |
| sm_100 / sm_103 (B200/B300) | Native block-scaled FP4 MMA; separate tuning. |
| sm_90 (H100) | No FP4 tensor cores. The proposed load-time bf16 dequantization / activation-QDQ fallback is **not implemented** (see [N1](N1.md)); do not treat it as a supported NVFP4 path. |

## Prerequisites

| | GB10 / DGX OS (aarch64) | x86_64 Ubuntu 24.04 |
|---|---|---|
| Driver | R580+ (ships with DGX OS) | R580+ |
| CUDA | `sudo apt-get install -y cuda-toolkit-13-4 cuda-tileiras-13-4` (DGX OS ships 13.0) | same packages |
| bindgen | `sudo apt-get install -y clang libclang-dev cmake build-essential pkg-config` | same |
| Rust | rustup; the toolchain (1.98.0) comes from `rust-toolchain.toml` | same |
| SIMT kernels (cuda-oxide, when needed) | LLVM 21+ via `https://apt.llvm.org/llvm.sh 21` (not in Ubuntu 24.04), Rust 1.99.1 with `rustc-dev`/`llvm-tools` (from cuda-oxide's pin) | same |
| Reference / vLLM (developer-only) | Historical reference containers: NGC arm64 images (`nvcr.io/nvidia/pytorch:26.10-py3`, `nvcr.io/nvidia/vllm:26.10-py3`) | Historical pinned upstream reference image; not needed to serve public BF16 |
| Model download | `hf` CLI; [public pinned download](GETTING_STARTED.md#download-the-public-tested-checkpoint-and-build) needs no private-checkpoint access | same |
