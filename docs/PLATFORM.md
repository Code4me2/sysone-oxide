# Platforms: DGX Spark (GB10) vs discrete x86 GPUs

sysone is developed on a DGX Spark and must also run on x86 hosts with discrete GPUs (the reference x86 box,
later B200/H100). The two differ in ways that change engine behaviour, not just tuning. This file lists the
differences, what the design does about each, and the prerequisites per platform. `scripts/doctor.sh` checks the
prerequisites.

Sources: measurements in `docs/PHASE0.md`; cuda-rust's DGX Spark tutorial
(`cutile-rs/cutile-book/tutorials/12-dgx-spark-inference.md`); the NVFP4 tutorial (`11-nvfp4-inference.md`);
the cutile-examples requirement table (`cutile-examples/src/requirements.rs`).

## Differences that shape the design

| # | GB10 (DGX Spark) | Discrete x86 GPU | Design response |
|---|---|---|---|
| P1 | **One memory pool:** 128 GB LPDDR5X shared by CPU, OS and GPU. ~243 GB/s measured (273 peak). | Separate HBM (3–8 TB/s) plus host DRAM over PCIe or NVLink-C2C. | All memory budgets (§6.5) count against the whole system on GB10: server RSS, tokenizer, page cache and GPU buffers compete. The token budget and prefix cache size come from a platform profile, not constants. |
| P2 | **Page cache doubles weights.** Reading safetensors via mmap leaves a file-cache copy next to the device copy. Harmless for 16 GB; 2 × 54 GB for mammoth-1 bf16 would crowd 128 GB. | The page-cache copy lives in host RAM, separate from VRAM. | The loader streams each tensor through a reusable staging buffer and drops file pages behind it (`posix_fadvise(DONTNEED)`, or `O_DIRECT`). The same code is correct on x86. Detect integrated memory with `CU_DEVICE_ATTRIBUTE_INTEGRATED`. |
| P3 | **Host pointers are valid on the GPU** (ATS), but readback into fresh pageable memory runs at ~0.13 GB/s because page faults on the destination dominate. | Normal pinned/pageable split; PCIe readback at GB/s. | Device→host results (slot logits, metrics) go to persistent, pre-touched pinned buffers. Never compute into a fresh `Vec`. Don't rely on ATS for correctness (pointer-attribute checks misclassify such pointers). |
| P4 | **Compute/bandwidth crossover is low:** a 57 ms weight floor per forward; compute-bound above ~1k packed tokens; bf16 GEMM ~100 TFLOP/s. | B200: ~8 TB/s, ~2.2 PF bf16, crossover in the hundreds of tokens but a far higher ceiling. H100 differs again. | Token-budget, max-wait and bucket sizes come from `hwprobe` per machine (§7), saved as a platform profile. GB10 results are never extrapolated to x86. |
| P5 | **sm_121 (consumer-Blackwell family):** a smaller shared-memory budget per SM than sm_100, and no tcgen05/TMEM path. Tile IR 13.3+ supports block-scaled FP4/FP8 MMA here. PDL is not listed for sm_12x. | sm_100: tcgen05 and larger smem. sm_90: **no FP4** (block-scaled MMA unsupported). | Kernels are cuTile with per-arch tuning records (`tuning/sm_121`, `sm_100`, `sm_90`); never reuse sm_120 (RTX 5090) records. D=256 attention tiles are chosen per arch. NVFP4 needs a fallback on sm_90 (§NVFP4 below). PDL is optional. |
| P6 | **Shared power and thermal envelope** (~140 W SoC): CPU load lowers GPU clocks, and sustained throughput sits below burst. | The GPU has its own power budget; host CPU load barely affects it. | Benchmarks report sustained steady-state numbers (≥ 60 s per point) with GPU clocks logged. Tokenization stays lean and off the critical path. |
| P7 | **CPU:** 20 Arm cores (performance + efficiency). The desktop session shares the GPU. | Many x86 cores; usually headless. | Pin the GPU worker and tokenizer pool to performance cores. Benchmark with the desktop idle; record it in run metadata. |
| P8 | **aarch64 software gaps:** no pinned `torch==2.8.0+cu128` wheel; fla/causal-conv1d build from source; DGX OS ships CUDA 13.0 and libclang without resource headers. | The reference image runs as-is (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`). | Two golden tiers (DESIGN §9.1): dev goldens on GB10 (NGC `pytorch:26.10-py3` arm64, or a venv with an aarch64 CUDA torch), authoritative goldens on x86 with the pinned image. Prerequisites are installed, never patched around. |
| P9 | **Driver R580** (CUDA 13.0 driver API) on DGX OS with toolkit 13.4. Tile IR emits native sm_121 code, so no newer driver is needed. | Varies; usually newer. | R580 is the minimum supported driver. Driver-API bindings (cudarc `cuda-13000`) target 13.0; the engine uses cuda-core for contexts and streams. |
| P10 | **Single GPU,** no MIG, no NVLink peers. | Possibly several GPUs. | v1 is one engine per GPU. Multi-GPU means one process per GPU behind a router; no tensor parallelism. GPU jobs lock and check only their own GPU (`scripts/gpu.sh --gpu N`, per-GPU lock files). |

## The x86 host

The user's x86 workstation (2x RTX PRO 6000 Blackwell 96 GB and 1x RTX 5090 32 GB, all sm_120) is where throughput
for bulk tagging is measured and tuned. It holds sensitive material, so the GB10 side has no access to it: a
separate Claude Code agent works there and collaborates only through GitHub issues and PRs.
[docs/x86/AGENT.md](x86/AGENT.md) is its operating manual (setup, per-GPU lock discipline, data hygiene, the
GitHub protocol) and [docs/x86/TASKS.md](x86/TASKS.md) its task list. Multi-GPU support so far is the discipline
(P10: one process per GPU): `scripts/gpu.sh --gpu N` takes a per-GPU lock and `require_idle_gpu` checks only its
own GPU. sysone is pinned to one cuda-rust revision (`CUDA_RUST_REV`, checked by `scripts/doctor.sh`) so both
machines build the same code. The SIMT GDN artifact for sm_120 is checked in but has not run yet.

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
| sm_90 (H100) | No FP4 tensor cores. Fallback: dequantize to bf16 at load and run the bf16 GEMM. This gives W4A4 *QDQ numerics* (activations still fake-quantized) at bf16 speed. It exists for correctness parity, not speed. |

## Prerequisites

| | GB10 / DGX OS (aarch64) | x86_64 Ubuntu 24.04 |
|---|---|---|
| Driver | R580+ (ships with DGX OS) | R580+ |
| CUDA | `sudo apt-get install -y cuda-toolkit-13-4 cuda-tileiras-13-4` (DGX OS ships 13.0) | same packages |
| bindgen | `sudo apt-get install -y clang libclang-dev cmake build-essential pkg-config` | same |
| Rust | rustup; the toolchain (1.98.0) comes from `rust-toolchain.toml` | same |
| SIMT kernels (cuda-oxide, when needed) | LLVM 21+ via `https://apt.llvm.org/llvm.sh 21` (not in Ubuntu 24.04), Rust 1.99.1 with `rustc-dev`/`llvm-tools` (from cuda-oxide's pin) | same |
| Reference / vLLM | Docker with NGC arm64 images (`nvcr.io/nvidia/pytorch:26.10-py3`, `nvcr.io/nvidia/vllm:26.10-py3`); user in group `docker` | the pinned upstream image (`setup.sh`); vLLM image |
| HF | `hf auth login` (private NVFP4 checkpoint) | same |
