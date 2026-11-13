# Phase 1 report (2026-11-13)

Nine workstreams ran as two workflows (CPU/Python track W1a, W1b, W2, W5; GPU track W3a, W3b, W3c, W4a, W4b).
Each was implemented on its own branch, reviewed independently, fixed where the review blocked (W1b only), and
merged into `main`. The full test suite passes on the merged tree (`cargo test --release --all-targets`: 59 unit
tests and 39 integration tests; the two CPU-reference tests are `#[ignore]`d because they need the checkpoint and
minutes of CPU, and were reproduced by the W4b reviewer).

## What exists now

| Workstream | Result |
|---|---|
| W1a render goldens | `reference/` harness around qev (OneJev@81ce62f + `serve.patch`) driven through qev's own FastAPI app; 298-case corpus in 14 groups; render goldens in `tests/golden/dev/render/` (byte-identical on rerun). |
| W1b model goldens | 52-request interim corpus run in fp32, bf16 and NVFP4 (QDQ) over batched, sequential and per-question paths; hidden-state dumps (70 files in `sysone-ref/golden/dev`, sha256 manifest committed); `tests/tolerances.toml` derived from measured noise, with a consistency test (`src/tolerances.rs`). `reference/layer_dump.py` dumps per-layer states for any request. |
| W2 frontend | Schema validation, rendering, chat template, tokenization, prefix split, planning, assembly, calibration. Matches qev on 100% of the 298 cases (11,960 field checks), including pydantic's exact 422 bodies. Plan p50 0.35 ms per request. |
| W5 server | axum server (`/v1/systemone`, `:batch`, `/health`, `/v1/models`, `/metrics`), bounded admission queue, token-budget batcher, GPU worker thread, `loadgen`. Runs on `MockEngine`. |
| W3a GDN | `tree_conv1d_silu`, `gdn_prep`, chunked scan (prefix and suffix phases, suffixes start from the parent's state), `gated_rmsnorm`, `GdnPlan` for graph replay. Within 2× of transformers' own f32 error vs an f64 oracle; bit-identical alone vs packed. |
| W3b attention and norms | `seg_attn` (D=256, prefix/suffix ranges, GQA, sigmoid gate), zero-centred norms, partial RoPE, pointwise ops, slot readout. P kept in fp32 (3-part split) because the reference's sdpa falls back to the fp32 MATH backend at D=256. |
| W3c NVFP4 | Activation quantization bit-exact vs compressed-tensors; block-scaled FP4 GEMMs within 0.5 ulp of the exact dequant-matmul, with grouped rasterization. |
| W4a engine foundations | Loader (O_DIRECT, no page-cache copy: bf16 14.8 GiB in 3.3 s), weights incl. NVFP4, cuBLAS bf16 GEMM, StepGraph arena, double-buffered batch metadata, platform profile. |
| W4b CPU reference | f32 CPU forward over packed trees. One packed pass reproduces qev's fp32 sequential results: max \|Δp\| 4.7e-7, smaller than qev's own batched-vs-sequential spread (8.5e-7). This validates the design's central claim before any GPU forward exists. |

## Idle-GPU measurements (2026-11-13, after Phase 1)

Phase 1's own speed numbers were taken while reference jobs used the GPU without the lock, so they were rerun on
an idle GPU (`scripts/gpu.sh`, benchmarks refuse to start on a shared GPU). Raw output: `benchmarks/results/phase1-idle/`
(not committed); platform profile: `profiles/sm_121-gb10.json`.

- Bandwidth 221 GB/s (d2d copy, 1 GiB); cuBLAS bf16 GEMM-only forward unchanged from Phase 0 (150 ms at T=1k,
  624 ms at 4k, 2,320 ms at 16k; ~6,800–7,060 tok/s).
- **GDN mixer, per layer** (unchanged by contention: the problem is real):

| T | mixer ms | ×24 layers | share of (GEMM + GDN) |
|---:|---:|---:|---:|
| 1,024 | 3.9 | 95 ms | 38% |
| 4,096 | 15.1 | 362 ms | 37% |
| 16,384 | 58.0 | 1,391 ms | 38% |

  The sequential scan (`scan_p` 18–25 ms at 16k) and the WY step (16 ms) dominate.
- **seg_attn, per layer**: 5.0 ms at 4k (36 TFLOP/s), 78 ms at 16k (37 TFLOP/s) with the default tiles;
  `bm8/bn32/p_split=1` gives 4.3 / 65 ms (42–44 TFLOP/s). cutile-kernels' dense f16 fmha at D=256 reaches
  52–54 TFLOP/s on the same GPU. Over 8 layers at 16k: ~0.63 s against ~2.3 s of GEMM.
- **NVFP4 MLP** (forward GEMM time, 32 layers): MLP GEMMs 2.5–2.85× faster than cuBLAS bf16; whole forward GEMM
  time 1.70× faster at 4k with a standalone quantize kernel, 1.83× with the quantize fused (625 → 367 ms).
- Pointwise kernels run at 160–290 GB/s (bandwidth bound).

Estimated bf16 forward at T=4k today: 624 (GEMM) + 362 (GDN) + 40 (attention) + ~30 (norms, pointwise) ≈ 1.06 s,
about 3,900 tok/s. GDN is the first lever, NVFP4 the second.

## Open issues carried into Phase 2

1. **GDN speed** (above). cuTile's f32 `mma` is emulated off the tensor cores, so W3a runs 3×TF32 splits; the
   scan spills and runs one CTA per SM. Plan: blocked 16×16 triangular solve (as fla's `solve_tril`), recompute
   instead of storing ~72 KB/token of fp32 intermediates, per-phase block size, precision variants chosen by the
   end-to-end gate, and a SIMT scan once cuda-oxide is usable (needs LLVM 21).
2. **Attention at D=256**: register pressure ([64,256] fp32 accumulator); plan a D-split accumulator and 2 CTAs/SM.
3. **NVFP4 reference instability**: the QDQ reference moves by up to 0.2 per option under a batch-shape change, so
   the NVFP4 model gate is statistical (DESIGN §3.1, §9.2).
4. **Reference engine choice**: match the text `DecisionEngine` with the block-bidirectional mask, not
   `MMDecisionEngine` (DESIGN §1).
5. **Server details** from the W5 review: `forward_s` includes queue time (qev's covers only the forward); an empty
   `x-typesafe-request-id` is echoed instead of replaced; no backpressure before admission; threads not pinned to
   performance cores; `debug=1` lacks `allowed_token_mass` (needs an engine-side full-vocabulary logsumexp, only
   for debug requests).
6. **Graph replay hazard** (W3a review): `GdnPlan::update` does not check scratch slots; size `GdnScratch` for the
   plan's maximum segment count.
7. **cuTile issues** to report upstream (the user decides whether to file): emulated f32 `mma`; a `[1,1]` f32 tile
   store corrupting neighbours; two JIT type-inference failures (inline shape literals, nested `convert_tile`).
8. The NVFP4 layer-0 test data (`reference/nvfp4/data`) is generated, not committed; the test now fails instead of
   passing vacuously when it is missing.
9. Model goldens cover the 52-request interim corpus; the 298-case corpus has render goldens only.
