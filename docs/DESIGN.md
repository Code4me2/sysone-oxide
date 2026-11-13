# sysone: design

A Rust inference server for Base Labs' System One decision models (`baselabs/elephant-1`, later `mammoth-1`),
forked from grout and built on `../cuda-rust` (cutile-rs, cuda-async, cuda-oxide). It is a drop-in replacement for
OneJev's `qev` server (`POST /v1/systemone`). Its goal is many more decisions per second than that reference or a
vLLM-based setup, while matching the reference's probabilities.

Status: **accepted (v1 baseline), 2026-11-13**. Section 13 records the resolved decisions and the remaining risks.
This document is the design rationale; for how the code is actually organised today, see
[ARCHITECTURE.md](ARCHITECTURE.md). Notes marked *Update* below correct statements that the implementation has
since overtaken.

## 1. Decisions taken

| Topic | Decision |
|---|---|
| Hardware | Develop and benchmark on GB10 (DGX Spark, sm_121, 128 GB unified memory). Kernels stay arch-agnostic (cuTile + per-arch tuning records), so sm_100/sm_90 are a tuning task, not a redesign. Platform differences that change engine behaviour are listed in `docs/PLATFORM.md` (P1–P10). |
| v1 scope | Text only (requests containing images get a 422), `slot_lm` readout. Two weight formats: **bf16** (`baselabs/elephant-1`) and **NVFP4 W4A4 on the MLP projections** (`Code4me2/elephant-1-NVFP4-MLP`, compressed-tensors, §3.1). NVFP4 was moved up from Phase 3 (2026-11-13). Vision and the learned `head` readout come in later phases. |
| Code home | This repo: a standalone fork of grout. It depends on cuda-rust by path and does not live inside it. |
| Dev vs validation | All development and CI-style testing happen on this GB10 (aarch64). An x86 box runs the pinned reference image for the authoritative goldens and a final cross-arch validation. |
| Length limits | Match the reference defaults (32,768 per branch, 65,536 per request), configurable by CLI flag. |
| API | Drop-in `/v1/systemone` with the same request and response schema, `debias`/`debug` query params, and error codes as qev at OneJev@81ce62f + `serve.patch`. Extensions are additive. |
| Reference engine (2026-11-13) | Match qev's text `DecisionEngine` with `serve.patch`'s block-bidirectional mask. `qev serve` routes configs that have a `vision_config` (elephant-1 does) to `MMDecisionEngine`, which `serve.patch` does not patch and which has no block-bidirectional mask; we treat that as a reference defect, not behaviour to copy. The config's `block_bidirectional` override and the model card's `DecisionEngine` usage support this. |
| GPU discipline (2026-11-13) | Every GPU job (tests, reference/golden runs, benchmarks) runs through `scripts/gpu.sh`, which holds `/tmp/sysone-gpu.lock`. Benchmarks also refuse to start while another process holds a CUDA context (`--allow-shared-gpu` overrides and marks the output). Phase 1 lost its speed numbers to unlocked reference jobs. |
| Precision knobs (2026-11-13) | Where a kernel can trade fp32-grade internal precision for speed (GDN scan TF32/bf16 operands, attention P split), the end-to-end gate (§9.2) decides, not a per-kernel bit-level match to transformers' fp32 fallback. Defaults stay at the most precise variant until the end-to-end gate has been run with the faster one. |

## 2. Goals and non-goals

Goals
1. **Exactness:** reproduce the reference's math: prompt bytes, token ids, prefix/suffix split, block-bidirectional
   mask, slot readout, calibration, option chunking and debias averaging.
2. **Throughput:** batch across requests so each forward is a large, compute-bound packed pass; zero cache copies.
3. **Latency:** p50 at low load is no worse than the reference, and p99 is bounded under load by a token-budget scheduler.

Non-goals (v1): text generation or decode, training, LoRA, multi-GPU, images.

## 3. Verified model facts (elephant-1)

Checked against `config.json`, the safetensors header, and `transformers==5.19.0` `modeling_qwen3_5.py`.

- `Qwen3_5ForConditionalGeneration`; the text model is 32 layers with `layer_types` = `[linear, linear, linear, full] × 8`
  (24 GatedDeltaNet layers, 8 full-attention layers). hidden 4096, MLP 12288 (SwiGLU), vocab 248,320, untied
  `lm_head`, `rms_norm_eps` 1e-6. All weights are bf16, **including `A_log` and `dt_bias`**. No MTP weights are shipped.
- Parameters: language model 7.94 B (of which `embed_tokens` is 1.02 B), `lm_head` 1.02 B, vision tower 0.46 B. The
  per-token GEMM weights stream **≈ 6.9 B params ≈ 13.8 GB per forward** (layers only; embedding is a gather, and
  the LM head is reduced to slot rows, see §6.6).
- Weight names: `model.language_model.layers.{i}.{input_layernorm,post_attention_layernorm,mlp.{gate,up,down}_proj}`,
  plus per-type mixer weights:
  - linear: `linear_attn.{in_proj_qkv [8192,4096], in_proj_z [4096,4096], in_proj_b [32,4096], in_proj_a [32,4096],
    conv1d.weight [8192,1,4], A_log [32], dt_bias [32], norm.weight [128], out_proj [4096,4096]}`
  - full: `self_attn.{q_proj [8192,4096], k_proj [1024,4096], v_proj [1024,4096], o_proj [4096,4096],
    q_norm [256], k_norm [256]}`
  - `model.language_model.{embed_tokens,norm}`, `lm_head.weight`; `model.visual.*` is ignored in v1.

Numerics to replicate exactly:
- **RMSNorm is zero-centred:** `y = (x_f32 * rsqrt(mean(x²)+eps)) * (1 + w_f32)`, cast to bf16 at the end. This
  applies to the layer norms, the final norm and `q_norm`/`k_norm`. grout's `rms_norm_f16` uses a plain `w`: it must change.
- **Gated RMSNorm (GDN output):** fp32 normalisation, cast to bf16, multiply by `w` (plain, not `1+w`) in bf16, then
  multiply by `silu(z_f32)` and cast to bf16. Applied per 128-wide v-head.
- **Full attention:** `q_proj` output is viewed `[heads=16, 2*256]` and split per head into query `[..., :256]` and
  gate `[..., 256:]`. Then:
  - q_norm/k_norm, then RoPE on the first 64 dims only (`partial_rotary_factor` 0.25, θ=1e7, rotate-half style,
    32 frequencies).
  - GQA 16q/4kv, `head_dim` 256, scale 1/16.
  - Multiply the output by `sigmoid(gate)`, then `o_proj`.
  - For text-only inputs the 3 MRoPE position streams are equal, so MRoPE reduces exactly to 1-D RoPE (§13 Q4
    covers images).
- **GatedDeltaNet (per layer):**
  1. `qkv = in_proj_qkv(x)`, a causal depthwise conv1d (kernel 4, no bias) over the 8192 channels, then SiLU.
  2. Split into q `[16×128]`, k `[16×128]`, v `[32×128]`.
  3. `beta = sigmoid(in_proj_b x)`; `g = -exp(A_log_f32) * softplus(in_proj_a x + dt_bias)` (fp32).
  4. q and k are repeat-interleaved ×2 to 32 heads (v-head `h` uses k-head `h/2`), then L2-normalised in fp32
     (eps 1e-6). q is scaled by 128^-0.5.
  5. Chunked gated delta rule in fp32 with chunk size 64 and recurrent state `[32, 128(k), 128(v)]` fp32.
  6. Output: gated RMSNorm with `z = in_proj_z(x)`, then `out_proj`.
- **The mask:** for full-attention layers it is `causal OR (block[q]==block[kv] AND block[q]>=0)`, with block ids
  0 = shared prefix, 1 = question suffix, -1 = padding. So:
  - prefix tokens attend to the whole prefix;
  - suffix tokens attend to the whole prefix plus their whole own suffix;
  - linear-attention layers stay causal.

### 3.1 NVFP4 checkpoint (`Code4me2/elephant-1-NVFP4-MLP`, private, experimental)

Checked against its `config.json`, `recipe.yaml`, `provenance.json` and the safetensors header (2026-11-13).

- **What is quantized:** exactly the 96 MLP projections (`gate/up/down_proj` × 32 layers), as W4A4 NVFP4
  (compressed-tensors 0.20.1, `nvfp4-pack-quantized`). Everything else is byte-identical bf16: attention and GDN
  projections, norms, embeddings and `lm_head`.
- **MLP share of the model:** 4.83 B of the 6.92 B GEMM parameters (≈ 70% of GEMM FLOPs). Streamed weight bytes
  per forward drop from 13.8 GB to ≈ 6.9 GB.
- **Tensors per projection:**
  - `weight_packed` u8 `[N, K/2]` (two e2m1 values per byte);
  - `weight_scale` f8e4m3 `[N, K/16]`;
  - `weight_global_scale` f32 `[1]`;
  - `input_global_scale` f32 `[1]`.
- **Weights:** RTN quantization, group 16 along K, with an FP8 block scale per group and a per-tensor global scale.
- **Activations:** `dynamic: local`, `tensor_group` 16. Each token row's group of 16 gets its own FP8 scale at
  run time, under a *static* per-tensor `input_global_scale` (calibrated min-max).
  - There is no cross-token statistic, so per-row quantization keeps the shared-prefix forward equivalent to the
    per-question forward the checkpoint was evaluated with **in exact arithmetic**. In floating point the QDQ
    reference is very sensitive to kernel shape: changing only the batch size moves its option probabilities by up
    to 0.2 (W1b, confirmed by its reviewer), and its per-question vs shared-prefix paths differ by up to 0.23. NVFP4
    is therefore gated statistically (§9.2), not per branch.
  - `gate_proj.input_global_scale == up_proj.input_global_scale` in all 32 layers (verified), so the normed MLP
    input is quantized **once** for the fused gate/up GEMM. The loader asserts this equality and refuses the
    checkpoint otherwise.
- **Numerics to match** are compressed-tensors 0.20.1's quantize/dequantize (scale computation, FP8 cast, e2m1
  rounding, global-scale division). W3 ports them from that source and tests them bit-exactly against it.
- **The checkpoint's own evaluation** was QDQ simulation (`run_compressed=False`): weights materialized to bf16,
  activation fake-quant, batch 1, no cache, one full forward per question, TF32 off. That simulation is our
  NVFP4 reference (§9), *not* bf16. Its documented behaviour:
  - cross-process probability drift up to 5.8 pp, and 41 pp max vs bf16 on one question;
  - calibration data up to 16,017 tokens, with a 16,384-token documented limit.
- The checkpoint is composite. Vision tensors sit in `model_mtp.safetensors` despite the name, and v1 ignores
  them; `model_backbone.safetensors` holds the text model. The loader works from `model.safetensors.index.json`.

## 4. Reference semantics to reproduce

Source: OneJev `qev/{prompt,engine,schema,answers,calibrate,server}.py` @81ce62f + elephant-1's `serve.patch`.

1. **Rendering**
   - The system prompt is fixed (`SYSTEM_PROMPT`). The user turn is `state_block(state) + suffix`.
   - The suffix is the `Question:` or `Rate the state:` header plus the instructions, then `Options:` with one
     `"{LABEL}. {name}: {desc}"` line per option, then `Answer with one letter: A, B, ….` (or `one label` beyond 26
     options).
   - The chat template is applied with `add_generation_prompt=True, enable_thinking=False`.
   - Slot labels are A–Z and then two-letter labels (minus `_NOT_ONE_TOKEN`), 256 in total. Each must be exactly one
     token; this is verified at load.
2. **Prefix split**
   - Tokenize each full branch prompt.
   - Take as prefix the longest common token prefix that ends at or before the state block's end (found with a
     marker render).
   - Under the noncausal attention setting, `serve.patch` also requires the prefix block to end exactly at `P`;
     otherwise it is an error.
3. **Questions → branches**
   - `noul` is a 2-option choice (default true/false wording).
   - `score` has ordered levels.
   - A `choice` with more than `max_slots` options is split into balanced chunks of ≤ `max_slots-1` options plus a
     shared "none of the options listed here" slot. Results are recombined with keep-weights (`decide()` in
     `engine.py`).
   - `debias=k` adds k-1 cyclic order shifts per question; all of them share the prefix.
4. **Readout:** take the LM-head logits of the slot tokens at the last suffix token, then softmax at temperature
   `T(kind, k)`. elephant-1 ships no `calibration.json`, so T = 1.0. `allowed_token_mass` (full-vocab logsumexp) is
   diagnostic only.
5. **Limits:**
   - `max_branch_tokens` 32768 bounds prefix + the longest suffix.
   - `max_request_tokens` 65536 bounds prefix + the sum of suffixes.
   - Requests over a limit get a 422 with the reference message text. Nothing is truncated.
6. **Response:** a `SystemOneResponse` with answers and usage; `qev` meta when `debug`.

## 5. Why the reference is slow (what we remove)

| Reference behaviour | Cost | sysone |
|---|---|---|
| A sync `decide()` inside an `async` FastAPI handler | One request on the GPU at a time; the event loop blocks | Token-budget batching across requests |
| Two HF eager forwards per request (prefix, then suffixes) | Weights streamed twice per request; Python/HF overhead | One packed forward per batch of requests |
| `_fork`: deepcopy + `repeat_interleave` of the KV and the GDN state (~48 MB per question) | Memory traffic and OOM back-offs | Branches read the prefix state by pointer; nothing is copied |
| Materialised SDPA block mask | 32-bit index overflow, falls back to one forward per question | Mask is implicit in segment metadata |
| Full 248k-vocab LM head at answer positions | 2 GB GEMV per forward | Slot-row GEMM (≤256×4096) |

Throughput model: a forward costs about `max(weight_bytes / BW, tokens × 13.8 GFLOP / F_bf16)`. On GB10 the
weight-streaming floor (~13.8 GB per forward) dominates until a forward carries several thousand tokens. Batching
across requests is therefore the primary lever, and kernel efficiency the second. Phase 0 measures BW and `F_bf16`
on this box (§11), and the targets below are stated relative to those measurements.

## 6. Execution model

### 6.1 Packed tree batch

One GPU step processes a `DecisionBatch`: several requests, each a tree of one prefix segment and N suffix segments,
all packed into one token stream with no padding.

```
tokens:  [ P_r1 | S_r1q1 | S_r1q2 | P_r2 | S_r2q1 | S_r2q2 | S_r2q3 | ... ]   (T tokens total)
```

```rust
pub struct Segment {            // device-side array of these, plus host mirror
    start: u32, len: u32,       // range in the packed stream
    parent: i32,                // -1 for a prefix, else index of its prefix segment
    pos0: u32,                  // first RoPE position (0 for prefix, P for suffix)
}
pub struct DecisionBatch {
    token_ids: Vec<u32>,                // [T]
    segments: Vec<Segment>,             // prefixes first or interleaved; parent index always valid
    answer_rows: Vec<u32>,              // [Q] packed index of each suffix's last token
    answer_slots: Vec<u16>,             // [Q] number of slots (k) to read for each branch
    // host-only: per-branch owner (request, qid, option order, chunk id) for answer assembly
}
```

*Update:* the implemented types are in `src/contract.rs` (all fields public, `Segment` is `#[repr(C)]`). The
per-branch owners are not stored in the batch: they live in the frontend's `Plan`/`Branch`, and the scheduler maps
logit rows back to requests by each request's first answer row (ARCHITECTURE.md §3.1).

Every layer is computed for the whole stream. The prefix never depends on suffixes, so one pass gives exactly the
reference's two-forward result:

- **Full attention:** query segment `s` attends to KV ranges `{s}` if it is a prefix, or `{parent(s), s}` if it is a
  suffix. Both ranges are non-causal. Two ranges are combined with a log-sum-exp merge.
- **GDN conv1d:** token `t` at segment offset `i < 3` takes its earlier taps from the parent prefix's last `3-i`
  tokens if `parent ≥ 0`, else zeros. This is a gather, so there is no conv-state tensor.
- **GDN delta rule:**
  - Prefix segments scan from a zero state and *write* their final state `[32,128,128]` f32 to a per-batch
    scratch buffer (2 MB × #prefixes per layer, reused across layers).
  - Suffix segments scan from `state[parent]`, read-only. Their final state is never written.
  - Chunk boundaries restart at each segment's start, which matches the reference's prefill-then-suffix chunking.
- **Everything else** (norms, GEMMs, MLP, residuals) is token-parallel over the packed stream.

The suffix scan depends on the prefix's final state *within the same layer*. Each GDN layer is therefore split into
two launches: a prefix scan, then a suffix scan (or one persistent kernel with a grid-wide dependency). Optimisation
later: a request's suffixes can share one inter-chunk pass over the prefix state.

### 6.2 StepGraph changes (from grout)

- Add `GraphDType::{BF16, F32}`. `TensorSpec`, the pool and op signatures become dtype-generic. bf16 is the activation
  type; f32 is used for GDN gates and state and for readout logits.
- Shapes are dynamic in `T` (total tokens) and `num_segments`. Graphs are built and CUDA-graph-captured per
  **T bucket** (e.g. 256, 512, 1k, 2k, 4k, 8k, 16k, 32k), with segment metadata read from device buffers, so one
  captured graph serves any batch whose packed length fits the bucket. Padding tokens form a dummy segment that is
  masked out.
- Remove the decode path (`DecodeCudaGraphRunner`, KV caches, sampling, argmax) from the fork. grout's Qwen3 engine
  is not kept.

### 6.3 Per-layer op list

Linear (GDN) layer:
```
h = zc_rmsnorm(x, ln1)
qkv, z, a, b = gemm(h, [Wqkv; Wz; Wa; Wb])                         (one fused [12352,4096] GEMM)
qkv = tree_conv1d_silu(qkv, conv_w, segments)
q,k,v,g,beta = gdn_prep(qkv, ab, A_log, dt_bias)                  (split, l2norm, gates; f32 g/beta)
o = gdn_chunk_scan(q,k,v,g,beta, segments, prefix_state_scratch)   (2 phases: prefix, then suffix)
o = gated_rmsnorm(o, z, norm_w)
x = x + gemm(o, Wout)
x = x + mlp(zc_rmsnorm(x, ln2))
```
Full-attention layer:
```
h = zc_rmsnorm(x, ln1)
qg = gemm(h, Wq) ; kv = gemm(h, [Wk;Wv])
q,gate,k,v = qk_norm_rope_partial(qg, kv, positions)               (zero-centred norm, rope on dims 0..64)
o = seg_attn_gqa_d256(q,k,v, segments)                             (non-causal, ≤2 KV ranges, LSE merge)
o = o * sigmoid(gate)                                               (fused into attn epilogue)
x = x + gemm(o, Wo)
x = x + mlp(zc_rmsnorm(x, ln2))
```
Readout: `hs = gather(x, answer_rows)` → `zc_rmsnorm(hs, final_norm)` → `logits = gemm(hs, W_slot^T)` with
`W_slot = lm_head[slot_ids]` (256×4096, f32 accumulate) → copied to the host as `[Q, 256]` f32. An opt-in
`allowed_token_mass` runs the full `lm_head` on the `Q` rows only when `debug` is set.

*Update:* the engine applies the final norm over all `T` rows and then gathers (`forward::build`), and the slot
GEMM runs on cuBLAS with f32 output. `allowed_token_mass` is not implemented in the GPU engine yet
(`DecisionEngine::run_with_mass` has a default that reports none).

### 6.4 Kernels (ownership: Tile = cutile-rs, SIMT = cuda-oxide)

| Kernel | Model | Notes |
|---|---|---|
| bf16 GEMM (all projections, MLP) | Tile, or cuBLAS(Lt) via cudarc as baseline | Start on cuBLAS bf16 (Phase 0: 92–101 TFLOP/s at T ≥ 4k on GB10); replace with cutile `persistent_gemm` only where autotuned faster. Fuse gate/up into one GEMM [24576,4096]. Fuse `in_proj_qkv`, `in_proj_z`, `in_proj_a` and `in_proj_b` into one [12352,4096] GEMM (n=64 alone runs at ~15 TFLOP/s). |
| `zc_rmsnorm`, fused `add + zc_rmsnorm` | Tile | Adapt grout's `add_rms_norm_*` to bf16 and `(1+w)`. |
| `qk_norm_rope_partial` | Tile | head_dim 256, rotary 64; also splits the q/gate interleave. |
| `seg_attn_gqa_d256` | Tile | Adapt `fmha_prefill_gqa`: non-causal, segment table, D=256 (check smem; may need BN=32/64), gate epilogue. |
| `tree_conv1d_silu` | SIMT | Gather via the parent table; 8192 channels, kernel 4. |
| `gdn_prep` | Tile | split/repeat/l2norm/softplus/sigmoid, f32 out. |
| `gdn_chunk_scan` | Tile, or SIMT (`GdnOptions::backend`) | WY/UT form as in `torch_chunk_gated_delta_rule`: per 64-chunk, `A = (k_β kᵀ)⊙decay`, unit-lower-triangular solve, then the inter-chunk scan. Varlen over segments, initial state from the parent. Tile (P1): 3xTF32 matmuls, blocked chunk inverse, scan recomputing the WY operands (Tinv and P stored). SIMT (G1, cuda-oxide, `kernels-simt/`, docs/G1.md): `gdn_simt_tinv` (exact bf16 Gram + blocked 16×16 fp32 solve, only Tinv stored) and `gdn_simt_scan` (recomputes every WY operand and P per chunk; `mma.sync` bf16 with raw bf16 q/k rows as the exact operand, fp32 operands split three ways in the products that feed the state and two ways in the output-only ones; state transposed in registers), plus `gdn_simt_gates` (g, beta; q, k normalised in the scan). ~2.7× faster than P1's Tile mixer (0.80 / 3.05 / 11.5 ms against 2.36 / 8.28 / 32.3 ms per layer at 1k / 4k / 16k) and more precise on every prefix of the test batch; Tile stays the default until the end-to-end gate (§9.2) has run with SIMT. |
| `gated_rmsnorm` | Tile | Per 128-head; plain `w`; silu(z). |
| `gather_rows`, slot readout GEMM | Tile | Small. |

Every kernel gets a standalone test against a PyTorch reference implementation (§9.2), at tolerances stated in the
test.

*Update (Phase 2a):* every kernel above is implemented in cuTile (`src/kernels/`), including `tree_conv1d_silu` and
the GDN scan. *Update (G1):* the GDN delta rule also has cuda-oxide SIMT kernels (`kernels-simt/`, docs/G1.md),
opt-in through `GdnOptions::backend`. The
projection GEMMs and the slot readout run on cuBLAS (`src/engine/gemm.rs`). The kernel tests compare against CPU f64
references transcribed from transformers' `modeling_qwen3_5.py` rather than against PyTorch at run time. The op-by-op
mapping is in ARCHITECTURE.md §5.

### 6.5 Memory plan (GB10, T = 16k tokens per step)

- Weights ≈ 15.9 GB bf16, or ≈ 9.0 GB with NVFP4 MLP (layers 13.8 / 6.9 GB + embed 2.0 GB + slot rows 2 MB). The
  full `lm_head` (2 GB) is loaded only if debug mass is enabled.
- On GB10 every byte here is shared with the OS and the server process. The loader must not leave a page-cache
  copy of the safetensors resident (PLATFORM.md P1, P2).
- Activations are transient per layer; the peak is the MLP intermediate at 16k×24576×2 B ≈ 0.8 GB, plus residual and
  QKV buffers.
- GDN prefix-state scratch: #prefixes × 2 MB.
- No persistent KV cache in v1.
- Total is well under 32 GB, so cache and batching headroom are plentiful.

### 6.6 Precision policy

bf16 storage, fp32 accumulation in every GEMM, and fp32 for all norm statistics, GDN gates, state and scan, softmax
LSE, and the readout. This mirrors the reference's casts op for op so drift stays inside bf16 kernel-rounding noise.

NVFP4 MLP path (§3.1):
```
h  = zc_rmsnorm(x, ln2) → quantize_nvfp4(h, gate.input_global_scale)      (fused: norm epilogue emits e2m1 + e4m3)
gu = gemm_fp4(h_q, [W_gate; W_up]_q)  · 1/(a_gs·w_gs) per half            (block-scaled MMA, fp32 accumulate)
m  = silu(g) * u → quantize_nvfp4(m, down.input_global_scale)             (fused)
x  = x + gemm_fp4(m_q, W_down_q) · 1/(a_gs·w_gs)
```
- gate and up have different `weight_global_scale`s, so each half of the fused output gets its own epilogue
  factor.
- On sm_90 (no FP4 MMA) the fallback dequantizes the weights at load and keeps activation QDQ, matching the
  checkpoint's evaluation numerics at bf16 speed (PLATFORM.md).

## 7. Request path and scheduler

```
HTTP (axum/hyper, tokio)
  → validate (serde schema = qev pydantic)        → 422 on failure, same messages
  → render + tokenize (CPU worker pool; tokenizers + minijinja with chat_template.jinja)
  → plan branches (chunking, debias), split prefix/suffix, enforce limits
  → admission queue
  → batcher: packs whole requests into a DecisionBatch until token_budget (default 16k) or max_wait (default 2 ms)
  → GPU worker (single owner of the CUDA context): H2D metadata → graph replay for the T bucket → D2H slot logits
  → answer assembly (calibration, chunk recombination, debias averaging) on CPU
  → response
```

- A request is never split across batches in v1. A request larger than the budget runs alone, using the largest
  bucket or an uncaptured graph. *Update:* there is no uncaptured fallback; the server requires the largest bucket
  (`--max-step-tokens`) to cover `--max-request-tokens` and `--token-budget`. The engine may still split a batch by
  request into several steps when it exceeds a bucket's answer or segment capacity or the GDN state slots.
- Double-buffer the host-to-device metadata so CPU packing of batch n+1 overlaps the GPU run of batch n.
- Backpressure: a bounded queue; return 503 when it is full. Metrics are exposed at `/metrics` (queue depth, batch
  tokens, step ms, decisions/s).
- Other endpoints: `/health` and `/v1/models` (same as qev); `/v1/systemone:batch` is an additive extension.

## 8. Prefix cache across requests (phase 2)

The prefix is one bidirectional block, so only **identical** (system + state) prefixes can be reused; a partial
prefix match is never reusable. Cache by `hash(prefix_ids)` and store:
- full-attention K/V for the prefix: 32 KiB/token;
- GDN final states: 24 × 2 MB;
- conv tails: 24 × 3 × 8192 bf16.

A cached request then becomes "suffixes only" in the packed batch, with parent pointers to cache memory. This uses an
LRU bounded by bytes. It matters for workloads that ask new questions about the same state.

## 9. Correctness strategy

### 9.1 Golden corpus (built by the reference workstream)
- About 300 requests covering: every question type; option counts 2, 5, 26, 27, 255 and 300 (chunking); `debias`
  1 and 3; JSON and string states; Unicode; states from 10 to 16k tokens; one over-limit request of each kind.
- For each request record: rendered prompt strings, `prefix_ids`, `suffix_ids`, slot logits, final probabilities,
  and the response JSON.
- For 10 requests also record per-layer hidden states (after layers 0, 3, 4, 15, 31 and the final norm) at answer
  rows and at sampled prefix rows (W1b, 8 rows per request). Gate (a) uses denser dumps of the same 10 requests
  (`reference/hidden_dense.py`, Phase 2a A1): every prefix row, or every `P // 1024`-th row for the 4k and 16k
  requests, plus every answer row (3,829 rows per tap), in fp32 and bf16 on the served path, plus bf16 null
  realizations (per-question path; three hidden-basis permutations) and two degraded paths for validation. Stored outside git with a committed sha256 manifest.
- The reference runs in **fp32** (the "truth") and in **bf16** (the "production reference"): HF eager with OneJev +
  `serve.patch`.
- **NVFP4 tier:** the same corpus through the checkpoint's evaluation path (compressed-tensors
  `run_compressed=False`, QDQ, batch 1, no cache, TF32 off), and additionally through the shared-prefix path, to
  confirm the equivalence claimed in §3.1. Repeat 3 fresh processes to measure its own run-to-run drift.
- **Two-tier goldens:**
  - *Dev goldens* are generated on this GB10, so development is never blocked: the NGC PyTorch aarch64 container
    if it can run the pinned stack, else a CPU fp32 run. They use the pure-torch GDN fallback if `fla` or
    `causal-conv1d` are unavailable.
  - *Authoritative goldens* are generated on the x86 box with the exact pinned image (`setup.sh`).
  - Both are stored with their provenance (arch, image, torch version, GDN backend). W1 diffs the two tiers once
    and records the spread; that spread feeds the tolerance calibration in §9.2.

### 9.2 Gates

Tolerances are **initial values, expected to be relaxed** once we measure the noise floor. The binding rule is
relative: sysone's error vs the fp32 truth must stay within the envelope the bf16 reference itself shows vs fp32
truth, and vs its own batched/sequential paths. The absolute numbers below are the starting point. Thresholds live in
one config file (`tests/tolerances.toml`), which every test reads, so loosening them is a reviewed one-line change.
Every test always reports the measured distribution (p50/p99/max), not just pass/fail.

| Layer | Gate |
|---|---|
| Frontend | Prompt strings and token ids byte-identical to the goldens for 100% of the corpus; limit and error responses identical. |
| Kernels | Each op against a torch reference on random inputs (fp32 tolerance per op, documented). |
| Model | Per-layer hidden states, gate (a): per row, relative error vs fp32 truth, compared with the bf16 reference's own error on the same rows (the dense row set, ≥ 1000 rows per tap). Per tap: (i) p99 ≤ 1.5 × the reference's p99 (max reported only); (ii) median over rows of ln(sysone / reference error) ≤ ln 1.5; (iii) rows > 1.5× worse than the reference are not significantly more frequent than rows > 1.5× better (one-sided exact binomial sign test, α = 0.01, on counts deflated by a per-tap design effect measured from five bf16 realizations of the reference, because rows of one request move together). Heavy-tailed deep-layer errors are row-specific (summation order decides which rows are sensitive), so a marginal quantile alone is not enough; the paired criteria catch systematic regressions. Validation used five bf16 realizations of the reference (served, per-question, and three hidden-basis permutations): every pairing passes, synthetic regressions fail (`tests/golden/dev/model/hidden_gate_report.json`). The 6 taps x 3 criteria combine without multiple-testing correction: each criterion is evaluated per tap (the sign test at α = 0.01) and the gate fails if any one fails, which is conservative. The norm tap's sign-test verdict for sysone sits close to the 1.5x margin on the design effect (p = 0.038 with D = 18.4; 0.011 with D = 1.0 x measured). |
| End-to-end | Per-option probability |Δ| vs bf16 reference: start at p99 ≤ 0.01 and max ≤ 0.05 (the reference's own batched-vs-sequential spread is ≤ 0.03). Top choice agrees wherever the reference margin is > 0.02. Calibration quality (Brier/ECE on the golden corpus labels where available) is not worse than the bf16 reference beyond ε. |
| NVFP4 | Kernels: activation quantization bit-exact vs compressed-tensors; FP4 GEMMs within 1 ulp of the exact f64 dequant-matmul. Model: **statistical**, because the QDQ reference itself moves by up to 0.2 per option under a batch-shape change (§3.1). On a labeled eval set rebuilt from the checkpoint card's public sources (BANKING77, LEDGAR, LongBench-v2): sysone NVFP4's accuracy, top-1 agreement with the QDQ reference, and mean/p99 \|Δp\| must lie inside the envelope the QDQ reference shows against itself across batch shapes and paths (paired bootstrap). Report vs bf16 as quality information. |
| Batch invariance | The same request alone vs packed with 63 others: max |Δp| ≤ 0.01. |

*Update:* the thresholds actually applied are those in `tests/tolerances.toml` (derived from measured noise), and
ARCHITECTURE.md §6.2 lists which test implements each gate.

## 10. Benchmarks

Harness at `benchmarks/s1/`. It sweeps state length {256, 1k, 4k, 12k}, questions per request {1, 3, 8, 24}, and
concurrency {1, 4, 16, 64, 256}, and reports decisions/s, requests/s, p50/p99 latency, and tokens/s per step.

Baselines:
1. qev reference server (OneJev + `serve.patch`).
2. vLLM with one request per question (`max_tokens=1`, `logprobs` restricted to the slot ids, automatic prefix
   caching on). This is causal by necessity, so it is a speed baseline only; we report its accuracy loss alongside.
   It runs only if vLLM supports Qwen3.5 on this aarch64 sm_121 box (Phase 0 checks).

v1 targets on GB10, re-baselined by Phase 0 (`docs/PHASE0.md`). The box is compute-bound above about 1k packed
tokens, with a bf16 GEMM ceiling of ~7,000 tok/s:
- end-to-end packed throughput ≥ 75% of the hwprobe GEMM-only tokens/s at T ≥ 4k (≈ 5,000 tok/s bf16);
- vs the reference, reported per shape: ≥ 3× decisions/s for states ≤ 1k tokens at concurrency ≥ 16, and ≥ 1.3×
  for states ≥ 4k tokens;
- ≥ 2× vLLM's decisions/s;
- single-request p50 ≤ the reference's.

## 11. Phases and workstreams

The interfaces in §6.1 (`DecisionBatch`), §6.3 (op list), and the `Engine` trait below are the contract between
workstreams:

```rust
pub trait DecisionEngine: Send {
    fn max_tokens_per_step(&self) -> usize;
    fn run(&mut self, batch: &DecisionBatch) -> anyhow::Result<SlotLogits>;  // [Q, 256] f32, row i = answer_rows[i]
}
```
*Update:* the trait also has `run_with_mass` (for debug `allowed_token_mass`), with a default implementation; see
`src/contract.rs`.

**Phase 0: foundations (serial, small)**
- W0. **Done except items blocked on the user** (CUDA 13.4 + tileiras install, docker group); see
  `docs/PHASE0.md`. Done: deps switched to `../cuda-rust` path crates, the Qwen3 engine moved to `legacy/grout/`,
  the contract implemented in `src/contract.rs`, config parsing, and the GB10 bandwidth and GEMM probe.

**Phase 1: parallel workstreams**
- W1 **reference & goldens:** the Python harness and corpus from §9.1. It runs on GB10 first (dev tier) and on x86
  later (authoritative tier). Its output is committed under `tests/golden/{dev,x86}/` (Git LFS for the tensors).
  It also produces `tests/tolerances.toml` seeded from measured noise.
- W2 **frontend:** schema, rendering, chat template, tokenization, prefix split, branch planning, limits, answer
  assembly, calibration. CPU only. Gate: the frontend rows of §9.2.
- W3 **kernels:** the §6.4 table. Each kernel has a test and a microbench, and GDN is first (highest risk).
  - Also the NVFP4 pair: the `quantize_nvfp4` epilogue (bit-exact vs compressed-tensors) and the fused gate/up and
    down block-scaled FP4 GEMMs (from cutile's `nvfp4` and `persistent_gemm`).
  - The FP4 GEMM rate is also added to `hwprobe`.
- W4 **engine:** loader, StepGraph dtype work, packed forward, T-bucket graphs. It starts with cuBLAS and
  torch-validated reference kernels, then swaps in W3 kernels.
  - Loader: bf16 per §3, plus compressed-tensors NVFP4 per §3.1, via the index file; streaming without
    page-cache residency.
  - Weight format is a per-projection property (`Linear::{Bf16, Nvfp4}`), so mixed checkpoints need no special
    case.
- W5 **server & scheduler:** HTTP, batcher, GPU worker, metrics. It develops against a mock `DecisionEngine`.
- W6 **benchmarks:** load generator, plus reference and vLLM baselines.

**Phase 1: done (2026-11-13).** All nine workstreams merged; see `docs/PHASE1.md` for results and open issues.

**Phase 2: integration and performance**
- Phase 2a (parallel):
  - I1 **engine integration (bf16):** the `Kernels` implementation over the W3 kernels, the 32-layer packed
    forward, T-bucket graphs, the real engine behind the server, end-to-end gates against the bf16 goldens, and a
    per-op forward profile.
  - P1 **GDN speed:** blocked triangular solve, recompute instead of storing intermediates, per-phase block size;
    precision variants exposed for the end-to-end gate to choose.
  - P2 **attention speed:** a D-split accumulator for head_dim 256, 2 CTAs/SM.
  - E1 **eval set:** the labeled set and statistical NVFP4 gate (§9.2), plus a scorer for any `/v1/systemone`
    server.
  - S1 **server fixes** from the W5 review.
  - *Update:* Phase 2a is done; see `docs/PHASE2A.md`.
  - G1 **GDN in cuda-oxide** (docs/G1.md, after Phase 2a): SIMT WY/solve and chunk scan, `GdnBackend::{Tile, Simt}`;
    mixer 2.36 / 8.28 / 32.3 ms (P1 Tile) -> 0.80 / 3.05 / 11.5 ms per layer at 1k / 4k / 16k (~2.7×;
    ~5× against Phase 1). Opt-in until the end-to-end gate; the 16k target (~10 ms) is not met yet.
- Phase 2b: NVFP4 end to end; fuse the GDN conv into the SIMT kernels (G1's next step); precision choices from the end-to-end
  gate; GEMM autotune and fusions guided by I1's profile.
- Prefix cache (§8).

**Phase 3**
- Images (vision tower + MRoPE).
- Wider quantization (NVFP4/FP8 on attention and GDN projections), gated by calibration metrics (Brier/ECE on a held-out set within ε of bf16).
- mammoth-1 (27B).
- sm_100 tuning.

## 12. Repository layout (target)

*Update:* this was the planned layout. The actual layout differs (for example `sched/` also has `pool.rs`,
`metrics.rs` and `affinity.rs`; the engine has no `readout.rs` or `buckets.rs`; kernels live in per-family modules
under `src/kernels/`; the integration tests are `tests/engine_*.rs`, `tests/kernels_*.rs`, ...). See the module map in
ARCHITECTURE.md §4.

```
src/
  main.rs            server binary (CLI: --model, --listen, --token-budget, --max-wait-us)
  api/               schema (serde), handlers, errors
  frontend/          render.rs, template.rs, tokenize.rs, plan.rs (branches/chunks/debias), assemble.rs
  sched/             queue.rs, batcher.rs, worker.rs
  engine/            config.rs, loader.rs, graph.rs (StepGraph), forward.rs, readout.rs, buckets.rs
  kernels/           tile/ (cutile modules), simt/ (cuda-oxide crate), gemm.rs (cuBLASLt + cutile)
tests/               golden/, frontend_golden.rs, kernels_*.rs, model_layers.rs, e2e.rs, batch_invariance.rs
benchmarks/s1/       loadgen, baselines, sweep scripts
reference/           python harness (pinned OneJev + serve.patch), golden generator
```

The cuda-oxide SIMT kernels build with `cargo oxide` on a pinned toolchain (1.99.1) in the standalone crate
`kernels-simt/`. Its PTX is a checked-in artifact with a hash manifest (`kernels-simt/build.sh <arch>`), embedded by
`build.rs` only when it matches the sources, so the main crate stays on stable Rust and never needs cargo-oxide
(docs/G1.md, "Build").

## 13. Open questions and risks

1. **Trained length limit (resolved).** Match the reference defaults and make the limits configurable (§1). The
   card's 16,384 is documented as the trained length.
2. **D=256 attention on sm_121.** The shared-memory budget may force small tiles. Fallback: a split-D two-pass, or
   cuDNN SDPA with a segment mask, as an interim.
3. **GDN kernel performance in the Tile model.** The triangular solve and the sequential inter-chunk scan are
   awkward. Fallback: SIMT via cuda-oxide, with fla's Triton kernel as the algorithmic reference.
4. **Images (phase 3).** MRoPE position ids for image tokens and the block structure with images inside the prefix
   need a reference trace before design.
5. **Reference on aarch64 (resolved).** Dev goldens are generated on GB10 and authoritative goldens on the x86 box
   (§9.1). Residual risk: the aarch64 dev reference may differ from x86 by more than our tolerance. W1 measures this
   first.
6. **bf16 batch-variance.** The reference's own paths differ by up to 0.03. §9.2 tolerances are expected to need
   loosening; this is to be calibrated from measured noise, not guessed.
8. **NVFP4 (resolved: moved into Phase 1/2).** Lower precision is the main lever on compute-bound long states.
   Expected GEMM time ≈ 0.30 + 0.70/r of bf16, where r is the FP4/bf16 GEMM speedup measured on sm_121. The
   checkpoint is experimental: it narrowly missed its own accuracy non-inferiority bound. sysone serves it
   faithfully; qualifying it is outside sysone.
9. **Per-checkpoint request limits.** The NVFP4 checkpoint documents a 16,384-token limit (calibrated to 16,017).
   Proposal: limits come from a per-checkpoint profile, defaulting to the reference values for bf16 and to 16,384
   total for the NVFP4 checkpoint.
7. **cutile 0.4 → 0.6 API churn** when porting grout kernels. `cutile-kernels` upstream already contains ports of
   most of them.
