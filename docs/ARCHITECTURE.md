# sysone: architecture and code guide

This document explains how the code is put together: what each module does, how one request moves through the
system, what the main types and data layouts mean, which library runs each GPU op, and how correctness is checked.
It describes the code as it is. For *why* it is built this way, read [DESIGN.md](DESIGN.md); for measured numbers,
read the phase reports ([PHASE0.md](PHASE0.md), [PHASE1.md](PHASE1.md), [PHASE2A.md](PHASE2A.md), [G1.md](G1.md), [E2.md](E2.md),
[N1.md](N1.md), [BULK.md](BULK.md)).

Code references name files, types and functions (`src/engine/gpu.rs`, `GpuEngine::step`). Line numbers are avoided
on purpose. `cargo doc --no-deps --open` renders the per-module docs that this file points into.

## Contents

1. [Overview and status](#1-overview-and-status)
2. [Request lifecycle](#2-request-lifecycle)
3. [Core concepts](#3-core-concepts)
4. [Module map](#4-module-map)
5. [GPU ops of one forward](#5-gpu-ops-of-one-forward)
6. [Correctness and testing](#6-correctness-and-testing)
7. [Benchmarks and scripts](#7-benchmarks-and-scripts)
8. [Conventions for reading the code](#8-conventions-for-reading-the-code)
9. [Glossary](#9-glossary)
10. [Open questions](#10-open-questions)

## 1. Overview and status

sysone is an HTTP server that answers `POST /v1/systemone` decision requests with the elephant-1 model
(`baselabs/elephant-1`, a Qwen3.5 text decoder with 24 GatedDeltaNet layers and 8 full-attention layers). It is a
drop-in replacement for OneJev's `qev` server: same request and response JSON, same error bodies, same
probabilities within measured tolerances. A request is a *state* plus several *questions*. Each question becomes one
or more *branches*, and each branch is scored by reading the model's logits for the answer-slot tokens (`A`, `B`,
...) at the last token of that branch's prompt.

The main idea is the **packed tree forward**. All branches of a request share a prefix (system prompt + state), and
the server packs many requests into one token stream: each request contributes its prefix once, followed by one
short suffix per branch. One forward pass over that stream scores every branch of every request in the batch,
without copying any cache.

The code is Rust on top of `../cuda-rust` (cuda-core, cuda-async, cuTile via `cutile`/`cutile-kernels`), with
cuBLAS (through `cudarc`) for the bf16 GEMMs.

**Status (Phase 2a, see [PHASE2A.md](PHASE2A.md) for numbers):**

| Area | State |
|---|---|
| HTTP API, frontend, scheduler | Complete for text requests. Matches qev on the render golden corpus and on qev's error bodies. |
| bf16 GPU engine (`--engine bf16`) | Full 32-layer packed forward with T-bucket CUDA graphs. The end-to-end gate (b) and the batch-invariance gate pass. |
| Hidden-state gate (a) | Redefined by A1 (dense rows, paired criteria; PHASE2A.md) and passing with both GDN backends (E2.md). |
| NVFP4 engine (`--engine nvfp4`, N1) | The NVFP4-MLP checkpoint on the same engine: MLP GEMMs on the block-scaled FP4 kernels, activation quantization fused into the preceding norm / SiLU·mul (bit-identical to the standalone quantize), everything else bf16. Passes `[eval.nvfp4]`, `[nvfp4]` and `[batch_invariance.nvfp4]`; 1.48x the bf16 forward at 16k ([N1.md](N1.md)). |
| Batch invariance, last-layer trimming (L1) | bf16 GEMMs on one pinned cuBLASLt algorithm per weight shape (§3.6) and one-token RoPE tile blocks: a branch's logits are bit-identical alone, packed or in any bucket, in both engines. The last layer runs its queries, attention, MLP and the final norm on the answer rows only (§3.5), bit-identical to the untrimmed forward. [L1.md](L1.md). |
| `allowed_token_mass` (`?debug=1`) | Computed by `MockEngine` only. `GpuEngine` does not override `DecisionEngine::run_with_mass`, so the field is absent with the GPU engine. |
| Prefix cache across requests (DESIGN §8), images, `head` readout | Not implemented. |
| Bulk tagging (B1) | `sysone bulk` (`src/bulk/`, docs/BULK.md): JSONL jobs in, results keyed by id out, through the server's own `prepare`/`finish`; window packing to full buckets; one worker process per GPU from one queue; resumable. |
| GDN on cuda-oxide SIMT kernels (G1, E2) | Implemented (`kernels-simt/`, `src/kernels/gdn/simt.rs`, docs/G1.md) and wired into the engine (E2, docs/E2.md): `--gdn-backend auto\|tile\|simt`, `EngineOptions::gdn_backend`. Default on sm_121 (`auto`), where every engine gate and the eval gate pass with it; cuTile on every other architecture until validated there. |

## 2. Request lifecycle

```
client
  │  POST /v1/systemone?debug=..&debias=..      (api::server::systemone)
  ▼
read_body ── body budget (BodyBudget) ──────────────── 413 too large / 503 body memory
  │
  ▼  CpuPool::try_run (new-work queue, bounded) ────── 503 cpu_queue
prepare(): pyjson::parse → SystemOneRequest::validate → check_model → Frontend::plan
  │        (422 schema / 404 model / 422 ValueError from planning)
  │        Frontend::plan = plan_unchecked (branches, render, tokenize, prefix split) + check (limits, _split)
  ▼
run_ready(): Scheduler::submit([Work{plan, debug}]) → AdmissionQueue::try_push_all ── 503 queue
  │
  ▼  batcher thread (sched::batcher::run)
Packer: Plan::push_into → DecisionBatch::push_request, until token_budget or max_wait (+ engine idle)
  │  Packer::take → DecisionBatch::validate
  ▼  mpsc::sync_channel(1)   (one batch running, one packed and waiting)
GPU worker thread (sched::worker::run): engine.run(&batch) or engine.run_with_mass(&batch)
  │
  ▼  GpuEngine::run → split (if needed) → pick_bucket → step:
  │     stage: TileKernels::prepare, MetaUploader::pack + upload, TileKernels::update
  │     BucketGraphs::launch (replay)  or  Executor::run_eager (+ capture on first use)
  │     read_logits: D2H into pinned buffer → SlotLogits [Q, 256]
  ▼  oneshot per request: Scored { logits rows, allowed_mass, timings }
CpuPool::run (finishing queue, never refused) → finish(): Frontend::assemble → response + qev meta + timing
  │
  ▼
respond(): metrics → JSON response
```

Step by step, with the code that runs:

1. **HTTP** (`src/api/server.rs`). `router` maps `/v1/systemone` to `systemone`, which reads the body with
   `read_body` (at most `MAX_BODY_BYTES`, charged against the in-flight `BodyBudget`) and calls `decide_one`.
   `/v1/systemone:batch` (`systemone_batch` → `decide_batch`) takes a JSON array of request bodies and admits the
   valid ones together. Query parameters are parsed by `params` using `api::query` (`parse_qsl`, `py_int`), with
   qev's semantics: any non-empty `debug` value turns debug on.
2. **Parse and validate** (`src/api/schema.rs`, `src/frontend/pyjson.rs`), on a CPU pool thread inside `prepare`.
   `pyjson::parse` is Python's `json.loads` (key order kept, duplicate keys, big integers). `SystemOneRequest::validate`
   produces pydantic's exact 422 body (`ApiError::Unprocessable`); `check_model` gives the 404 for an unknown model.
3. **Plan** (`src/frontend/plan.rs`): `Frontend::plan` = `plan_unchecked` then `check`.
   - Branches: one per question; `debias=k` adds `k-1` cyclic shifts of the options; a `choice` with more options
     than `max_slots` is split by `plan_chunks` into balanced chunks, each with an extra `NONE_OPTION` slot.
     `render::render_question` builds each branch's suffix text and slot labels.
   - Render: `template::ChatTemplate::frame` renders the chat template once per request and returns the `head`
     (system turn + state block) and `tail` around the first branch's suffix.
   - Tokenize: the head once, and `suffix_text + tail` once per branch (`tokenize::Tokenizer`).
   - Prefix split: normally the prefix is exactly the head's tokens. If the state itself contains `PREFIX_MARK`,
     the reference's marker-render rule (longest common token prefix) is reproduced.
   - `check`: the two length limits (`Limits`), then serve.patch's `_split` check that the prefix block ends exactly
     at the shared prefix, which also fills `Branch::option_token_spans`.
   The result is a `Plan`: `prefix_ids`, one `Branch` per answer row (with `suffix_ids`, `labels`, `order`, `chunk`),
   and the counts used for limits and `usage`.
4. **Admission** (`src/sched/mod.rs`, `src/sched/queue.rs`). `Scheduler::submit` wraps each plan in a `Job` with a
   oneshot reply channel and pushes all of them or none (`AdmissionQueue::try_push_all`). A full queue is a 503.
5. **Batching** (`src/sched/batcher.rs`). The batcher thread pops jobs and appends them to a `Packer`, which calls
   `Packable::push_into` (for the server, `Plan::push_into` → `DecisionBatch::push_request`). A batch closes when it
   reaches `token_budget`, or when its oldest job has waited `max_wait` *and* the engine is idle (`in_flight == 0`).
   A job that would cross the budget is carried to the next batch; a job larger than the budget runs alone; a job
   larger than the engine's `max_tokens_per_step` is refused with an error. `Packer::take` runs
   `DecisionBatch::validate` before handing the batch over.
6. **GPU worker** (`src/sched/worker.rs`). One thread owns the engine (and its CUDA context). It receives `Packed`
   batches through a one-slot channel, so the batcher can pack batch n+1 while batch n runs. It calls `run_with_mass`
   when any job wants debug masses, else `run`; catches panics; checks the row count; and sends each job its rows of
   logits as a `Scored`.
7. **Engine step** (`src/engine/gpu.rs`, `GpuEngine::run`):
   - `split` / `split_batch`: if the batch exceeds the largest bucket's answer or segment capacity, or holds more
     prefixes than the GDN state slots (`--max-prefixes`), it is split by request into several steps.
   - `pick_bucket`: the smallest `Bucket` that fits tokens, answers and segments (plus one pad segment). If none fits,
     the batch is split; in practice the scheduler prevents a token overflow, because its token budget is clamped to
     `GpuEngine::max_tokens_per_step` (the largest bucket).
   - `stage`: `TileKernels::prepare` (allocate the bucket's GDN plan and attention work list once),
     `MetaUploader::pack` + `upload` (batch metadata into a pinned host slot, then async H2D), and
     `TileKernels::update` (rewrite the GDN segment/chunk tables and the attention work list for this batch).
   - Run: replay the bucket's CUDA graph (`BucketGraphs::launch`) if one is captured; otherwise run eagerly
     (`Executor::run_eager`) and then capture it (`Executor::capture`, `BucketGraphs::insert`).
   - `read_logits`: copy the `[Q, n_slots]` f32 logits into a persistent pinned buffer and widen each row to
     `MAX_SLOTS` (256) columns as `SlotLogits`.
8. **Readout and assembly** (`src/frontend/assemble.rs`), back on the CPU pool's finishing queue in `finish`:
   `Frontend::assemble` applies the calibration temperature (`Calibration`; elephant-1 ships none, so T = 1),
   softmax, chunk recombination and debias averaging, all in f64 with Python's float semantics, and builds each
   `Answer`. `Frontend::response` and `Frontend::meta` build the body; `finish` adds `qev.timing` (`render_s`,
   `queue_s`, `forward_s`, `total_s`) and, with `debug`, `qev.diagnostics`.
9. **Respond**: `respond` records metrics (`sched::metrics::Metrics`) and returns the response.

The bulk path (`sysone bulk`, docs/BULK.md) reuses steps 2, 3 and 8 (`api::server::prepare` and `finish`, the same
functions) and replaces the rest: no HTTP, a window of planned requests packed into full steps (`bulk::pack`), and
one engine per GPU in a worker process (`bulk::worker`) fed from one queue (`bulk::run`).

The process is wired together in `src/main.rs`: it parses the CLI, loads `ModelConfig` and the `Frontend`, builds an
`EngineFactory` (mock, bf16 or nvfp4; the factory runs on the GPU worker thread so the CUDA context belongs to it), starts
`AppState::start_with`, and serves with `api::server::serve`.

## 3. Core concepts

### 3.1 The packed tree: `DecisionBatch` and `Segment`

Defined in `src/contract.rs`, the interface between frontend, scheduler and engine.

```rust
pub struct Segment { pub start: u32, pub len: u32, pub parent: i32, pub pos0: u32 }
pub struct DecisionBatch {
    pub token_ids: Vec<u32>,     // [T] the packed stream
    pub segments: Vec<Segment>,  // tile token_ids exactly, in order
    pub answer_rows: Vec<u32>,   // [Q] last token of each suffix segment
    pub answer_slots: Vec<u16>,  // [Q] slots (options) read for each branch, 1..=256
}
```

- A **prefix** segment has `parent = -1` and `pos0 = 0`.
- A **suffix** segment has `parent` = index of its prefix segment (an earlier entry of `segments`) and
  `pos0 = parent.len`, so its RoPE positions continue where the prefix ended.
- `answer_rows[i]` is the packed index of the last token of the i-th suffix segment; that is where the slot logits
  are read.

Example, from the unit test `contract::tests::push_builds_a_valid_tree`: two requests, the first with prefix
`[1,2,3]` and suffixes `[10,11]` and `[12]`, the second with prefix `[4,5]` and suffix `[20,21,22]`:

```
token_ids:   1  2  3 | 10 11 | 12 | 4  5 | 20 21 22
index:       0  1  2 |  3  4 |  5 | 6  7 |  8  9 10
segments:    {0,3,-1,0} {3,2,0,3} {5,1,0,3} {6,2,-1,0} {8,3,3,2}
answer_rows: [4, 5, 10]
```

`DecisionBatch::push_request` builds this layout (prefix, then that request's suffixes) and `DecisionBatch::validate`
checks every invariant. The contract allows other orders (any suffix after its prefix), but the engine's
`split_batch` assumes the `push_request` layout (see §4.4).

`SlotLogits` is the engine's answer: `rows × 256` f32, row-major, row i for `answer_rows[i]`; only the first
`answer_slots[i]` values of each row are meaningful.

### 3.2 What each layer does across segments

Every op runs over the whole packed stream. Token-parallel ops (norms, GEMMs, MLP, residual adds) do not care about
segments. Three places do:

- **Full attention (block-bidirectional mask).** The reference mask is `causal OR (same block)`, where the prefix is
  block 0 and each question suffix is block 1. Concretely: a prefix token attends to its whole prefix; a suffix
  token attends to its whole prefix plus its whole own suffix (both non-causal), never to other suffixes or other
  requests. `kernels::attention::AttnPlan::new` turns the segment table into work items: each covers up to `bm`
  query rows of one segment and attends to range A (the prefix: the segment itself, or the parent) and range B (the
  suffix itself, empty for a prefix), merged with an online-softmax (log-sum-exp) merge.
- **GDN causal conv1d (kernel 4).** A token at offset `i < 3` in a suffix takes its missing earlier taps from the
  last `3 - i` tokens of the parent prefix; in a prefix they are zeros. `kernels::gdn::GdnTables::build` encodes this
  per 64-row chunk as `prev_end, prev_avail` (for a suffix's first chunk: the parent's end and `min(3, parent.len)`).
- **GDN delta rule (recurrent state).** Prefix segments scan from a zero state and write their final state
  `[32, 128, 128]` f32 into a state slot of `GdnScratch.state`. Suffix segments then scan starting from their
  parent's state slot, read-only. So the scan runs in two phases per layer: all prefixes, then all suffixes. Chunks
  restart at every segment start.

Linear-attention (GDN) layers stay causal; only full attention uses the bidirectional block.

`src/engine/cpu_ref` implements exactly these semantics on the CPU in f32 (`Layout::from_batch`), and also the
reference's single-sequence form with an explicit mask (`Layout::flat`); `src/engine/cpu_ref/tests.rs` checks that
the two agree.

### 3.3 T buckets, the arena and CUDA graphs

A CUDA graph replays fixed kernel launches with fixed pointers and grid sizes, so it cannot follow a batch's exact
size. The engine therefore plans for a few capacities:

- `graph::Bucket { tokens, answers, segments }`. `BucketPlan::with_answer_density` derives
  `answers = min(ceil(T / tokens_per_answer), max_answers)` (`EngineOptions::tokens_per_answer`,
  default 8; L1, [L1.md](L1.md)) and `segments = min(T, max_segments) + 1` for each token size T (the `+1` is the
  padding segment). The answer-row values (the readout, and the trimmed last layer) are sized by `answers`; a batch
  denser in answers than its bucket allows runs in the next larger bucket. `tokens_per_answer = 1` gives Phase 2a's
  `min(T, max_answers)`. `contract::bucket_answers` is the same formula for `StepLimits` (bulk packing).
- Buckets come from `EngineOptions::buckets`; `main.rs` uses `EngineOptions::buckets_up_to(max_step)`, i.e. powers of
  two from 256 up to the first one ≥ `--max-step-tokens` (by default the larger of `--token-budget` and
  `--max-request-tokens`, so 256 … 65536 with default flags).
- `forward::build` produces one `StepGraph` (a list of `GraphOp`s over SSA `ValueId`s) whose shapes use symbolic
  dims (`Dim::Tokens`, `Dim::Answers`, `Dim::Segments`). `StepGraph::plan` turns it into a `PoolPlan` per bucket: a
  byte offset for every value inside one activation arena, reusing memory by liveness (first fit). `BucketGraphs::new`
  allocates one arena sized for the largest bucket; all buckets share it, because only one step runs at a time.
- Every batch-dependent input lives in fixed device buffers that are rewritten before each step: the metadata block
  (`engine::meta::DeviceMeta`), the GDN tables (`GdnPlan`, one per bucket) and the attention work list (padded to the
  bucket's maximum item count with empty items).
- The first step in a bucket runs eagerly (this also JIT-compiles the cuTile kernels and, in pinned GEMM mode,
  pins each new weight shape's cuBLASLt algorithm; §3.6), then `Executor::capture` records the same launches as a
  CUDA graph. Later steps in that bucket only
  upload metadata and replay. `tests/engine_e2e.rs::graph_replay_equals_eager` checks replay is bit-identical.

Padding: a batch of `t` tokens in a bucket of `T > t` gets a dummy segment covering rows `t..T` with
`parent = meta::PAD_PARENT` (-2). Its token ids are 0 and its positions count from 0. Segment-aware kernels skip it;
token-parallel kernels compute garbage rows there that no real row ever reads.

### 3.4 Metadata layout

`MetaUploader::pack` writes one contiguous, 256-byte-aligned block per step (see the `meta.rs` module docs):
`header [8] u32` (`T`, real segments, answers, prefixes, bucket tokens, pad segment index), `token_ids`, `positions`
(`pos0 + offset`), `token_seg` (segment of each token), `segments` (16 bytes each, same layout as `contract::Segment`,
which is `#[repr(C)]`), `seg_state` (GDN state slot per segment), `answer_rows`, `answer_slots`, `answer_pos` (the
RoPE position of each answer row, for the trimmed last layer). Two pinned host slots alternate (double buffering); a
slot is reused only after the event behind its last copy completes.

### 3.5 Readout (`slot_lm`)

The model's full LM head has 248,320 rows; sysone keeps only the rows of the slot tokens. The frontend finds each slot
label's token id (`Tokenizer::verify_slots`, 256 labels: A–Z then two-letter labels minus `render::NOT_ONE_TOKEN`),
the loader gathers `lm_head[slot_ids]` into `WeightRef::SlotHead` (`[n_slots, 4096]`), and the forward ends with:

```
untrimmed (EngineOptions::trim_last_layer = false, or a debug tap on the last layer / final norm):
hn     = add_zc_rmsnorm(x += last MLP output, final_norm)    over all T rows
hs     = gather_rows(hn, answer_rows)                         [Q, 4096] bf16
logits = hs · W_slotᵀ                                         [Q, n_slots] f32 (cuBLAS, f32 output)

trimmed (default, L1): the last layer (full attention) gathers the answer rows after its input norm
kv     = hn · [Wk; Wv]ᵀ, k = k_norm_rope(kv)                  every token (the answer rows attend to them)
q_a    = qk_norm_rope(gather(hn) · Wqᵀ)                       [Q, ..] at the answer positions
o_a    = seg_attn_answers(q_a, k, kv) · Woᵀ                   one work item per answer row
x_a    = gather(x) += o_a; post-norm, MLP, final norm         [Q, 4096]
logits = hs_a · W_slotᵀ
```

Only rows that are never read are skipped: the last layer's queries, attention output, `o_proj`, residual,
post-norm and MLP of non-answer rows, and the final norm of non-answer rows. With pinned GEMMs (§3.6) the trimmed
logits are bit-identical to the untrimmed ones (`tests/engine_e2e.rs::trimmed_last_layer_equals_untrimmed`).
`forward::can_trim_last_layer` turns trimming off when a tap needs the last layer's rows (`engine_layers`).

Softmax, temperature and everything after the logits happen on the CPU in `frontend::assemble`.

### 3.6 bf16 GEMMs: pinned algorithms (batch invariance)

`EngineOptions::gemm` (`--gemm pinned|heuristic`). `GemmMode::Heuristic` is `cublasGemmEx` with
`CUBLAS_GEMM_DEFAULT`: cuBLAS picks the kernel and split-K per call shape, so a row's bf16 projection can round
differently in a 1k bucket than in a 16k one, and a request packed with others is not bit-identical to it alone
(N1: 103/164 branches). `GemmMode::Pinned` (default) runs cuBLASLt with one algorithm per weight shape (N, K,
strides, output type), chosen once from cuBLASLt's heuristic at `gemm::PINNED_REF_ROWS` (16,384) rows, restricted
to candidates with `SPLITK_NUM <= 1` and no reduction scheme, and used for every T (checked with
`cublasLtMatmulAlgoCheck` per row count). Each output element is then one fp32 dot product whose order depends only
on the kernel's K tiling, so a row's result does not depend on the number of rows or its position
(`tests/engine_gemm.rs::pinned_gemm_rows_do_not_depend_on_the_batch`). Same numerics otherwise (fp32 compute and
scale, one output rounding). Costs and the resulting batch invariance: [L1.md](L1.md).

## 4. Module map

`src/lib.rs` declares the modules; `src/main.rs` is the server binary. Each subsection lists responsibility, key
items, inputs/outputs and the gotchas worth knowing before changing the code.

### 4.1 `contract` (`src/contract.rs`)

- **Does:** defines the batch/engine interface: `Segment`, `DecisionBatch`, `SlotLogits`, `MAX_SLOTS`,
  the `DecisionEngine` trait (`max_tokens_per_step`, `run`, `run_with_mass`, `info`, `step_limits`,
  `last_bucket_tokens`), `StepLimits`, `MassReadout`, `allowed_mass`, `logsumexp_f32`, and `MockEngine`.
  `step_limits` (buckets, answer/segment/prefix limits) lets `sysone bulk` pack steps that fit exactly;
  `last_bucket_tokens` reports the padded size of the last run (`sysone_bucket_tokens_total`, bulk padding).
- **Invariants:** see `DecisionBatch::validate` (§3.1). Every producer must hand over a valid batch; engines may
  assume it.
- **`MockEngine`:** deterministic fake logits from a hash of each branch's own tokens, so results are
  batch-invariant. Used by `--engine mock`, the scheduler tests and `tests/server_e2e.rs`.

### 4.2 `api` (`src/api/`)

| File | Responsibility | Key items |
|---|---|---|
| `server.rs` | axum router, handlers, request flow, overload handling, qev's headers and slash redirects | `router`, `serve`, `AppState::start_with`, `ServerInfo`, `ServerOptions`, `BodyBudget`, `prepare`, `run_ready`, `finish`, `Outcome` |
| `schema.rs` | Request/response types and pydantic-identical validation errors | `SystemOneRequest::{from_body, validate}`, `Question`, `QuestionKind`, `ApiError`, `check_model`, `Answer`, `SystemOneResponse`, `QevMeta` |
| `query.rs` | Starlette/Python query parsing | `parse_qsl`, `get`, `py_int` |
| `client.rs` | Minimal HTTP/1.1 keep-alive client for tests and `loadgen` | `Client`, `Reply` |

Gotchas:
- Differences from qev that a client can see (503s from the queues and body budget, 413 for huge bodies or oversize
  `:batch` calls, the extra `qev.timing.queue_s`) are listed in the `server.rs` module docs.
- Validation is hand-written, not derived, because the 422 body has to match pydantic's byte for byte. Change it
  only together with `tests/golden/dev/render/invalid.jsonl` / `tests/frontend_golden.rs`.
- `Work` (in `server.rs`) is the scheduler's `Packable` for a request: its `num_tokens` is `Plan::total`.

### 4.3 `frontend` (`src/frontend/`)

CPU only. `Frontend` (in `mod.rs`) holds the tokenizer, chat template, slot ids, limits, the `noncausal` flag and the
calibration; it is `Sync` and shared by all CPU workers.

| File | Responsibility | Key items |
|---|---|---|
| `mod.rs` | Loading a checkpoint's frontend parts; response and meta builders | `Frontend::{load, assemble, response, meta}` |
| `render.rs` | Question → suffix text and slot labels; state → state block (`qev/prompt.py`) | `render_question`, `question_parts`, `state_block`, `slot_labels`, `SYSTEM_PROMPT`, `PROMPT_VERSION` |
| `template.rs` | `chat_template.jinja` via minijinja, configured like transformers' Jinja2 | `ChatTemplate::{render, frame}` |
| `tokenize.rs` | `tokenizer.json` via HF tokenizers, no special tokens | `Tokenizer::{encode, encode_with_offsets, verify_slots}` |
| `plan.rs` | Request → branches → token ids; limits; serve.patch `_split`; a question set rendered and its suffixes tokenized once for many states | `Frontend::{plan, plan_unchecked, check, render_branches, prepare_questions, plan_prepared}`, `Plan`, `Branch`, `PreparedQuestions`, `Limits`, `PlanError`, `plan_chunks` |
| `assemble.rs` | Slot logits → answers (temperature, softmax, chunk recombination, debias averaging), op for op in f64 | `assemble`, `Calibration`, `Assembled` |
| `pyjson.rs` | JSON and float formatting with Python's exact semantics | `Json`, `parse`, `dumps_compact`, `dumps_indent2`, `py_float_repr`, `py_round6`, `py_fsum` |

Gotchas:
- Byte-exactness with the reference is the requirement here: prompts, token ids, error messages and float formatting
  are all compared byte for byte in `tests/frontend_golden.rs` and `tests/frontend_assemble.rs`. "Equivalent" changes
  (e.g. using serde_json, or `{}` float formatting) break the gate.
- The once-per-request head tokenization relies on the tokenizer always splitting between the head (ending in
  `</state>\n\n`) and the suffix; `frontend_golden::per_request_shortcuts_equal_full_prompt_rendering_and_tokenization`
  checks this on the whole corpus.
- The same split makes a branch's tokens after the head `encode(suffix_text + tail)`, which does not depend on the
  state: `plan_prepared` reuses them across documents when the request's tail is the prepared one (it always is with
  qev's template); `tests/bulk_qset.rs::prepared_plans_equal_full_plans` checks plan equality.
- `PlanError::Value` becomes a 422 with qev's message; `PlanError::Internal` becomes a 500.

### 4.4 `sched` (`src/sched/`)

| File | Responsibility | Key items |
|---|---|---|
| `mod.rs` | Wiring: starts the GPU worker (which builds the engine) and the batcher | `Scheduler::{start_with, submit, policy, shutdown}`, `SchedConfig`, `Packable`, `Job`, `Packed`, `Scored`, `EngineFactory` |
| `queue.rs` | Bounded admission queue; never blocks the producer | `AdmissionQueue::{try_push_all, pop, close}`, `Pop` |
| `batcher.rs` | Packs whole requests into one batch (token budget, max wait, engine-idle rule) | `run`, `Packer`, `BatchPolicy` |
| `worker.rs` | The thread that owns the engine; runs batches, splits results per job | `run` |
| `pool.rs` | Fixed CPU thread pool with a bounded new-work queue and an unbounded, prioritised finishing queue | `CpuPool::{try_run, run}`, `PoolError`, `DEFAULT_QUEUE_PER_THREAD` |
| `metrics.rs` | Prometheus text for `/metrics` | `Metrics`, `render_front` |
| `affinity.rs` | `--pin-cores`: find performance cores in sysfs and pin threads | `performance_cores`, `maybe_pin` |

Gotchas:
- `BatchPolicy::token_budget` is clamped to the engine's `max_tokens_per_step` in `Scheduler::start_with`.
- A request is never split across batches by the scheduler. (The GPU engine may still split a *batch* by request
  into several steps, see §4.5.)
- The engine is built inside the GPU thread by the `EngineFactory` closure, because a CUDA context is bound to the
  thread that creates it.

### 4.5 `engine` (`src/engine/`)

| File | Responsibility | Key items |
|---|---|---|
| `config.rs` | Parse `config.json` and refuse anything v1 cannot serve; the per-checkpoint profile (weight format, request limits; NVFP4: 16,384 per branch) | `ModelConfig::{load, check, profile}`, `CheckpointProfile`, `WeightFormat`, `TextConfig`, `LayerType` |
| `loader.rs` | Read safetensors headers; stream tensors to the GPU without leaving a page-cache copy | `Checkpoint::open`, `Streamer`, `CacheMode::{Direct, DropBehind, Keep}`, `StreamOptions` |
| `weights.rs` | Device weights; fused projections; NVFP4 or bf16 per projection; slot head | `ModelWeights::load`, `Linear::{Bf16, Nvfp4}`, `LinearPart`, `LoadOptions`, `LoadReport` |
| `graph.rs` | The StepGraph IR, arena planning, executor, profiler, the `Kernels` trait, bucket graphs | `StepGraph`, `GraphOp`, `ValueId`, `TensorRef`, `Dim`, `Bucket`, `BucketPlan`, `PoolPlan`, `Executor`, `BucketGraphs`, `Kernels`, `WeightStore`, `Profiler` |
| `graph/tensor.rs` | Device allocations and typed, strided views | `DevBuf`, `DevTensor`, `GraphDType` |
| `forward.rs` | Builds the 32-layer forward as a `StepGraph`, bf16 or NVFP4 per MLP projection; the last layer trimmed to the answer rows (L1) | `build`, `build_opts` (`ForwardOptions`: NVFP4 quantize fused or standalone, `trim_last_layer`), `build_with` (untrimmed), `can_trim_last_layer`, `check_dims`, `Forward`, `Taps` |
| `kernels.rs` | `Kernels` implementation over the cuTile kernels in `crate::kernels`, with the GDN delta rule on the cuTile or the SIMT backend | `TileKernels::{new, prepare, update, gdn_backend}` |
| `gemm.rs` | bf16 GEMM, fp32 compute, fixed workspace (graph-safe): cuBLASLt with one pinned algorithm per weight shape (`GemmMode::Pinned`, default; batch-invariant) or `cublasGemmEx` (`GemmMode::Heuristic`) | `Blas::{with_mode, gemm_bf16, pinned_algos}`, `GemmMode`, `PINNED_REF_ROWS` |
| `meta.rs` | Per-step batch metadata (incl. answer positions), double-buffered upload | `MetaUploader::{pack, upload}`, `DeviceMeta`, `MetaShape`, `PAD_PARENT` |
| `gpu.rs` | The GPU `DecisionEngine` (bf16 or nvfp4 engine, following the checkpoint) | `GpuEngine::{load, run, step, profile, read_tap, gdn_backend}`, `EngineOptions` (`weights`, `nvfp4_fused`), `split_batch`, `StepInfo`, `parse_gdn_backend`, `gdn_backend_from_env`, `weight_format_from_env` |
| `goldens.rs` | Readers for the model goldens (corpus, runs, hidden dumps) used by tests and `bench_forward` | `model_corpus`, `golden_run`, `hidden_dump`, `plan_corpus`, `batch_of` |
| `platform.rs` | Platform profile written by `hwprobe`; the idle-GPU guard for benchmarks | `PlatformProfile`, `require_idle_gpu`, `gpu_contention` |
| `cpu_ref/` | f32 CPU reference forward over packed trees (tests only; slow) | `CpuRefEngine`, `Layout::{from_batch, flat}`, `check` |

**How `forward::build` lays out one layer** (see the `forward.rs` module docs for the full listing):

- GDN layer: one GEMM `h · [Wqkv; Wz; Wa; Wb]ᵀ` gives a `[T, 12352]` value; later ops read column slices of it
  (`TensorRef::Cols`): conv input `0..8192`, z `8192..12288`, a and b 32 columns each. The conv output's v columns
  (`4096..8192`) feed the scan.
- Attention layer: `qg = h · Wqᵀ` (`[T, 8192]`, per head 256 query | 256 gate), `kv = h · [Wk; Wv]ᵀ`
  (`[T, 2048]`). `seg_attn` reads the gate from `qg` and v from `kv` in place.
- The residual add is fused into the next norm (`GraphOp::AddZcRmsNorm`) and done **in place**: `residual_out`
  aliases `residual`'s storage, and `StepGraph::plan` enforces that nothing reads the old value afterwards.
- GDN intermediates (g, beta, chunk inverse, prefix states; f32 q, k and the intra-chunk scores on the cuTile
  backend) are not graph values: they live in the `GdnScratch` owned by `TileKernels` (`GdnScratch::for_engine`,
  which leaves out the buffers the chosen backend never reads). That is why `GraphOp::GdnPrep` has no outputs and
  `GdnChunkScan` reads the scratch implicitly. `GdnChunkScan` takes the whole `[T, 8192]` conv output: the SIMT
  scan reads q, k and v from it, the cuTile scan only its v columns.
- **GDN backend** (`EngineOptions::gdn_backend`, `--gdn-backend auto|tile|simt`, `$SYSONE_GDN_BACKEND` in the
  engine tests and `bench_forward`): fixed per engine. `None`/`auto` resolves to `GdnBackend::default_for_arch` of
  the device; the resolved backend and the reason are logged at load and exported as
  `sysone_engine_info{engine,gdn_backend,arch}` in `/metrics` (`DecisionEngine::info`; `/health` keeps qev's
  body). With SIMT, `TileKernels::new` loads the SIMT module for the device, so a missing or stale artifact for
  the device's architecture fails the load; nothing falls back to cuTile.

Gotchas (from code comments and the Phase 2a review):
- **`GdnPlan` and the attention work list are rewritten in place** by `TileKernels::update` before each step (the
  fused path's `qk_inv` is per-batch data too, though the engine does not use that path yet). One engine therefore runs one step at a time on one
  stream; the plans cannot be shared across concurrent streams.
- **Stale pad rows:** the arena is shared across buckets and not cleared, so rows past the batch's tokens hold
  leftovers from earlier steps. Nothing reads them into real rows (segment-aware kernels skip the pad segment,
  token-parallel ops are row-local, the readout gathers only answer rows), but do not assume pad rows are zero.
- **`split_batch` assumes `push_request` batches:** it rebuilds sub-batches by finding each prefix's suffixes and
  assumes answers are numbered in suffix-segment order, which is what `push_request` produces.
- `TileKernels::update` refuses a batch with more prefixes than GDN state slots before anything runs: `GdnPlan::update`
  itself does not check the scratch, and a replayed scan would write past the state buffer.
- The engine uses `AttnTiles::default()` at every T on purpose: `AttnTiles::for_tokens` is faster at small T but not
  bit-identical, which would make a request's result depend on how it was batched.
- `TileKernels` calls the GDN kernels one by one, not `gdn::gdn_mixer`. Tile backend: `gdn::tree_conv1d_silu`,
  `gdn::gdn_prep` (stores f32 q, k), `gdn::gdn_chunk_scan`, `gdn::gated_rmsnorm`. SIMT backend:
  `gdn::tree_conv1d_silu`, `gdn::simt::gates`, `gdn::simt::chunk_scan`, `gdn::gated_rmsnorm` (the same launches as
  `gdn_mixer_with(.., Simt)`). P1's fused cuTile conv (`tree_conv1d_silu_inv` + `gdn_chunk_scan_conv`, no stored
  f32 q/k) is not the engine's Tile path: it is not bit-identical to the stored-q/k path (fp32-noise level,
  `kernels_gdn::scan_from_bf16_conv_matches_stored_f32_qk`), and the SIMT backend does not need it (it normalises
  q and k inside its own scan). See docs/E2.md.
- **NVFP4 MLP** (N1): `forward::build_with` emits, per NVFP4 projection, `AddZcRmsNormNvfp4` + `GemmNvfp4` (gate/up)
  and `SiluMulNvfp4` + `GemmNvfp4` (down); with `nvfp4_fused = false` the bf16 `AddZcRmsNorm` / `SiluMul` outputs
  go through a standalone `QuantizeNvfp4`. The packed activations (`U8 [T, K/2]`) and scales (`F8E4M3 [T, K/16]`)
  are arena values. `TileKernels::gemm_nvfp4` runs the `Raw` FP4 GEMM on the checkpoint's packed weights in place,
  with the per-half epilogue computed from the parts' global scales (`RawAlpha`), at fixed tiles for every T. An
  NVFP4 projection outside the MLP is refused, and `GpuEngine::load` checks that the loaded weights match the
  checkpoint profile (every MLP projection NVFP4, or none).
- `GpuEngine`'s field order is its drop order: graphs and buffers must drop before the CUDA context.

### 4.5a `bulk` (`src/bulk/`)

`sysone bulk` (docs/BULK.md): a JSONL file of `/v1/systemone` jobs (or a question set and its documents) in, one
result line per job id out.

| File | Responsibility | Key items |
|---|---|---|
| `job.rs` | Input lines (`id`, `body`/`body_text`, `query`), document lines (`id`, `state`, `query`, `overrides`) and files, result lines, id digests | `parse_line`, `parse_doc_line`, `doc_file`, `Line`, `Job`, `Payload`, `result_line`, `id_key` |
| `output.rs` | The results file: `flock`, resume (done ids), cutting a torn last line, `fdatasync` | `Output::{open, write, sync}`, `Resume` |
| `qset.rs` | Question-set runs: the question file validated once (before any worker), the equivalent full body of a document, planning from the prepared questions (or the full body when a document overrides `model`/`questions`) | `QuestionSet::{load, body, prepare}` |
| `pack.rs` | Choosing a step: first-fit decreasing from the window into the worker's largest bucket; bucket choice at the tail by a throughput curve; ladders; FIFO and bulk packing simulations | `choose`, `Size`, `Throughput`, `PackPolicy`, `ladder`, `simulate_fifo`, `simulate_bulk` |
| `run.rs` | The coordinator: reader (JSONL jobs or documents, or a directory of text files; credits bound the read-ahead), planner threads (`server::prepare`), dispatcher (reorder to input order, window, `choose`, steps to workers, requeue on worker exit), writer (`server::finish`, append); the summary | `run`, `RunOptions`, `Summary`, `WorkerSummary` |
| `worker.rs` | Workers: engine build and sizing per device (`EngineSpec`), the worker process (`sysone bulk worker`) and thread loops, the coordinator's handle, the mock `SimEngine` | `EngineSpec`, `build`, `process_main`, `Handle::{thread, process}`, `Event`, `command` |
| `wire.rs` | The coordinator ↔ worker pipe protocol (length-prefixed frames: HELLO, STEP, DONE, FAILED) | `Frame`, `Hello`, `Done`, `encode`, `decode` |
| `compare.rs` | Two result files by id (statuses, fields, probability differences) | `main`, `walk`, `Diffs` |
| `synth.rs`, `cli.rs` | Synthetic requests (loadgen's) and corpora (also as a question set and documents); the `sysone bulk` subcommands | `body`, `CorpusSpec`, `question_set`, `doc_line`; `main` |

Gotchas:
- Results must be *keyed by id*: steps reorder requests by size and mix them.
- GPU workers are processes started through `scripts/gpu.sh --gpu N`, never engines on several devices in one
  process: the engine uses the default device in several places (docs/BULK.md "Multi-GPU"). A GPU listed twice
  (two engines on one GPU) needs the coordinator to run under gpu.sh already.
- A step is chosen only from a full window or the exhausted input, which makes the step sequence deterministic;
  do not add timing-based flushes without keeping a deterministic mode.
- `EngineOptions::memory_fraction` (set by bulk workers) drops buckets whose arena, GDN scratch and readback do not
  fit the memory free after the weights (`gpu::step_bytes`, `GdnScratch::engine_bytes`).

### 4.6 `kernels` (`src/kernels/`)

GPU kernels written with cuTile (`#[cutile::module]` device code, JIT-compiled through Tile IR on first launch).
Each module's docs state the reference op it reproduces and its rounding points.

| Module | Kernels | Notes |
|---|---|---|
| `norm` | `zc_rmsnorm`, `add_zc_rmsnorm` | Zero-centred RMSNorm `x·rsqrt(mean x²+eps)·(1+w)`, fp32 statistics; the fused variant rounds the new residual to bf16 first, as the reference does. |
| `pointwise` | `silu_mul`, `gather_rows` | `gather_rows` serves the readout gather (and the embedding in the kernel tests). |
| `attention` | `qk_norm_rope`, `seg_attn`, `AttnPlan`, `AttnTiles` | head_dim 256, 16 q / 4 kv heads, partial RoPE on 64 dims; softmax and P kept in fp32 (P fed to the bf16 MMA in `p_split` parts). |
| `gdn` (`mod.rs`, `tile.rs`) | `tree_conv1d_silu`, `gdn_prep`, `gdn_chunk_scan(_with)`, `gated_rmsnorm`, `gdn_mixer`, `GdnPlan`, `GdnScratch`, `GdnPrecision`, `GdnOptions` | Chunked gated delta rule (chunk 64, WY/UT form) in fp32 via split TF32 by default. `tile.rs` documents the chunk and segment tables. |
| `readout` | `slot_logits` | bf16 × bf16 → f32 slot GEMM. Tested and benchmarked, but the engine computes the readout with cuBLAS instead (`GraphOp::Gemm` on `WeightRef::SlotHead`). |
| `nvfp4` | `quantize_nvfp4`, `zc_rmsnorm_nvfp4`, `add_zc_rmsnorm_nvfp4`, `silu_mul_nvfp4`, `gemm_nvfp4(_raw)`, `gemm_raw_op`, `RawAlpha` | compressed-tensors' NVFP4 numerics (module docs list every rounding step); the fused variants equal the bf16 op followed by `quantize_nvfp4` bit for bit. |

All kernel launchers that take raw device pointers are `unsafe fn`: the caller keeps every buffer alive and
unaliased until the returned operation completes.

### 4.7 `tolerances` (`src/tolerances.rs`)

Reads `tests/tolerances.toml` (a small TOML subset) so every test takes its thresholds from one file. `Stats` is the
p50/p99/max summary tests print; `EvalGate` mirrors the statistical eval gate of `reference/eval/stats.py`. Its unit
tests check that the file is internally consistent and that the Rust gate agrees with the Python one.

### 4.8 Outside `src/`

| Path | What it is |
|---|---|
| `tests/` | Integration tests (§6) and goldens (`tests/golden/dev/`), `tests/tolerances.toml`. |
| `benchmarks/s1/` | Benchmark and probe binaries (§7). Outputs go to `benchmarks/results/` (git-ignored). |
| `reference/` | Python harness around the reference (qev @ OneJev 81ce62f + `serve.patch`) that produces every golden; see `reference/README.md` and `reference/eval/README.md`. |
| `scripts/` | `gpu.sh` (per-GPU lock), `doctor.sh` (prerequisites, incl. the pinned cuda-rust revision in `CUDA_RUST_REV`), `x86/` (runners for the x86 host), `tests/` (the scripts' own tests, run by `tests/scripts.rs`). |
| `profiles/` | Platform profiles written by `hwprobe --profile-out` (`sm_121-gb10.json`). |
| `legacy/grout/` | grout's Qwen3 engine, read-only, kept as porting reference. Not built. |

## 5. GPU ops of one forward

The ops below are the classes `bench_forward`'s per-op profile reports (`Kernels::profile_label`, by default
`GraphOp::profile_label`; the four GDN scan stages are sub-marks added by `TileKernels::gdn_chunk_scan`, and the
`gdn.scan` mark before them holds ~0 ms). The GDN rows depend on the engine's GDN backend: `gdn.prep` and
`gdn.chunk_*`/`gdn.scan_*` on the cuTile backend, `gdn.simt_*` on the SIMT backend. "Count" is per forward of elephant-1 (24 GDN layers,
8 attention layers). All activations are bf16 unless stated. With the last layer trimmed (L1, default), that
layer's `gemm.AttnQ`, `gemm.AttnO`, MLP, post-norm and the final norm run on the `[Q, ..]` answer rows of the
bucket (Q = `Bucket::answers`), its `attn.qk_norm_rope` is two launches (`k_norm_rope` over every token,
`qk_norm_rope` over the answer rows) and its `attn.seg_attn` runs one work item per answer row; `readout.gather` is
three gathers before the layer's queries (`hn`, `kv` and the residual) instead of one after the final norm.

| Profile label | Count | Implementation | Precision | Test |
|---|---|---|---|---|
| `embedding` | 1 | cutile-kernels `embedding_batch_f16` (bf16 bits moved as f16), called by `Executor::embedding` | exact gather | `tests/engine_graph.rs`, `tests/kernels_norm.rs` (`gather_rows_and_embedding_are_exact`) |
| `norm.zc_rmsnorm` | 1 + 64 | `kernels::norm::{zc_rmsnorm, add_zc_rmsnorm}` (cuTile): one plain norm (layer 0's input norm), then 64 fused add + norm, each updating the residual in place; the last one applies the final norm | fp32 statistics, bf16 out | `tests/kernels_norm.rs` |
| `gemm.GdnInProj` | 24 | cuBLAS(Lt), `engine::gemm::Blas::gemm_bf16`, weight `[12352, 4096]`; one pinned cuBLASLt algorithm per weight shape at every T (§3.6) | bf16 in, fp32 compute (`CUBLAS_COMPUTE_32F`, no split-K when pinned; no reduced-precision reduction otherwise), bf16 out | `tests/engine_gemm.rs` (accuracy; row invariance when pinned) |
| `gemm.GdnOutProj` | 24 | cuBLAS, `[4096, 4096]` | as above | `tests/engine_gemm.rs` |
| `gemm.AttnQ` | 8 | cuBLAS, `[8192, 4096]` | as above | `tests/engine_gemm.rs` |
| `gemm.AttnKv` | 8 | cuBLAS, `[2048, 4096]` | as above | `tests/engine_gemm.rs` |
| `gemm.AttnO` | 8 | cuBLAS, `[4096, 4096]` | as above | `tests/engine_gemm.rs` |
| `gemm.MlpGateUp` | 32 | cuBLAS, `[24576, 4096]` | as above | `tests/engine_gemm.rs` |
| `gemm.MlpDown` | 32 | cuBLAS, `[4096, 12288]` | as above | `tests/engine_gemm.rs` |
| `gemm.SlotHead` | 1 | cuBLAS, `[n_slots, 4096]` on `[Q, 4096]` rows | fp32 compute, **f32 out** | `tests/engine_gemm.rs`; end to end in `tests/engine_e2e.rs` |
| `gdn.conv1d_silu` | 24 | `kernels::gdn::tree_conv1d_silu` (cuTile) | bf16 in/out, fp32 accumulation | `tests/kernels_gdn.rs` |
| `gdn.prep` | 24 | `kernels::gdn::gdn_prep` (cuTile): l2-normalised q, k and gates g, beta into `GdnScratch` | f32 out | `tests/kernels_gdn.rs` |
| `gdn.chunk_tinv` | 24 | first op of `kernels::gdn::gdn_chunk_scan` (`gdn_chunk_tp`): per-chunk inverse of the unit lower-triangular matrix and masked intra-chunk scores | fp32 via split TF32 (`GdnPrecision::Fp32Split3`) | `tests/kernels_gdn.rs` |
| `gdn.chunk_wy` | 24 | empty slot: the Phase 1 WY launch is now fused into the scans | — | — |
| `gdn.scan_prefix` | 24 | inter-chunk scan over prefix segments; writes each prefix's final state | fp32 state, split TF32 products | `tests/kernels_gdn.rs` |
| `gdn.scan_suffix` | 24 | inter-chunk scan over suffix segments, starting from the parent's state | as above, bf16 out | `tests/kernels_gdn.rs` |
| `gdn.simt_gates` | 24 | SIMT backend: `kernels::gdn::simt::gates` (cuda-oxide), g and beta only (q, k are normalised in the scan) | f32 out | `tests/kernels_gdn.rs` (`GDN_BACKENDS=simt`) |
| `gdn.simt_tinv` | 24 | SIMT backend, first op of `kernels::gdn::simt::chunk_scan`: blocked 16x16 forward substitution after an exact bf16 Gram `k kᵀ` | fp32 SIMT | `tests/kernels_gdn.rs` (`simt_tinv_matches_f64_reference`) |
| `gdn.simt_slot_guard` | 24 | SIMT backend: writes NaN over segments whose state slot is past the scratch (a replay that missed the host check); ~2 us | — | `simt_replay_past_slot_capacity_is_nan` |
| `gdn.simt_scan_prefix` | 24 | SIMT backend: `gdn_simt_scan` phase 0, prefix segments, register-resident state, `mma.sync` with split-bf16 operands | fp32-grade state products (three-way splits), two-way for output-only products | `tests/kernels_gdn.rs` |
| `gdn.simt_scan_suffix` | 24 | SIMT backend: `gdn_simt_scan` phase 1, suffixes from the parent's state | as above, bf16 out | `tests/kernels_gdn.rs` |
| `gdn.gated_rmsnorm` | 24 | `kernels::gdn::gated_rmsnorm` (cuTile), per 128-wide v head, `· w · silu(z)` | fp32 statistics, bf16 out | `tests/kernels_gdn.rs` |
| `attn.qk_norm_rope` | 8 | `kernels::attention::qk_norm_rope` (cuTile), one token per tile block (`ROPE_TOKENS`, L1: position-invariant) | reference rounding points (module docs) | `tests/kernels_attn.rs` |
| `attn.seg_attn` | 8 | `kernels::attention::seg_attn` (cuTile), `AttnTiles::default()` | fp32 scores and softmax, P in 3 bf16 parts (`p_split = 3`), sigmoid gate, bf16 out | `tests/kernels_attn.rs` |
| `pointwise.silu_mul` | 32 | `kernels::pointwise::silu_mul` (cuTile) | bf16 rounding of `silu(g)` and of the product, as the reference | `tests/kernels_norm.rs` |
| `readout.gather` | 1 | `kernels::pointwise::gather_rows` at `answer_rows` | exact | `tests/kernels_norm.rs` |
| `debug.copy` | taps only | `Executor::copy` (device copy) for `Taps` | exact | `tests/engine_layers.rs` uses taps |

The nvfp4 engine (N1) replaces the MLP rows: `nvfp4.add_zc_rmsnorm_nvfp4` (32; residual add + norm + quantize,
`tests/kernels_nvfp4_fused.rs`), `gemm_nvfp4.MlpGateUp` and `gemm_nvfp4.MlpDown` (32 each; block-scaled FP4 MMA,
fp32 accumulate, `1/(a_gs·w_gs)` epilogue, bf16 out; `tests/kernels_nvfp4.rs`), `nvfp4.silu_mul_nvfp4` (32), and
with `--nvfp4-unfused` `nvfp4.quantize_nvfp4` (64) after the bf16 norm and `silu_mul`. `ZcRmsNormNvfp4` exists
in the IR and the kernels but the forward does not need it (layer 0's plain norm feeds the mixer, not the MLP).

Where the time goes at each bucket is in PHASE2A.md ("Forward speed").

## 6. Correctness and testing

### 6.1 Where the expected values come from

- **The reference** is qev (OneJev @ 81ce62f) with elephant-1's `serve.patch`, run through qev's own FastAPI app by
  the scripts in `reference/` (in a venv under `../sysone-ref`, see `reference/README.md`). Nothing in `reference/`
  reimplements the model.
- **Render goldens** (`tests/golden/dev/render/*.jsonl`, written by `reference/render_golden.py`): prompts, token
  ids, owners, limits and error bodies for the 298-case corpus in `reference/corpus/requests.jsonl`.
- **Assemble goldens** (`tests/golden/dev/assemble/`, `reference/assemble_golden.py`): qev's answer assembly on
  synthetic logits, schema error bodies, and Python float built-ins.
- **Model goldens** (`tests/golden/dev/model/runs/<run>.json`, `reference/run_golden.py`): slot logits and
  probabilities of the 52-request model corpus (`reference/corpus_model/requests.jsonl`) in fp32, bf16 and NVFP4,
  over several reference paths (`batched`, `sequential`, `per_question`). Hidden-state dumps for 10 requests are too
  large to commit: they live under `../sysone-ref/golden/dev/`, indexed by `tests/golden/dev/model/manifest.json`.
- **Eval set** (`reference/eval/`, manifest in `tests/golden/dev/eval/`): 278 labeled records from public sources,
  with several reference runs, for the statistical gate.
- **Tolerances** (`tests/tolerances.toml`): generated from measured reference noise by `reference/noise_stats.py
  --write-tolerances` and `reference/eval/stats.py --write-tolerances`, read by `src/tolerances.rs`. Tests never
  hard-code a threshold that belongs in this file.

### 6.2 The gates and the tests that implement them

| Gate (DESIGN §9.2) | What is compared | Test |
|---|---|---|
| Frontend | Prompt bytes, token ids, owners, limits, error bodies vs qev, for 100% of the render corpus | `tests/frontend_golden.rs`, `tests/frontend_assemble.rs`; goldens self-check in `tests/render_golden_shape.rs` |
| Per kernel | Each GPU kernel vs a CPU f64 reference transcribed from transformers 5.19 `modeling_qwen3_5.py`, with the reference's bf16 rounding points | `tests/kernels_norm.rs`, `tests/kernels_attn.rs`, `tests/kernels_gdn.rs`, `tests/kernels_nvfp4.rs` (bit-exact vs compressed-tensors), `tests/kernels_nvfp4_fused.rs` (fused = op then quantize, bit for bit), `tests/engine_gemm.rs` |
| Engine plumbing | StepGraph arena, eager vs captured graphs, metadata upload | `tests/engine_graph.rs`, `tests/engine_loader.rs` |
| Tree semantics (CPU) | Packed tree forward = flat forward with explicit mask; packed vs alone; vs qev fp32 | `src/engine/cpu_ref/tests.rs` (lib, tiny random model), `tests/cpu_ref.rs` (ignored; real weights) |
| (a) Hidden states | Per-row relative error at layers 0, 3, 4, 15, 31 and the final norm vs qev fp32, within 1.5× the bf16 reference's own error (`[model.hidden.*]`) | `tests/engine_layers.rs::hidden_states_vs_fp32_truth` (A1's definition, PHASE2A.md; passes with both GDN backends, E2.md) |
| (b) End to end | Per-option probability \|dp\| vs qev bf16 (`[e2e.bf16]`), also reported and gated vs fp32 (`[e2e.fp32_truth]`); top-1 agreement where the reference margin is large enough. nvfp4 engine (`SYSONE_ENGINE=nvfp4`): vs the NVFP4 QDQ per-question reference (`[nvfp4]`), fused == standalone quantize bit for bit | `tests/engine_e2e.rs::slot_probabilities_match_goldens` (`nvfp4_fused_equals_unfused`); over HTTP: `tests/engine_server.rs` (bf16) |
| Batch invariance | Each request alone vs packed with the others vs alone in a larger bucket (`[batch_invariance]`; nvfp4 engine: `[batch_invariance.nvfp4]`). Since L1 every branch is bit-identical in both engines (pinned GEMMs, one-token RoPE blocks) | `tests/engine_e2e.rs::batch_invariance`; kernel level: `kernels_attn::seg_attn_branch_alone_is_bit_identical`, `kernels_attn::qk_norm_rope_rows_do_not_depend_on_their_position`, `kernels_gdn::tree_batched_matches_each_branch_alone`, `engine_gemm::pinned_gemm_rows_do_not_depend_on_the_batch`; diagnostic: `engine_e2e::packing_diagnostic` (ignored) |
| Last-layer trimming (L1) | Trimmed vs untrimmed logits, bit for bit with pinned GEMMs (heuristic GEMMs reported) | `tests/engine_e2e.rs::trimmed_last_layer_equals_untrimmed`; kernel level: `kernels_attn::seg_attn_answers_match_full_rows`, `kernels_attn::qk_norm_rope_k_ignores_q_and_gathered_rows_match_bits`; diagnostic: `engine_e2e::trim_diagnostic` (ignored) |
| Graph replay | Replay with new metadata = eager, bit for bit | `tests/engine_e2e.rs::graph_replay_equals_eager`, `tests/engine_graph.rs` |
| Eval (statistical) | Accuracy, top-1 agreement, \|dp\| and Brier vs the envelope of the reference's own paths (`[eval.bf16]`, `[eval.nvfp4]`) | `reference/eval/score.py` against a live server; `src/tolerances.rs` tests keep the Rust `EvalGate` equal to `stats.py` |
| Server behaviour | Routes, error bodies, backpressure, timing semantics, `:batch`, debug diagnostics (mock engine) | `tests/server_e2e.rs` |

### 6.3 Running tests

```bash
cargo test --release --lib                      # unit tests; CPU only, no checkpoint needed
cargo test --release --test frontend_golden     # needs the tokenizer + chat template (see below)
scripts/gpu.sh cargo test --release --all-targets            # everything, holding the GPU lock
scripts/gpu.sh cargo test --release --test engine_e2e -- --nocapture --test-threads=1
cargo test --release --test cpu_ref -- --ignored --nocapture # minutes of CPU, real weights
```

- **GPU lock rule:** every command that touches the GPU (GPU tests, benchmarks, reference/golden runs) runs through
  `scripts/gpu.sh`, which holds the GPU's lock (`flock`) and sets `SYSONE_GPU_LOCKED=1`: `/tmp/sysone-gpu.lock` on
  a single-GPU host (GB10), `/tmp/sysone-gpu<N>.lock` per selected GPU on a multi-GPU host (`--gpu N` or
  `CUDA_VISIBLE_DEVICES`; see the script's header and docs/x86/AGENT.md §4).
- **Data prerequisites:** the checkpoint directory is read from `$SYSONE_MODEL_DIR` (frontend, server and engine
  tests; default `/home/velvet/Desktop/hf_models/elephant-1`); `tests/engine_loader.rs` uses `SYSONE_BF16_DIR` /
  `SYSONE_NVFP4_DIR`, and so does `tests/engine_e2e.rs` with `SYSONE_ENGINE=nvfp4`. `tests/engine_layers.rs` needs the hidden-state dumps under `../sysone-ref`. The NVFP4 layer-0
  test needs `reference/nvfp4/data`, which is generated (`reference/nvfp4/gen_layer0.py`), not committed.
- **Tooling tests:** `tests/scripts.rs` runs the shell tests of `scripts/gpu.sh` and `scripts/x86/check_outgoing.sh`
  (no GPU; bash, flock, git); `tests/platform_gpu.rs` checks on a real GPU that CUDA device 0 is the GPU gpu.sh
  selected. `scripts/x86/run_tests.sh` groups every test by its data needs (docs/x86/AGENT.md §7).
- **Ignored tests:** `tests/cpu_ref.rs` (both tests) and `tests/engine_layers.rs::hidden_state_row_profile` (a
  diagnostic driven by `SYSONE_DUMP` / `SYSONE_DUMP_ID`). Run them with `-- --ignored`.
- Some tests skip with a message when their data is missing (`engine_loader`, the chat template unit test); others
  fail (the NVFP4 layer-0 case, the frontend and server tests). Read the test's module docs before trusting a pass.
- `tests/engine_server.rs` targets an already running server when `SYSONE_SERVER_ADDR=host:port` is set.

## 7. Benchmarks and scripts

All GPU benchmark and probe binaries (not `loadgen` or `cpu_ref_check`) call `engine::platform::require_idle_gpu`: they refuse to start while another process holds
a CUDA context, unless `--allow-shared-gpu` is passed (the output is then marked not authoritative). Run them through
`scripts/gpu.sh` so they also hold the lock. Build first with `cargo build --release --bins`.

| Binary (`benchmarks/s1/` unless noted) | What it measures | Typical command |
|---|---|---|
| `bench_forward` | Whole forward (bf16 or nvfp4, following `--model`'s checkpoint; `--nvfp4-unfused`): graph replay and eager time per bucket, plus the per-op profile (§5). Uses the real frontend on synthetic requests. | `scripts/gpu.sh ./target/release/bench_forward --tokens 1024,4096,16384` (flags: `--model`, `--state-tokens`, `--iters`, `--eager-iters`, `--no-profile`, `--gemm pinned\|heuristic`, `--no-trim`, `--tokens-per-answer`) |
| `bench_gdn` | GDN mixer kernels per layer on segment trees, precision and tile variants | `scripts/gpu.sh ./target/release/bench_gdn` |
| `bench_attn` | `seg_attn`, norms, RoPE, pointwise and readout kernels; `--sweep` for tile shapes, `--baseline` for cutile-kernels' dense fmha | `scripts/gpu.sh ./target/release/bench_attn --tokens 1024,4096,16384` |
| `bench_nvfp4` | NVFP4 quantize and FP4 GEMMs at the MLP shapes vs cuBLAS bf16 | `scripts/gpu.sh ./target/release/bench_nvfp4 --tokens 1024,4096,16384` |
| `hwprobe` | Memory bandwidth, cuBLAS bf16 GEMM rates at the model's shapes, GEMM-only forward estimate; `--profile-out` writes a platform profile | `scripts/gpu.sh ./target/release/hwprobe --profile-out profiles/<name>.json` |
| `tile_gemm_probe` | Plain cuTile bf16 and NVFP4 GEMM rates (Phase 0 floor) | `scripts/gpu.sh ./target/release/tile_gemm_probe` |
| `loadgen` | Closed-loop HTTP load against a running server; one JSON line of rates, latency quantiles, timing parts and `/metrics` batch stats (busy fraction over the scrape window, step padding). `--corpus FILE` sends every job of a bulk job file once (`--responses FILE` keeps the answers as bulk result lines) | `./target/release/loadgen --addr 127.0.0.1:8000 --concurrency 64 --state-tokens 1000 --questions 8 --duration-s 60` |
| `sysone bulk` (`src/bulk/`) | Bulk tagging throughput: docs/s, decisions/s, packed tok/s, padding, per-GPU busy fraction and share in its summary; `bulk simulate` compares packings on CPU; `bulk compare` compares result files | `./target/release/sysone bulk --model M --input corpus.jsonl --output out.jsonl` (docs/BULK.md) |
| `cpu_ref_check` (`src/engine/cpu_ref/check_main.rs`) | CPU reference vs the qev fp32 run recorded by `reference/hf_check/run_qev.py` | `cargo run --release --bin cpu_ref_check -- --model <dir> --recorded reference/hf_check/out/qev_fp32.json` |

Scripts:

| Script | Does |
|---|---|
| `scripts/gpu.sh [--gpu N] CMD...` | Runs `CMD` holding the selected GPU's lock (`flock`; `/tmp/sysone-gpu.lock` on a single-GPU host, `/tmp/sysone-gpu<N>.lock` per GPU otherwise; `SYSONE_GPU_LOCK` overrides), with `SYSONE_GPU_LOCKED=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` set to the selection. `require_idle_gpu` only looks at the process's own GPU (CUDA device 0, matched to nvidia-smi by PCI address). |
| `scripts/x86/*` | Runners (`run_tests.sh`, `run_bench.sh`, `run_bulk.sh`) and the pre-push leak check for the x86 host: [docs/x86/AGENT.md](x86/AGENT.md). |
| `scripts/doctor.sh [--simt]` | Checks build and run prerequisites (CUDA toolkit and tileiras, driver, clang headers, rustup, `../cuda-rust`, Docker, HF auth, ...) and prints the fix for each missing one. `--simt` also checks the cuda-oxide toolchain. |
| `reference/*.py`, `reference/eval/*` | Golden and eval-set generation; see their READMEs. GPU runs go through `scripts/gpu.sh`. |

## 8. Conventions for reading the code

**Units and names.**

| Term in code | Meaning |
|---|---|
| `T`, `tokens`, `num_tokens()` | Packed tokens in a step (all prefixes + all suffixes), or a bucket's token capacity. |
| `Q`, `answers`, `num_branches()` | Answer rows = branches = suffix segments in a batch. |
| `S`, `segments` | Prefix + suffix segments (a bucket also counts the pad segment). |
| row | One token's position in the packed stream (a row of every `[T, ...]` activation). |
| bucket | A `graph::Bucket`, the capacities a CUDA graph is planned for. |
| slot / `n_slots` | An answer label's token and its logit column; `MAX_SLOTS = 256`. |
| `prefixes` / state slot | Prefix segments in a step; each needs one 2 MiB GDN state slot (`--max-prefixes`). |

**Layouts.** Activations are row-major `[T, features]`. Weights are row-major `[N, K]` (out × in), and every GEMM is
`y = x · wᵀ` (`gemm.rs` explains how this maps onto column-major cuBLAS). Fused weights are row-concatenated
(`[gate; up]`), so their outputs are column-concatenated and consumers read column slices.

**Dtypes.** bf16 storage everywhere; fp32 for GEMM accumulation, norm statistics, softmax, GDN gates/state/scan and
the slot logits; f64 on the CPU for answer assembly (Python floats) and for test references. Kernel docs name each
point where the reference rounds to bf16, because matching those is what keeps errors inside the reference's own
noise.

**References to the reference.** Comments cite qev/transformers names (`decide()`, `_split`, `_plan_chunks`,
`Qwen3_5GatedDeltaNet`) so you can find the Python counterpart. Phase/workstream tags (W1–W6, I1, P1, P2, E1, S1,
G1) refer to DESIGN.md §11 and the phase reports; "DESIGN §x" and "PLATFORM.md Pn" refer to those documents.

**Unsafe and FFI.** Kept to the GPU boundary:
- `src/kernels/*`: kernel launchers taking raw device pointers are `unsafe fn` (safety contract in each module doc);
  `gdn/tile.rs` and the `#[cutile::module]` blocks are device code.
- `src/engine/kernels.rs`: adapters that turn arena views into kernel arguments; `unsafe impl Send/Sync` for
  `TileKernels`.
- `src/engine/gemm.rs`: cuBLAS calls through cudarc's sys/result bindings.
- `src/engine/graph/tensor.rs`, `meta.rs`, `gpu.rs`: raw device pointers, async memcpys, `unsafe impl Send for
  GpuEngine`.
- `src/engine/loader.rs`: `O_DIRECT` reads and `posix_fadvise`, declared in a local `extern "C"` block (no `libc`
  crate); `src/sched/affinity.rs`: `sched_setaffinity` the same way.

**Errors.** Library code returns `anyhow::Result`. The API layer maps errors to HTTP outcomes: `ApiError` (422/404,
with qev's bodies), `PlanError::Value` (422) vs `PlanError::Internal` (500), queue/pool/body overload (503,
`Outcome::Overloaded` with a reason counted in `/metrics`). An engine error fails every request of that batch with a
500; an engine panic is caught in `sched::worker::run`, and a CPU-task panic surfaces as `PoolError::Panicked`.
Configuration that v1 cannot serve is refused at load (`ModelConfig::check`, `Frontend::load`), never served
differently.

**Owners in module docs.** "Owner: W3b" etc. records which workstream wrote a module; it is history, not access
control.

## 9. Glossary

| Term | Meaning |
|---|---|
| qev | OneJev's reference server and decision runtime (Python). sysone reproduces its API and math. |
| serve.patch | elephant-1's patch to qev that adds the block-bidirectional mask and the `_split` checks. |
| state | The context a request asks about (string or JSON), rendered into the `<state>` block of the prompt. |
| question / branch | A question (`noul`, `choice`, `score`) becomes one or more branches (debias shifts, chunks); each branch is one suffix and one answer row. |
| prefix / suffix | The shared head of all branch prompts (system turn + state block) / one branch's remaining tokens. |
| segment | A contiguous run of the packed stream: a prefix or a suffix (`contract::Segment`). |
| packed tree / `DecisionBatch` | Several requests' prefixes and suffixes in one token stream with a parent table. |
| slot | An answer label (`A`, `B`, ..., then two-letter labels) that is a single token; its logit is read at the answer row. |
| `slot_lm` | The readout: LM-head logits of the slot tokens at the last suffix token. |
| debias | `?debias=k`: score k cyclic shifts of each question's options and average. |
| chunk (frontend) | A slice of a `choice` question with more options than slots, plus a "none of the options listed here" slot. |
| chunk (GDN) | 64 consecutive rows of one segment, the unit of the chunked delta rule. |
| GDN | GatedDeltaNet, the linear-attention mixer of 24 of the 32 layers: causal conv1d, gated delta rule with a `[32,128,128]` f32 state per sequence, gated RMSNorm. |
| WY / UT form | The chunked delta rule's rewrite of a chunk's sequential updates as a triangular solve plus matrix products. |
| `Tinv` | Inverse of a chunk's unit lower-triangular matrix (`gdn.chunk_tinv`). |
| zc_rmsnorm | Zero-centred RMSNorm: weight applied as `1 + w`. |
| T bucket | A fixed token capacity (256, 512, ...) with its own arena plan and CUDA graph. |
| arena | The one device buffer holding every activation of a step, at offsets planned per bucket. |
| tap | A debug copy of the residual stream after a layer (`forward::Taps`). |
| `p_split` | Number of bf16 parts the fp32 attention probabilities are split into for the `P · V` MMA (3 = fp32-grade). |
| split TF32 / 3xTF32 | Emulating an fp32 product with three TF32 products (`GdnPrecision::Fp32Split3`). |
| NVFP4 | 4-bit float (e2m1) weights and activations with an FP8 (e4m3) scale per 16 values and a per-tensor global scale. |
| QDQ | Quantize-dequantize: simulating quantization in higher precision. The NVFP4 checkpoint's reference numbers come from QDQ. |
| W4A4 | 4-bit weights and 4-bit activations (the NVFP4 MLP checkpoint). |
| fork mode | qev's name for how branches reuse the prefix; sysone reports `batched`. |
| gate (a) / (b) | The hidden-state gate and the end-to-end probability gate of DESIGN §9.2 (§6.2 here). |
| goldens | Recorded reference outputs the tests compare against (`tests/golden/dev`). |
| dev / authoritative goldens | Goldens generated on this GB10 vs on the x86 reference box (DESIGN §9.1). |
| GB10 | The DGX Spark GPU (sm_121, unified memory) sysone is developed on. |
| cuTile / Tile IR | The tile-based GPU programming model of cuda-rust's `cutile` crate; kernels are JIT-compiled through `tileiras`. |
| cuda-oxide / SIMT | cuda-rust's thread-level (SIMT) kernel compiler; builds the SIMT GDN delta rule (G1, `kernels-simt/`, docs/G1.md). |

## 10. Open questions

Things that look inconsistent or unclear in the current tree. They are recorded here instead of guessed at.

1. **Default bucket lists differ.** `graph::BucketPlan::DEFAULT_TOKENS` stops at 32768, `EngineOptions::default()`
   at 65536, and the server uses `EngineOptions::buckets_up_to(max_step)`. Which one is meant to be the default?
2. **Platform profile is barely read.** The `platform.rs` docs say the engine and scheduler read the profile to size
   token budgets and buckets. The server does not: its budgets and buckets come from CLI flags. `sysone bulk`
   workers read it (`--profile`) only for the packer's throughput curve; their buckets come from the device's free
   memory (`EngineOptions::memory_fraction`), because the measured forward is flat per token from ~1k up.
3. **Images.** DESIGN §1 says requests containing images get a 422. The schema has no image field and ignores extra
   keys, so an image key would be dropped silently. Is the 422 still intended?
4. **GDN path in the engine.** Answered by E2 (docs/E2.md): the engine runs the SIMT backend's launches
   (`tree_conv1d_silu`, `simt::gates`, `simt::chunk_scan`, `gated_rmsnorm`) or the cuTile unfused path
   (`gdn_prep` stores f32 q/k). P1's fused cuTile conv is still not the engine's Tile path: it is not bit-identical
   to the stored-q/k path, so adopting it would be a numerics change of the fallback needing its own gate runs,
   and on sm_121 the SIMT backend is the faster path anyway. Open only for architectures that stay on Tile.
5. **Slot readout kernel.** `kernels::readout::slot_logits` is tested but unused; the engine runs the readout on
   cuBLAS. Keep both?
6. **`gdn.chunk_wy` profile row** is an empty operation kept for launch-order compatibility; profiles will show it
   with ~0 ms.
