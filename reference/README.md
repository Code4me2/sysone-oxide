# reference/: the Python reference harness

The reference is OneJev's `qev` runtime at `81ce62f` with elephant-1's `serve.patch` (DESIGN §4). This directory
holds the scripts that run it to produce goldens. Nothing here reimplements the reference: every golden comes
from OneJev's own code paths.

| File | What it does |
|---|---|
| `setup.sh` | Builds `sysone-ref/OneJev-patched` (a copy of `sysone-ref/OneJev` with `serve.patch` applied by `git apply`) and installs it with `pip install -e --no-deps` into `sysone-ref/venv`. Idempotent: an existing copy is checked byte for byte against a fresh application of the patch. The unpatched copy is never touched. |
| `qev_shim.py` | `ShimEngine`, a `qev.engine.DecisionEngine` with no weights (tokenizer, config and `readout.json` only). Only the two forward methods are replaced. |
| `corpus/generate.py` | Deterministic corpus generator (fixed seed). Writes `corpus/requests.jsonl`. |
| `corpus/requests.jsonl` | 298 cases (format below). |
| `render_golden.py` | Writes the render ("frontend") goldens to `tests/golden/dev/render/`. `--check` regenerates them into a temp dir and fails unless the output is byte-identical. `--only id,id` prints records for debugging. |

```sh
bash reference/setup.sh
PY=/home/velvet/Desktop/sysone-ref/venv/bin/python
$PY -I reference/corpus/generate.py     # ~5 s; rewrites corpus/requests.jsonl
$PY -I reference/render_golden.py       # ~13 s, ~1.2 GB RSS, CPU only; rewrites tests/golden/dev/render/
$PY -I reference/render_golden.py --check
cargo test --test render_golden_shape   # goldens vs the contract (DecisionBatch) and their own counts
```

Both scripts are byte-deterministic on rerun. The corpus depends on the tokenizer only for the exact-boundary limit
cases.

## How the render goldens are produced

Each case is POSTed to `qev.server.create_app(engine, "elephant-1")` through Starlette's `TestClient`, as
`/v1/systemone?debias=<debias>`. The engine is the shim. So these all run unchanged, as reference code:

- pydantic schema validation and its 422 detail;
- model routing (404);
- `decide()`'s branch planning:
  - debias cyclic shifts, `min(debias, k)`, with one branch for k = 1;
  - `_plan_chunks` plus the `"none of the options listed here"` slot;
  - the owner list;
- `_prefix_and_suffix_ids`: the marker render, then the longest common token prefix;
- the two limit checks and their messages;
- serve.patch's `_split`: the noncausal check "prefix block ends at token X but the shared cache at P", and the
  option token spans;
- the response's `usage` and `qev` meta.

The shim's forward reads `decide()`'s locals from its caller frame (`rendered`, `owners`, `prefix_ids`, `suffixes`,
`longest`, `total`), runs `_split` (the first thing the real forwards do), and returns zero logits. When `decide()`
raises before the forward (the limit checks), the plan is taken from the traceback frame instead. Answers built
from zero logits are meaningless, so they are dropped. Real answers and probabilities belong to the model goldens
(W1, later).

## Corpus format (`corpus/requests.jsonl`)

One JSON object per line, UTF-8 and not ASCII-escaped. Split lines on `\n` only: strings contain U+2028/U+2029,
which Python's `str.splitlines()` also splits on.

```json
{"id": "choice-004", "group": "choice", "debias": 1, "body": {...}, "engine_direct": true, "note": "..."}
```

- `body` is the exact POST body. Object key order is significant: it sets option and question order, so parse with
  order preservation (serde_json `preserve_order`).
- `engine_direct` is optional.
- `note` is optional.

Groups:

| Group | Cases | What it covers |
|---|---|---|
| `noul` | 32 | default and custom criteria: missing, null, partial, empty, object/array, extra keys |
| `choice` | 39 | 1–25 options, null or empty instructions, odd option names (empty, case, spaces, colon, newline, emoji, the chunk none-option name), empty or null descriptions |
| `choice_many` | 16 | 26, 27 (first "Answer with one label"), 28, 40, 52, 100, 128, 200, 254, 255 options; then 256, 300 and 511, which the schema rejects (`engine_direct`) |
| `score` | 17 | 1–10 levels, entries that are empty, null, padded, object or array |
| `mixed` | 53 | 2–8 questions of every kind, 40 nouls, unsorted question ids, duplicate questions |
| `medium` | 30 | states of ~10 to ~2.5k tokens (text and JSON) with 1–4 questions |
| `states` | 20 | null, absent, empty and whitespace strings, numbers, bools, `[]`, `{}`, deep nesting, float/int formatting (`1e+21`, `5e-324`, `-0.0`, 2^64+1), escapes and control characters, odd keys |
| `unicode` | 19 | CJK, Arabic/Hebrew with RTL marks, Devanagari, Thai, emoji ZWJ/flags/skin tones, non-BMP math, NFD vs NFC, zero-width characters, BOM, NBSP, U+2028 |
| `hostile` | 6 | chat-template tokens, a fake `</state>`, a fake question header, control characters, CR/CRLF, and `QEV_PREFIX_MARK` inside the state (triggers serve.patch's noncausal prefix error) |
| `whitespace` | 17 | leading/trailing spaces, newlines and tabs in states, instructions, names and descriptions (`render_entry` strips entries, not names) |
| `entries` | 12 | instructions and criteria as objects and arrays (`json.dumps(indent=2)`), empty containers, floats inside entries |
| `long` | 17 | string states of ~10 to ~15k tokens, JSON states of ~0.5k to ~15k tokens, and a 15k state with a 255-option choice and debias 3 |
| `limits` | 7 | branch at exactly 32768 (accepted) and 32769; request at exactly 65536 (accepted) and 65537; over the request limit only through debias 3; over both limits |
| `invalid` | 13 | schema errors, unknown model (404), alias and served-name model fields |

`debias` is 3 for 36 cases and 1 for the rest.

## Golden format (`tests/golden/dev/render/<group>.jsonl`)

One record per corpus case, in corpus order, UTF-8 (split lines on `\n` only):

```text
id, group, debias
request_sha256    sha256 of the corpus line (without its newline)
http              {status, body}: the reference server's response. For 200 the body keeps
                  model, usage, and qev = {prompt_version, prompt_sha256, fork_mode, prefix_tokens, branches};
                  answers and timing are dropped. For 404/422 it is the full JSON error body.
render            null when the request never reached decide() (schema error, 404). Otherwise:
  prefix_ids, prefix_tokens
  longest, total        the values decide() compared with the limits
  split_ran             whether serve.patch's _split ran (false when a limit check fired first)
  error                 "ValueError: <msg>" when decide() raised, else null
  prompt_head, prompt_tail
                        every branch prompt (chat template applied) is exactly
                        prompt_head + branch.suffix_text + prompt_tail; checked at generation
  branches[]            in decide() order:
    qid, kind
    chunk               chunk index, or null when not chunked
    order               option order: the original index of each slot
    labels, n_slots, descriptions
    suffix_text         RenderedQuestion.suffix
    option_char_spans   serve.patch: each option line's character span inside suffix_text
    suffix_ids          the branch's token ids after the shared prefix
    option_token_spans  _split's per-option token spans relative to the suffix
                        (absent when _split did not run)
engine_direct     only for engine_direct cases: {outcome, render}. The same render, made by calling decide()
                  with an unvalidated request (pydantic model_construct). outcome is
                  {ok: true, usage, qev} or {ok: false, error}.
```

`provenance.json` records:

- versions (python, transformers, tokenizers, torch, pydantic, fastapi, ...);
- the OneJev commit;
- sha256 of `serve.patch`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `config.json` and
  `readout.json`;
- the corpus sha256 and the generator scripts' sha256;
- the limits, `max_slots`, and the slot table (label → token id, all 256 single tokens);
- per-file case counts and sha256.

It contains no timestamps, so a rerun is byte-identical.

## Reference behaviours worth knowing (all visible in the goldens)

- The schema caps a choice at 255 options (`MAX_CHOICE_OPTIONS`), and elephant-1's tokenizer has all 256 slot
  labels as single tokens (`max_slots` = 256). So chunking (`_plan_chunks`) is unreachable through the HTTP API.
  Choices with 256 or more options get a pydantic 422. The `engine_direct` records show the chunk plan the
  engine would use: 300 options become 2×150, and 511 become 171/171/169 (each chunk plus the none slot). The 256-option case is not
  chunked.
- pydantic's union errors list one entry per question type (`NoulQuestion`, `function-after[_check(),
  ChoiceQuestion]`, …). The 422 `detail` reproduces that list.
- `_prefix_and_suffix_ids` locates the marker with `str.index`, so a state that contains the literal
  `QEV_PREFIX_MARK` shortens the prefix. serve.patch's `_split` then refuses the request (`hostile-006`).
- The branch limit is checked before the request limit, and both before `_split`.
- `usage.input_tokens` is `total` (prefix + every suffix, debias shifts included).

---

## Model-side golden harness (W1b)

Python harness around the patched qev reference (OneJev @81ce62f + elephant-1 `serve.patch`), DESIGN.md §9.1–9.2.
It runs with the reference venv, always isolated: `/home/velvet/Desktop/sysone-ref/venv/bin/python -I reference/<x>.py`.

| File | What it does |
|---|---|
| `qevref.py` | Loads the qev `DecisionEngine` (bf16, fp32, or NVFP4 as compressed-tensors QDQ with `run_compressed=False`), TF32 off; the per-question evaluation path; provenance. |
| `corpus_model/build_corpus.py` | Interim model corpus (`requests.jsonl`, 52 requests) until W1a's render corpus is on main. |
| `run_golden.py` | Runs the corpus through qev's own FastAPI app on one path (`batched`, `sequential`, `per_question`) and writes `tests/golden/dev/model/runs/<name>.json`; hidden-state dumps for the 10 `dump_hidden` requests. |
| `layer_dump.py` | Per-layer hidden states (after layers 0, 3, 4, 15, 31 and the final norm) at answer rows and sampled prefix rows. Importable, and a CLI for W4 (`--request` or `--ids`). |
| `hidden_dense.py` | Gate (a) (Phase 2a A1): dense per-layer hidden dumps of the 10 `dump_hidden` requests (every prefix row, or every `P // 1024`-th row, plus every answer row: 3,829 rows per tap) for eight runs (fp32/bf16 served, bf16 per-question, three hidden-basis permutations of bf16 served as null realizations, two degraded-norm bf16 paths); their manifest; and the gate's validation (`gate`): null candidates pass, synthetic regressions fail → `tests/golden/dev/model/hidden_gate_report.json` and the `[model.hidden]` block of `tests/tolerances.toml`. |
| `noise_stats.py` | Pairwise noise statistics between runs → `tests/golden/dev/model/noise_report.json`; `--write-tolerances` renders `tests/tolerances.toml` (via `tolerances.py`). |
| `write_manifest.py` | sha256/shape manifest of the binary goldens (`tests/golden/dev/model/manifest.json`); `--check` verifies them. |
| `run_all.sh` | Regenerates every run (bf16 + NVFP4 queue, then the fp32 queue; `all-parallel` overlaps them). |

Binary goldens (hidden-state safetensors) live in `/home/velvet/Desktop/sysone-ref/golden/dev/<run>/`, logs in
`.../golden/dev/logs/`. Prerequisites outside the repo: `sysone-ref/OneJev-patched` (installed editable, `--no-deps`,
into the venv) and Python 3.12 headers extracted to `sysone-ref/pydev/` for Triton (compressed-tensors' FP4 QDQ
kernels); `qevref.py` points `C_INCLUDE_PATH` there.

Dense hidden dumps for gate (a) (`<sysone-ref>/golden/dev/hidden_dense/<run>/`, manifest
`tests/golden/dev/model/hidden_dense_manifest.json`), one GPU job per run:

```sh
for r in fp32_served bf16_served bf16_perq bf16_norm_llama bf16_norm_bf16 bf16_perm1 bf16_perm2 bf16_perm3; do
  scripts/gpu.sh $PY -I reference/hidden_dense.py dump --run $r; done     # ~2-6 min each
$PY -I reference/hidden_dense.py manifest && $PY -I reference/hidden_dense.py manifest --check
$PY -I reference/hidden_dense.py gate --write-tolerances                   # CPU, ~1 min
```

`noise_stats.py --write-tolerances` keeps the marked `[eval]` and `[model.hidden]` blocks other generators own.

Reproduce: `bash reference/run_all.sh && venv/bin/python -I reference/noise_stats.py --write-tolerances &&
venv/bin/python -I reference/write_manifest.py`.

### Caveats for consumers (W3/W4)

- **Hidden-state goldens are not per fork path.** `run_golden.py` always dumps hidden states through
  `layer_dump.dump_tree(mode="served")`, which forks the prefix cache *sequentially*. The files under `bf16_batched/`
  and `bf16_sequential/` (and `fp32_batched/` / `fp32_sequential/`) are therefore bit-identical reruns of the same
  sequential path (`noise_report.json` `hidden.bf16_rerun` = 0.0), not batched-path hidden states. Only slot
  logits/probabilities/response JSON differ between the `*_batched` and `*_sequential` runs.
- **`qev.fork_mode = "per_question"`** in the per-question runs' recorded responses is a harness label; the real
  server only emits `batched` or `sequential`. Ignore that field when diffing response JSON against sysone.
- **Token ids** are stored per branch only as sha256 (`prefix_sha256`, `suffix_sha256`); the actual ids are in the
  10 hidden-state dumps. Render goldens (prompt strings, ids) are W1a's.
- **Memory figures** (`peak_gpu_gb` in each run's provenance) are `torch.cuda.max_memory_allocated() / 2**30`,
  i.e. GiB allocated, excluding the CUDA context and allocator reserve. The fp32 peak is 39.37 GiB (42.3 GB).
- **Tolerances loader**: `src/tolerances.rs` already parses `tests/tolerances.toml` and provides the `Stats`
  (p50/p99/max) summary; reuse it rather than writing a second loader. Its `repository_tolerances_are_consistent`
  test checks every threshold equals `factor x measured` (or its `floor_` value) to rounding.
