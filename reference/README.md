# reference/: model-side golden harness (W1b)

Python harness around the patched qev reference (OneJev @81ce62f + elephant-1 `serve.patch`), DESIGN.md §9.1–9.2.
It runs with the reference venv, always isolated: `/home/velvet/Desktop/sysone-ref/venv/bin/python -I reference/<x>.py`.

| File | What it does |
|---|---|
| `qevref.py` | Loads the qev `DecisionEngine` (bf16, fp32, or NVFP4 as compressed-tensors QDQ with `run_compressed=False`), TF32 off; the per-question evaluation path; provenance. |
| `corpus_model/build_corpus.py` | Interim model corpus (`requests.jsonl`, 52 requests) until W1a's render corpus is on main. |
| `run_golden.py` | Runs the corpus through qev's own FastAPI app on one path (`batched`, `sequential`, `per_question`) and writes `tests/golden/dev/model/runs/<name>.json`; hidden-state dumps for the 10 `dump_hidden` requests. |
| `layer_dump.py` | Per-layer hidden states (after layers 0, 3, 4, 15, 31 and the final norm) at answer rows and sampled prefix rows. Importable, and a CLI for W4 (`--request` or `--ids`). |
| `noise_stats.py` | Pairwise noise statistics between runs → `tests/golden/dev/model/noise_report.json`; `--write-tolerances` renders `tests/tolerances.toml` (via `tolerances.py`). |
| `write_manifest.py` | sha256/shape manifest of the binary goldens (`tests/golden/dev/model/manifest.json`); `--check` verifies them. |
| `run_all.sh` | Regenerates every run (bf16 + NVFP4 queue, then the fp32 queue; `all-parallel` overlaps them). |

Binary goldens (hidden-state safetensors) live in `/home/velvet/Desktop/sysone-ref/golden/dev/<run>/`, logs in
`.../golden/dev/logs/`. Prerequisites outside the repo: `sysone-ref/OneJev-patched` (installed editable, `--no-deps`,
into the venv) and Python 3.12 headers extracted to `sysone-ref/pydev/` for Triton (compressed-tensors' FP4 QDQ
kernels); `qevref.py` points `C_INCLUDE_PATH` there.

Reproduce: `bash reference/run_all.sh && venv/bin/python -I reference/noise_stats.py --write-tolerances &&
venv/bin/python -I reference/write_manifest.py`.

## Caveats for consumers (W3/W4)

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
