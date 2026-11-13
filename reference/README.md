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
| `run_all.sh` | Regenerates every run (bf16 + NVFP4 queue and fp32 queue side by side). |

Binary goldens (hidden-state safetensors) live in `/home/velvet/Desktop/sysone-ref/golden/dev/<run>/`, logs in
`.../golden/dev/logs/`. Prerequisites outside the repo: `sysone-ref/OneJev-patched` (installed editable, `--no-deps`,
into the venv) and Python 3.12 headers extracted to `sysone-ref/pydev/` for Triton (compressed-tensors' FP4 QDQ
kernels); `qevref.py` points `C_INCLUDE_PATH` there.

Reproduce: `bash reference/run_all.sh && venv/bin/python -I reference/noise_stats.py --write-tolerances &&
venv/bin/python -I reference/write_manifest.py`.
