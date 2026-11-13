# tests/golden/dev/eval/

Labeled eval set and statistical-gate data (DESIGN §9.2); see `reference/eval/README.md`.

- `set.json`: the set manifest (records, gold labels, token counts, request and source sha256s).
- `runs/<path>.json`: answer-level probability vectors of each qev reference path (4 NVFP4 QDQ, 3 bf16, 1 fp32 truth).
- `manifest.json`: sha256 of the full-logit part files kept in `sysone-ref/golden/dev/eval/runs/`.
- `report.json`: `reference/eval/stats.py` output (accuracies, CIs, envelopes, thresholds, gate validation).
- `smoke/`: a 28-record `score.py` run against a live qev NVFP4 server (test of the scorer, not a reference).
