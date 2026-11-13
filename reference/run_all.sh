#!/usr/bin/env bash
# Regenerate every W1b model-side dev golden run (DESIGN.md §9.1). Correctness only: no timing, no GPU lock.
# Two queues: bf16-sized processes (~19-23 GiB peak allocated) and fp32 (up to 39.4 GiB = 42.3 GB allocated,
# plus CUDA context). They run one after the other by default; `all-parallel` overlaps them (~62 GiB) and is
# only for a box with nothing else on the GPU. Logs go next to
# the binary goldens. Usage: bash reference/run_all.sh [bf16|fp32|all|all-parallel]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/velvet/Desktop/sysone-ref/venv/bin/python}
LOGS=/home/velvet/Desktop/sysone-ref/golden/dev/logs
mkdir -p "$LOGS"
run() { local name=$1; shift; echo "== $name"; "$PY" -I reference/run_golden.py --name "$name" "$@" >"$LOGS/$name.log" 2>&1; tail -1 "$LOGS/$name.log"; }

bf16_queue() {
  run bf16_batched --dtype bfloat16 --path batched
  run bf16_sequential --dtype bfloat16 --path sequential
  run bf16_perq --dtype bfloat16 --path per_question --no-hidden
  # hidden states of the per-question path, only for the dump requests (bf16 baseline of the §3.1 check)
  local ids
  ids=$("$PY" -I -c "import json; print(','.join(json.loads(l)['id'] for l in open('reference/corpus_model/requests.jsonl') if json.loads(l)['dump_hidden']))")
  run bf16_perq_hidden --dtype bfloat16 --path per_question --only "$ids"
  # NVFP4 QDQ reference: per-question evaluation path, 3 fresh processes
  run nvfp4_perq_r1 --nvfp4 --path per_question
  run nvfp4_perq_r2 --nvfp4 --path per_question --no-hidden
  run nvfp4_perq_r3 --nvfp4 --path per_question --no-hidden
  # NVFP4 through the shared-prefix served path (DESIGN §3.1 equivalence claim)
  run nvfp4_batched --nvfp4 --path batched --no-hidden
  run nvfp4_sequential --nvfp4 --path sequential
}

fp32_queue() {
  # Full corpus: measured peak 39.37 GiB allocated (torch max_memory_allocated; 42.3 GB) at a 16k-token prefix
  # (fp32 weights alone are ~33 GiB). On a box where
  # that is too much, set FP32_MAX_PREFIX; dropped requests are listed in each run's "dropped" field.
  local cap=()
  [[ -n "${FP32_MAX_PREFIX:-}" ]] && cap=(--max-prefix-tokens "$FP32_MAX_PREFIX")
  run fp32_batched --dtype float32 --path batched "${cap[@]}"
  run fp32_sequential --dtype float32 --path sequential "${cap[@]}"
}

case "${1:-all}" in
  bf16) bf16_queue ;;
  fp32) fp32_queue ;;
  all) bf16_queue; fp32_queue ;;
  all-parallel) bf16_queue & fp32_queue & wait ;;
  *) echo "usage: $0 [bf16|fp32|all|all-parallel]" >&2; exit 2 ;;
esac
