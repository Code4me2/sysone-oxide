#!/usr/bin/env bash
# Run one eval-set path to completion in GPU-locked chunks (each `scripts/gpu.sh` call loads the model, runs
# ~8 minutes of records, writes a part file and releases the lock), then merge the parts.
#   bash reference/eval/run_chunks.sh nvfp4_perq --nvfp4 --path per_question
#   bash reference/eval/run_chunks.sh bf16_batched --path batched
set -euo pipefail
NAME=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd -P)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-/home/velvet/Desktop/sysone-ref/venv/bin/python}
LOG=${SYSONE_REF:-/home/velvet/Desktop/sysone-ref}/golden/dev/eval/logs
mkdir -p "$LOG"
for i in $(seq 1 40); do
  out=$("$REPO/scripts/gpu.sh" "$PY" -I "$HERE/run_eval.py" --name "$NAME" "$@" 2>&1 | tee -a "$LOG/$NAME.log" | tail -1)
  echo "[$NAME] $out"
  case "$out" in
    *"complete ("*|*" 0 left") break ;;
    wrote*) ;;
    *) echo "[$NAME] unexpected end of chunk; see $LOG/$NAME.log" >&2; exit 1 ;;
  esac
done
"$PY" -I "$HERE/run_eval.py" --name "$NAME" --merge
