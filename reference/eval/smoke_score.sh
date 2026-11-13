#!/usr/bin/env bash
# End-to-end smoke test of score.py against a real qev server: start serve_ref.py, wait for /health, score a
# subset, stop the server. Run the whole thing under ONE GPU lock (it is one GPU job):
#   scripts/gpu.sh bash reference/eval/smoke_score.sh <tier> <serve_ref args...> -- <score.py args...>
#   scripts/gpu.sh bash reference/eval/smoke_score.sh nvfp4 --nvfp4 --path batched -- --limit 40
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
PY=${PY:-/home/velvet/Desktop/sysone-ref/venv/bin/python}
PORT=${PORT:-8765}
TIER=$1; shift
SERVE=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SERVE+=("$1"); shift; done; shift || true
"$PY" -I "$HERE/serve_ref.py" --port "$PORT" "${SERVE[@]}" >"${LOG:-/dev/null}" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; exit 3; }
  sleep 1
done
curl -s "http://127.0.0.1:$PORT/health"; echo
python3 -I "$HERE/score.py" --server "http://127.0.0.1:$PORT" --tier "$TIER" "$@"
