#!/usr/bin/env bash
# Run the labeled eval set through the sysone bf16 server over HTTP, in GPU-locked chunks, then merge and score.
# Each chunk is ONE scripts/gpu.sh job: start `sysone --engine bf16` (default production config), wait for
# /health (the server only listens once the engine is loaded and warmed up), POST records for up to
# CHUNK_SECONDS with run_sysone.py, stop the server, release the lock. Loops until every record is in a part.
#   bash reference/eval/run_sysone.sh                 # name sysone_bf16
#   NAME=sysone_bf16_x CHUNK_SECONDS=300 bash reference/eval/run_sysone.sh
# Then: python3 -I reference/eval/score.py --run-file tests/golden/dev/eval/candidates/sysone_bf16.json --tier bf16
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
REPO=$(cd "$HERE/../.." && pwd)
NAME=${NAME:-sysone_bf16}
PORT=${PORT:-8766}
MODEL=${SYSONE_MODEL_DIR:-/home/velvet/Desktop/hf_models/elephant-1}
BIN=${SYSONE_BIN:-$REPO/target/release/sysone}
REF=${SYSONE_REF:-/home/velvet/Desktop/sysone-ref}
PARTS=$REF/golden/dev/eval/candidates/$NAME
LOG=$REF/golden/dev/eval/logs
CHUNK_SECONDS=${CHUNK_SECONDS:-420}
# Server flags: the defaults of src/main.rs, spelled out so they are recorded. Buckets are
# EngineOptions::buckets_up_to(max(token_budget, max_request_tokens)) = 256, 512, ..., 65536, CUDA graphs on.
SERVER_ARGS=(--engine bf16 --model "$MODEL" --listen "127.0.0.1:$PORT" --token-budget 16384 --max-wait-us 2000
             --max-request-tokens 65536 --max-branch-tokens 32768 --max-prefixes 256)

if [ "${1:-}" = "__chunk" ]; then  # runs under the GPU lock
  t0=$(date +%s.%N)
  "$BIN" "${SERVER_ARGS[@]}" >>"$LOG/$NAME.server.log" 2>&1 &
  SRV=$!
  trap 'kill -INT $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
  for _ in $(seq 1 900); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "server died; see $LOG/$NAME.server.log"; exit 3; }
    sleep 1
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server not healthy after 900 s"; exit 3; }
  load=$(python3 -c "import time; print(round(time.time() - $t0, 1))")
  python3 -I - "$PARTS/build.json" "$load" <<'EOF'
import json, sys
p = sys.argv[1]; b = json.load(open(p)); b["provenance"]["load_seconds"] = float(sys.argv[2])
json.dump(b, open(p + ".chunk", "w"))
EOF
  python3 -I "$HERE/run_sysone.py" --name "$NAME" --server "http://127.0.0.1:$PORT" \
    --build "$PARTS/build.json.chunk" --max-seconds "$CHUNK_SECONDS"
  exit 0
fi

mkdir -p "$PARTS" "$LOG"
[ -x "$BIN" ] || { echo "build first: cargo build --release --bin sysone"; exit 1; }
# Engine build, recorded in every part (parts with a different build refuse to merge).
SHA=$(git -C "$REPO" rev-parse HEAD)
DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no -- src Cargo.toml Cargo.lock | wc -l)
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
if [ ! -f "$PARTS/model_sha256" ]; then sha256sum "$MODEL/model.safetensors" | cut -d' ' -f1 >"$PARTS/model_sha256"; fi
python3 -I - "$PARTS/build.json" "$SHA" "$DIRTY" "$BIN_SHA" "$(cat "$PARTS/model_sha256")" "$MODEL" "${SERVER_ARGS[*]}" <<'EOF'
import json, subprocess, sys
out, sha, dirty, bin_sha, model_sha, model, server_args = sys.argv[1:]
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
b = {"config": {"engine": "sysone GpuEngine bf16 (bf16 weights/activations, f32 slot readout)", "dtype": "bfloat16",
                "model": model.rstrip("/").split("/")[-1],
                "buckets": [256 << i for i in range(9)], "cuda_graphs": True,
                "token_budget": 16384, "max_wait_us": 2000, "max_prefixes": 256,
                "server_args": server_args.replace(model, "<model>")},
     "provenance": {"tier": "dev (GB10)", "git_sha": sha, "src_dirty_files": int(dirty), "binary_sha256": bin_sha,
                    "gpu": gpu, "model_files_sha256": {"model.safetensors": model_sha}}}
json.dump(b, open(out, "w"), indent=1)
EOF
for i in $(seq 1 40); do
  out=$("$REPO/scripts/gpu.sh" bash "$0" __chunk 2>&1 | tee -a "$LOG/$NAME.log" | tail -1)
  echo "[$NAME] $out"
  case "$out" in
    *"complete ("*|*" 0 left") break ;;
    wrote*) ;;
    *) echo "[$NAME] unexpected end of chunk; see $LOG/$NAME.log" >&2; exit 1 ;;
  esac
done
python3 -I "$HERE/run_sysone.py" --name "$NAME" --merge
