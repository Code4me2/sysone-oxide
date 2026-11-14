#!/usr/bin/env bash
# Run the labeled eval set through the sysone server over HTTP, in GPU-locked chunks, then merge and score.
# Each chunk is ONE scripts/gpu.sh job: start `sysone --engine $ENGINE` (default production config), wait for
# /health (the server only listens once the engine is loaded and warmed up), POST records for up to
# CHUNK_SECONDS with run_sysone.py, stop the server, release the lock. Loops until every record is in a part.
#   bash reference/eval/run_sysone.sh                 # name sysone_bf16
#   NAME=sysone_bf16_x CHUNK_SECONDS=300 bash reference/eval/run_sysone.sh
#   NAME=sysone_smoke RUN_ARGS='--limit 3' bash reference/eval/run_sysone.sh   # harness smoke (subset)
#   NAME=sysone_bf16_simt GDN_BACKEND=simt bash reference/eval/run_sysone.sh     # GDN backend (tile|simt|auto)
#   NAME=sysone_nvfp4 ENGINE=nvfp4 bash reference/eval/run_sysone.sh             # nvfp4 engine (N1); the model
#        defaults to the NVFP4 checkpoint and the limits to its profile (16,384 per branch, 65,536 per request)
#   NAME=sysone_nvfp4_c1 ENGINE=nvfp4 FP4_GEMM=cublaslt bash reference/eval/run_sysone.sh   # down GEMM on cuBLASLt (C1)
# Then: python3 -I reference/eval/score.py --run-file tests/golden/dev/eval/candidates/sysone_bf16.json --tier bf16
#       (nvfp4: --tier nvfp4)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd -P)
REPO=$(cd "$HERE/../.." && pwd)
ENGINE=${ENGINE:-bf16}
NAME=${NAME:-sysone_$ENGINE}
PORT=${PORT:-8766}
case "$ENGINE" in
  bf16) MODEL=${SYSONE_MODEL_DIR:-/home/velvet/Desktop/hf_models/elephant-1}; MAX_BRANCH=32768 ;;
  nvfp4) MODEL=${SYSONE_NVFP4_DIR:-/home/velvet/Desktop/hf_models/elephant-1-nvfp4-mlp}; MAX_BRANCH=16384 ;;
  *) echo "ENGINE must be bf16 or nvfp4"; exit 1 ;;
esac
BIN=${SYSONE_BIN:-$REPO/target/release/sysone}
REF=${SYSONE_REF:-/home/velvet/Desktop/sysone-ref}
PARTS=$REF/golden/dev/eval/candidates/$NAME
LOG=$REF/golden/dev/eval/logs
CHUNK_SECONDS=${CHUNK_SECONDS:-420}
GDN_BACKEND=${GDN_BACKEND:-auto}
# Server flags: the defaults of src/main.rs and the checkpoint profile, spelled out so they are recorded. Buckets
# are EngineOptions::buckets_up_to(max(token_budget, max_request_tokens)) = 256, 512, ..., 65536, CUDA graphs on.
SERVER_ARGS=(--engine "$ENGINE" --model "$MODEL" --listen "127.0.0.1:$PORT" --token-budget 16384 --max-wait-us 2000
             --max-request-tokens 65536 --max-branch-tokens "$MAX_BRANCH" --max-prefixes 256 --gdn-backend "$GDN_BACKEND")
# FP4_GEMM=cutile|cublaslt (nvfp4 engine, C1): passed as --fp4-gemm, so it is recorded with the other server flags.
if [ -n "${FP4_GEMM:-}" ]; then SERVER_ARGS+=(--fp4-gemm "$FP4_GEMM"); fi

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
  # The backend the engine resolved (also in the server log): recorded per part, so parts that ran different
  # backends are visible.
  info=$(curl -sf "http://127.0.0.1:$PORT/metrics" | grep '^sysone_engine_info' || true)
  echo "$info" >>"$LOG/$NAME.server.log"
  load=$(python3 -c "import time; print(round(time.time() - $t0, 1))")
  python3 -I - "$PARTS/build.json" "$load" "$info" <<'EOF'
import json, re, sys
p = sys.argv[1]; b = json.load(open(p)); b["provenance"]["load_seconds"] = float(sys.argv[2])
m = re.search(r'gdn_backend="([a-z]+)"', sys.argv[3])
b["config"]["gdn_backend_resolved"] = m.group(1) if m else None
json.dump(b, open(p + ".chunk", "w"))
EOF
  python3 -I "$HERE/run_sysone.py" --name "$NAME" --server "http://127.0.0.1:$PORT" \
    --build "$PARTS/build.json.chunk" --max-seconds "$CHUNK_SECONDS" ${RUN_ARGS:-}
  exit 0
fi

mkdir -p "$PARTS" "$LOG"
[ -x "$BIN" ] || { echo "build first: cargo build --release --bin sysone"; exit 1; }
# Engine build, recorded in every part (parts with a different build refuse to merge).
SHA=$(git -C "$REPO" rev-parse HEAD)
DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no -- src Cargo.toml Cargo.lock | wc -l)
BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
# Weight files: model.safetensors, or every shard of model.safetensors.index.json (the NVFP4 checkpoint), as JSON.
if [ ! -f "$PARTS/model_sha256.json" ]; then
  python3 -I - "$MODEL" >"$PARTS/model_sha256.json" <<'EOF'
import hashlib, json, pathlib, sys
d = pathlib.Path(sys.argv[1])
idx = d / "model.safetensors.index.json"
files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
out = {}
for f in files:
    h = hashlib.sha256()
    with open(d / f, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    out[f] = h.hexdigest()
print(json.dumps(out))
EOF
fi
python3 -I - "$PARTS/build.json" "$SHA" "$DIRTY" "$BIN_SHA" "$PARTS/model_sha256.json" "$MODEL" "${SERVER_ARGS[*]}" "$ENGINE" <<'EOF'
import json, subprocess, sys
out, sha, dirty, bin_sha, model_sha, model, server_args, engine = sys.argv[1:]
model_sha = json.load(open(model_sha))
desc = {"bf16": "sysone GpuEngine bf16 (bf16 weights/activations, f32 slot readout)",
        "nvfp4": "sysone GpuEngine nvfp4 (NVFP4 W4A4 MLP GEMMs, quantize fused; everything else bf16; f32 slot readout)"}
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
b = {"config": {"engine": desc[engine], "dtype": "bfloat16", "nvfp4": engine == "nvfp4",
                "model": model.rstrip("/").split("/")[-1],
                "buckets": [256 << i for i in range(9)], "cuda_graphs": True,
                "token_budget": 16384, "max_wait_us": 2000, "max_prefixes": 256,
                "gdn_backend": server_args.split("--gdn-backend ")[1].split()[0],
                "server_args": server_args.replace(model, "<model>")},
     "provenance": {"tier": "dev (GB10)", "git_sha": sha, "src_dirty_files": int(dirty), "binary_sha256": bin_sha,
                    "gpu": gpu, "model_files_sha256": model_sha}}
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
