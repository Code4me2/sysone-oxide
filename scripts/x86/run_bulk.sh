#!/usr/bin/env bash
# x86 bulk-throughput runner (docs/x86/TASKS.md T7, docs/BULK.md). Every GPU job runs under scripts/gpu.sh (the GPU
# locks of the GPUs it uses) and every GPU worker refuses to start while another process uses its GPU.
#
#   scripts/x86/run_bulk.sh --model DIR [--gpus 0,1,2|all] [--mode single|all|both] [--server] [--docs N]
#                           [--concurrency C] [--quick]
#
#   corpus        `sysone bulk gen --docs N` (default 2000: states of 300..8,000 tokens, 4..16 questions, seed 1),
#                 written to target/x86-bulk/ (local, never committed), and `sysone bulk simulate` on it (CPU only:
#                 padding of the server's batcher vs bulk windows) -> benchmarks/results/x86/host/<date>/bulk-corpus/
#   --mode single (default) `sysone bulk` on one GPU at a time -> .../<card-slug>/<date>/gpu<N>-bulk/
#   --mode all    `sysone bulk` on all selected GPUs at once (one worker per GPU, one shared queue)
#                 -> benchmarks/results/x86/host/<date>/bulk-all/ (metadata of the first GPU; the single runs have
#                 each GPU's own)
#   --mode both   single, then all
#   --server      also, per GPU: the HTTP server on that GPU, driven by `loadgen --corpus` (each document once,
#                 --concurrency clients, default 256) -> loadgen.json in that GPU's gpu<N>-bulk/ directory
#   --quick       200 documents (a smoke test of the runner; numbers are not for reporting)
#
# Each results directory holds the bulk summary (bulk.txt: one JSON line, no paths), `sysone bulk compare` reports
# between runs (compare-*.txt: ids, statuses and probability differences), summary.tsv and metadata.json. Result
# files (one line per document) stay in target/x86-bulk/.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$X86_ROOT"

model="${SYSONE_MODEL_DIR:-}" gpus=all mode=single server=0 docs=2000 conc=256 quick=0
while [ $# -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift ;;
        --gpus) gpus="$2"; shift ;;
        --mode) mode="$2"; shift ;;
        --server) server=1 ;;
        --docs) docs="$2"; shift ;;
        --concurrency) conc="$2"; shift ;;
        --quick) quick=1; docs=200 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) x86_die "unknown argument $1 (see --help)" ;;
    esac
    shift
done
case "$mode" in single|all|both) ;; *) x86_die "--mode single|all|both" ;; esac
[ -n "$model" ] || x86_die "needs the checkpoint: --model DIR (or SYSONE_MODEL_DIR)"
[ -f "$model/config.json" ] || x86_die "--model $model is not a checkpoint directory"
model=$(cd "$model" && pwd)
export SYSONE_MODEL_DIR="$model"
[ "$gpus" = all ] && gpus=$(x86_gpu_indices | paste -sd,)
IFS=, read -r -a gpu_list <<< "$gpus"
tag=$([ "$quick" = 1 ] && echo ' QUICK (not for reporting)' || true)

echo "x86: building the binaries (outside the GPU lock)" >&2
cargo build --release --bins >/dev/null 2>&1 || { cargo build --release --bins 2>&1 | tail -20; x86_die "build failed"; }
B="$X86_ROOT/target/release"
W="$X86_ROOT/target/x86-bulk"
mkdir -p "$W" "$X86_LOGS"

# host CMD...: a CPU job of the host directory $hd (stdout sanitised into $hd/NAME.txt).
host_job() {
    local name="$1"; shift
    local t0 rc
    t0=$(date +%s)
    echo "x86: [cpu] $name" >&2
    "$@" > "$hd/$name.txt" 2> "$X86_LOGS/host_bulk_$name.log"; rc=$?
    x86_sanitize "$hd/$name.txt"
    printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$hd/summary.tsv"
    [ "$rc" = 0 ] || { echo "x86: [cpu] $name FAILED (exit $rc)" >&2; failed=1; }
}

failed=0
dirs=()
corpus="$W/corpus-$docs.jsonl"
hd=$(x86_results_dir host bulk-corpus)
x86_write_metadata "$hd" "${gpu_list[0]}" bulk-corpus "run_bulk.sh: corpus of $docs documents and the packing simulation (CPU only)$tag"
host_job gen "$B/sysone" bulk gen --docs "$docs" --output "$corpus"
host_job simulate "$B/sysone" bulk simulate --model "$model" --input "$corpus" --max-step 16384,32768,65536 --window 64,256,1024
x86_seal "$hd" || failed=1
dirs+=("$hd")

# bulk DIR GPUSEL NAME: one bulk run on the GPUs GPUSEL (all their locks held by x86_job's gpu.sh; the workers'
# nested gpu.sh reuse them), results to target/x86-bulk/NAME.jsonl (fresh), summary to DIR/bulk.txt.
bulk() {
    local d="$1" sel="$2" name="$3"
    rm -f "$W/$name.jsonl"
    x86_job "$d" bulk "$sel" "$B/sysone" bulk --model "$model" --input "$corpus" --output "$W/$name.jsonl" \
        --gpus "$sel" --require-idle --progress-s 60 || failed=1
}

if [ "$mode" = single ] || [ "$mode" = both ]; then
    for g in "${gpu_list[@]}"; do
        d=$(x86_results_dir "$g" "gpu$g-bulk")
        x86_write_metadata "$d" "$g" bulk-single "run_bulk.sh single: sysone bulk on GPU $g, $docs documents$tag"
        bulk "$d" "$g" "bulk-gpu$g"
        if [ "$server" = 1 ]; then
            port=$((18000 + g))
            echo "x86: [gpu $g] server + loadgen --corpus (concurrency $conc)" >&2
            # The server holds GPU g's lock while it runs; setsid gives it its own process group to stop it with.
            setsid "$X86_ROOT/scripts/gpu.sh" --gpu "$g" "$B/sysone" --model "$model" --engine bf16 \
                --listen "127.0.0.1:$port" > "$X86_LOGS/bulk_server_gpu$g.log" 2>&1 &
            spid=$!
            up=0
            for _ in $(seq 1 600); do
                if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then up=1; break; fi
                kill -0 "$spid" 2>/dev/null || break
                sleep 1
            done
            if [ "$up" = 1 ]; then
                t0=$(date +%s)
                "$B/loadgen" --addr "127.0.0.1:$port" --concurrency "$conc" --corpus "$corpus" \
                    --responses "$W/server-gpu$g.jsonl" > "$d/loadgen.json" 2> "$X86_LOGS/bulk_loadgen_gpu$g.log"
                rc=$?
                x86_sanitize "$d/loadgen.json"
                printf 'loadgen\t%s\t%s\n' "$rc" "$(( $(date +%s) - t0 ))" >> "$d/summary.tsv"
                [ "$rc" = 0 ] || failed=1
            else
                printf 'loadgen\tNOT-RUN\tserver did not come up (local log target/x86-logs/bulk_server_gpu%s.log)\n' "$g" >> "$d/summary.tsv"
                failed=1
            fi
            kill -INT -- "-$spid" 2>/dev/null; wait "$spid" 2>/dev/null
            if [ -f "$W/server-gpu$g.jsonl" ] && [ -f "$W/bulk-gpu$g.jsonl" ]; then
                "$B/sysone" bulk compare "$W/server-gpu$g.jsonl" "$W/bulk-gpu$g.jsonl" > "$d/compare-server-vs-bulk.txt" 2>/dev/null
                x86_sanitize "$d/compare-server-vs-bulk.txt"
            fi
        fi
        x86_seal "$d" || failed=1
        dirs+=("$d")
    done
fi

if [ "$mode" = all ] || [ "$mode" = both ]; then
    d=$(x86_results_dir host bulk-all)
    x86_write_metadata "$d" "${gpu_list[0]}" bulk-all "run_bulk.sh all: sysone bulk on GPUs $gpus at once (gpu fields: GPU ${gpu_list[0]} only), $docs documents$tag"
    bulk "$d" "$gpus" bulk-all
    for g in "${gpu_list[@]}"; do
        if [ -f "$W/bulk-gpu$g.jsonl" ] && [ -f "$W/bulk-all.jsonl" ]; then
            "$B/sysone" bulk compare "$W/bulk-gpu$g.jsonl" "$W/bulk-all.jsonl" > "$d/compare-gpu$g-vs-all.txt" 2>/dev/null
            x86_sanitize "$d/compare-gpu$g-vs-all.txt"
        fi
    done
    x86_seal "$d" || failed=1
    dirs+=("$d")
fi

echo
echo "x86 bulk summary:"
for d in "${dirs[@]}"; do
    [ -d "$d" ] || continue
    echo "  ${d#"$X86_ROOT"/}"
    awk -F'\t' '{ printf "    %-12s %s\n", $1, ($2 == "0" ? "ok (" $3 " s)" : ($2 == "NOT-RUN" ? "NOT RUN: " $3 : "FAILED (exit " $2 ")")) }' "$d/summary.tsv"
    [ -f "$d/bulk.txt" ] && sed -n 's/.*"docs_per_s":\([0-9.]*\).*"packed_tok_per_s":\([0-9.]*\).*/    bulk: \1 docs\/s, \2 packed tok\/s/p' "$d/bulk.txt"
done
exit "$failed"
