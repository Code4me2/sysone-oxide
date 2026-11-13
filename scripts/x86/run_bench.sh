#!/usr/bin/env bash
# x86 benchmark runner (docs/x86/TASKS.md T1-T3). Every job runs through scripts/gpu.sh --gpu N (that GPU's lock)
# and every benchmark refuses to start while another process uses its GPU.
#
#   scripts/x86/run_bench.sh --model DIR [--gpus 0,2|all] [--mode single|concurrent|both] [--quick]
#   scripts/x86/run_bench.sh --no-forward ...            skip bench_forward (no checkpoint needed)
#
#   --mode single      (default) one GPU at a time, the full suite per GPU:
#                        hwprobe (writes the platform profile), bench_gdn --backend tile, bench_gdn --backend simt,
#                        bench_attn, bench_nvfp4, tile_gemm_probe, bench_forward
#                      -> benchmarks/results/x86/<card-slug>/<date>/gpu<N>-single/
#                      and profiles/<sm>-<card-slug>.json (from the lowest-numbered GPU of each card type)
#   --mode concurrent  all selected GPUs at once (each job under its own GPU's lock), CONCURRENT_BENCHES
#                      (default: bench_forward) -> .../gpu<N>-concurrent/; started together, finished with `wait`
#   --mode both        single, then concurrent
#   --quick            small shapes and few iterations (smoke test of the runner; numbers are not for reporting)
#   --iters N          iterations for bench_forward / bench_gdn / bench_attn / bench_nvfp4 (default: each binary's)
#   --forward-tokens L bench_forward's packed step sizes (default 1024,4096,16384; the 96 GB cards can try
#                      32768,65536 for throughput: TASKS.md T3)
#
# Results are small text files (each binary's stdout, sanitised), summary.tsv (name, exit code, seconds) and the
# sanitised metadata.json; every directory passes check_outgoing.sh --files or is moved to target/x86-quarantine/.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
cd "$X86_ROOT"

model="${SYSONE_MODEL_DIR:-}" gpus=all mode=single quick=0 forward=1 iters="" ftokens=1024,4096,16384
concurrent_benches="${CONCURRENT_BENCHES:-bench_forward}"
while [ $# -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift ;;
        --gpus) gpus="$2"; shift ;;
        --mode) mode="$2"; shift ;;
        --quick) quick=1 ;;
        --no-forward) forward=0 ;;
        --iters) iters="$2"; shift ;;
        --forward-tokens) ftokens="$2"; shift ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) x86_die "unknown argument $1 (see --help)" ;;
    esac
    shift
done
case "$mode" in single|concurrent|both) ;; *) x86_die "--mode single|concurrent|both" ;; esac
if [ "$forward" = 1 ]; then
    [ -n "$model" ] || x86_die "bench_forward needs the checkpoint: --model DIR (or SYSONE_MODEL_DIR), or --no-forward"
    [ -f "$model/config.json" ] || x86_die "--model $model is not a checkpoint directory"
    model=$(cd "$model" && pwd)
    export SYSONE_MODEL_DIR="$model"
fi
[ "$gpus" = all ] && gpus=$(x86_gpu_indices | paste -sd,)
IFS=, read -r -a gpu_list <<< "$gpus"

echo "x86: building the binaries (outside the GPU lock)" >&2
cargo build --release --bins >/dev/null 2>&1 || { cargo build --release --bins 2>&1 | tail -20; x86_die "build failed"; }
B="$X86_ROOT/target/release"

if [ "$quick" = 1 ]; then
    q_tok="1024" q_it="--iters ${iters:-2}"
    a_hwprobe=(--tokens 256,1024 --iters 3 --bw-mib 256)
    a_gdn=(--shapes 1024 $q_it)
    a_attn=(--tokens "$q_tok" $q_it)
    a_nvfp4=(--tokens "$q_tok" $q_it)
    a_tile=(--tokens "$q_tok" $q_it)
    a_forward=(--tokens "$q_tok" $q_it --eager-iters 1)
else
    it=(); [ -n "$iters" ] && it=(--iters "$iters")
    a_hwprobe=()
    a_gdn=("${it[@]}")
    a_attn=(--tokens 1024,4096,16384 "${it[@]}")
    a_nvfp4=(--tokens 1024,4096,16384 "${it[@]}")
    a_tile=()
    a_forward=(--tokens "$ftokens" "${it[@]}")
fi

# bench DIR GPU NAME: one benchmark of the suite.
bench() {
    local d="$1" g="$2" name="$3"
    case "$name" in
        hwprobe) x86_job "$d" hwprobe "$g" "$B/hwprobe" "${a_hwprobe[@]}" --profile-out "$d/profile.json" ;;
        bench_gdn_tile) x86_job "$d" bench_gdn_tile "$g" "$B/bench_gdn" --backend tile "${a_gdn[@]}" ;;
        bench_gdn_simt) x86_job "$d" bench_gdn_simt "$g" "$B/bench_gdn" --backend simt "${a_gdn[@]}" ;;
        bench_attn) x86_job "$d" bench_attn "$g" "$B/bench_attn" "${a_attn[@]}" ;;
        bench_nvfp4) x86_job "$d" bench_nvfp4 "$g" "$B/bench_nvfp4" "${a_nvfp4[@]}" ;;
        tile_gemm_probe) x86_job "$d" tile_gemm_probe "$g" "$B/tile_gemm_probe" "${a_tile[@]}" ;;
        bench_forward)
            if [ "$forward" = 1 ]; then x86_job "$d" bench_forward "$g" "$B/bench_forward" --model "$model" "${a_forward[@]}"
            else printf 'bench_forward\tNOT-RUN\t--no-forward\n' >> "$d/summary.tsv"; fi ;;
        *) x86_die "unknown benchmark $name" ;;
    esac
}

# Other GPUs busy during a single-GPU run share the host (PCIe, CPU, power): record it, do not refuse.
others_busy() {
    local g="$1" n
    n=$("$SMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    printf '%s' "$n"
}

failed=0
dirs=()
declare -A profiled=()
if [ "$mode" = single ] || [ "$mode" = both ]; then
    for g in "${gpu_list[@]}"; do
        d=$(x86_results_dir "$g" "gpu$g-single")
        x86_write_metadata "$d" "$g" bench-single \
            "run_bench.sh single$([ "$quick" = 1 ] && echo ' QUICK (not for reporting)'); compute processes on the host at start: $(others_busy "$g")"
        for b in hwprobe bench_gdn_tile bench_gdn_simt bench_attn bench_nvfp4 tile_gemm_probe bench_forward; do
            bench "$d" "$g" "$b" || failed=1
        done
        slug=$(x86_card_slug "$(x86_gpu_field "$g" name)")
        sm=$(x86_sm "$(x86_gpu_field "$g" compute_cap)")
        if [ -f "$d/profile.json" ] && [ "$quick" = 0 ] && [ -z "${profiled[$slug]:-}" ]; then
            x86_sanitize "$d/profile.json"
            cp "$d/profile.json" "profiles/$sm-$slug.json"
            profiled[$slug]=1
            echo "x86: wrote profiles/$sm-$slug.json (from GPU $g)" >&2
        fi
        x86_seal "$d" || failed=1
        dirs+=("$d")
    done
fi

if [ "$mode" = concurrent ] || [ "$mode" = both ]; then
    cdirs=()
    for g in "${gpu_list[@]}"; do
        d=$(x86_results_dir "$g" "gpu$g-concurrent")
        x86_write_metadata "$d" "$g" bench-concurrent \
            "run_bench.sh concurrent on GPUs $gpus: $concurrent_benches$([ "$quick" = 1 ] && echo ' QUICK (not for reporting)')"
        cdirs+=("$d")
    done
    pids=()
    for k in "${!gpu_list[@]}"; do
        ( for b in $concurrent_benches; do bench "${cdirs[$k]}" "${gpu_list[$k]}" "$b"; done ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || failed=1; done
    for d in "${cdirs[@]}"; do
        awk -F'\t' '$2 != "0" && $2 != "NOT-RUN" { bad = 1 } END { exit bad }' "$d/summary.tsv" || failed=1
        x86_seal "$d" || failed=1
        dirs+=("$d")
    done
fi

echo
echo "x86 benchmark summary:"
for d in "${dirs[@]}"; do
    [ -d "$d" ] || continue
    echo "  ${d#"$X86_ROOT"/}"
    awk -F'\t' '{ printf "    %-18s %s\n", $1, ($2 == "0" ? "ok (" $3 " s)" : ($2 == "NOT-RUN" ? "NOT RUN: " $3 : "FAILED (exit " $2 ")")) }' "$d/summary.tsv"
done
exit "$failed"
