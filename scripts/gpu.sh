#!/usr/bin/env bash
# Run a command that uses a GPU while holding that GPU's lock, so concurrent jobs (benchmarks, reference/golden
# runs, GPU tests) never overlap on one GPU. Every GPU job goes through this, not only timing runs: in Phase 1
# unlocked reference jobs kept the GPU busy and invalidated the benchmarks.
#
#   scripts/gpu.sh cargo test --release --test kernels_gdn
#   scripts/gpu.sh ./target/release/bench_gdn            (benchmarks also refuse to start on a busy GPU)
#   scripts/gpu.sh --gpu 1 ./target/release/hwprobe      (multi-GPU host: GPU 1 only, lock /tmp/sysone-gpu1.lock)
#   CUDA_VISIBLE_DEVICES=2 scripts/gpu.sh ...             (same as --gpu 2)
#   scripts/gpu.sh --gpu all CMD                           (every GPU visible, every GPU's lock held)
#
# GPU selection: `--gpu SEL` (or `SYSONE_GPU=SEL`), else `CUDA_VISIBLE_DEVICES`. SEL is what CUDA_VISIBLE_DEVICES
# takes: comma-separated indices or GPU UUIDs (MIG instances are not supported), or `all`. Indices are nvidia-smi's
# (PCI bus order): this script exports CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA numbers the GPUs the same way. The
# command sees only the selected GPUs (CUDA_VISIBLE_DEVICES is exported), so its CUDA device 0 is the first one.
#
# Locks: on a host with one GPU (GB10) the lock is /tmp/sysone-gpu.lock, as it always was. With several GPUs each
# selected GPU N has its own lock /tmp/sysone-gpu<N>.lock (directory: SYSONE_GPU_LOCK_DIR, default /tmp), taken in
# ascending order; when nothing is selected, GPU 0 is used (with a note on stderr). SYSONE_GPU_LOCK=<file>
# overrides all of this with one lock file.
#
# SYSONE_GPU_LOCKED=1 is exported for the command (benchmarks record whether the lock was held).
# SYSONE_GPU_DRY_RUN=1 prints the resolved selection and lock files instead of running (used by the tests).
set -euo pipefail
smi="${SYSONE_NVIDIA_SMI:-nvidia-smi}"
lockdir="${SYSONE_GPU_LOCK_DIR:-/tmp}"
die() { echo "gpu.sh: $*" >&2; exit 2; }

sel="${SYSONE_GPU:-}"
case "${1:-}" in
    --gpu) [ $# -ge 2 ] || die "--gpu needs a value"; sel="$2"; shift 2 ;;
    --gpu=*) sel="${1#--gpu=}"; shift ;;
esac
[ $# -ge 1 ] || die "usage: scripts/gpu.sh [--gpu N[,M...]|all] COMMAND [ARGS...]"

export SYSONE_GPU_LOCKED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPUs as nvidia-smi lists them: "index, uuid" (the UUIDs only resolve UUID selections; they are never printed).
gpus=()
if out=$("$smi" --query-gpu=index,uuid --format=csv,noheader 2>/dev/null); then
    while IFS= read -r l; do [ -n "${l// /}" ] && gpus+=("$l"); done <<< "$out"
fi
n=${#gpus[@]}

locks=()
if [ -n "${SYSONE_GPU_LOCK:-}" ]; then
    locks=("$SYSONE_GPU_LOCK")
    if [ "$sel" = all ]; then unset CUDA_VISIBLE_DEVICES; elif [ -n "$sel" ]; then export CUDA_VISIBLE_DEVICES="$sel"; fi
elif [ "$n" -le 1 ]; then
    # Single-GPU host (GB10) or no nvidia-smi: the one shared lock, unchanged.
    locks=("$lockdir/sysone-gpu.lock")
    if [ "$sel" = all ]; then unset CUDA_VISIBLE_DEVICES; elif [ -n "$sel" ]; then export CUDA_VISIBLE_DEVICES="$sel"; fi
else
    if [ "$sel" = all ]; then
        unset CUDA_VISIBLE_DEVICES
        idx=()
        for g in "${gpus[@]}"; do idx+=("${g%%,*}"); done
    else
        if [ -n "$sel" ]; then
            export CUDA_VISIBLE_DEVICES="$sel"
        elif [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
            echo "gpu.sh: $n GPUs and none selected; using GPU 0 (pass --gpu N or set CUDA_VISIBLE_DEVICES)" >&2
            export CUDA_VISIBLE_DEVICES=0
        fi
        [ -n "$CUDA_VISIBLE_DEVICES" ] || die "CUDA_VISIBLE_DEVICES is empty: no GPU selected"
        idx=()
        IFS=, read -r -a entries <<< "$CUDA_VISIBLE_DEVICES"
        for e in "${entries[@]}"; do
            e="${e// /}"
            if [[ "$e" =~ ^[0-9]+$ ]]; then
                [ "$e" -lt "$n" ] || die "GPU $e does not exist (this host has $n: 0..$((n - 1)))"
                idx+=("$e")
            elif [[ "$e" == GPU-* ]]; then
                hit=()
                for g in "${gpus[@]}"; do
                    u="${g#*,}"; u="${u// /}"
                    [[ "$u" == "$e"* ]] && hit+=("${g%%,*}")
                done
                [ "${#hit[@]}" -eq 1 ] || die "CUDA_VISIBLE_DEVICES entry ${e:0:8}... matches ${#hit[@]} GPUs (need exactly 1)"
                idx+=("${hit[0]}")
            else
                die "unsupported CUDA_VISIBLE_DEVICES entry '${e:0:12}' (use GPU indices or GPU UUIDs; MIG is not supported)"
            fi
        done
    fi
    while IFS= read -r i; do locks+=("$lockdir/sysone-gpu$i.lock"); done < <(printf '%s\n' "${idx[@]}" | sort -n -u)
fi

if [ "${SYSONE_GPU_DRY_RUN:-0}" = 1 ]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
    echo "locks=${locks[*]}"
    exit 0
fi

cmd=()
for l in "${locks[@]}"; do cmd+=(flock "$l"); done
exec "${cmd[@]}" "$@"
