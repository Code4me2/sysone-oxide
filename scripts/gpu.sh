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
# SYSONE_GPU_LOCK is a single lock that does NOT exclude jobs holding the per-device locks: on a multi-GPU host use
# it only when every GPU job on the host uses the same SYSONE_GPU_LOCK, never mixed with per-device locking.
#
# If nvidia-smi fails, gpu.sh refuses (it cannot tell which locks protect the GPU), unless SYSONE_GPU_LOCK is set.
#
# Nesting: SYSONE_GPU_LOCKED=1 and SYSONE_GPU_HELD=<lock files> are exported for the command (benchmarks record
# whether the lock was held). A gpu.sh started inside it whose locks are all already held runs its command without
# locking again (it would otherwise wait forever on its own parent); one that needs a lock the parent does not hold
# refuses, because taking locks out of order can deadlock. The x86 runners take their locks themselves: never wrap
# them in gpu.sh.
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

outer_held="${SYSONE_GPU_HELD:-}"
export SYSONE_GPU_LOCKED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# GPUs as nvidia-smi lists them: "index, uuid" (the UUIDs only resolve UUID selections; they are never printed).
gpus=()
if out=$("$smi" --query-gpu=index,uuid --format=csv,noheader 2>/dev/null); then
    while IFS= read -r l; do [ -n "${l// /}" ] && gpus+=("$l"); done <<< "$out"
elif [ -z "${SYSONE_GPU_LOCK:-}" ]; then
    die "nvidia-smi failed: cannot tell which GPU lock(s) to take; fix the driver (or set SYSONE_GPU_LOCK, see above)"
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

# Nested inside another gpu.sh: run unlocked if every lock is already held by it, refuse otherwise.
nested=0
if [ -n "$outer_held" ]; then
    nested=1
    for l in "${locks[@]}"; do
        case " $outer_held " in *" $l "*) ;; *) nested=missing ;; esac
    done
    [ "$nested" = 1 ] || die "nested gpu.sh needs lock(s) ${locks[*]} but its parent holds only $outer_held (do not wrap GPU jobs or the x86 runners in gpu.sh)"
fi
export SYSONE_GPU_HELD="${locks[*]}"
[ "$nested" = 1 ] && SYSONE_GPU_HELD="$outer_held"

if [ "${SYSONE_GPU_DRY_RUN:-0}" = 1 ]; then
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
    if [ "$nested" = 1 ]; then echo "locks=(held by parent)"; else echo "locks=${locks[*]}"; fi
    exit 0
fi

[ "$nested" = 1 ] && exec "$@"
cmd=()
for l in "${locks[@]}"; do cmd+=(flock "$l"); done
exec "${cmd[@]}" "$@"
