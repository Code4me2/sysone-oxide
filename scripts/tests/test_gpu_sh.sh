#!/usr/bin/env bash
# Tests of scripts/gpu.sh's GPU selection and per-device locks, against a fake nvidia-smi (no GPU needed).
#   scripts/tests/test_gpu_sh.sh
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gpu="$here/../gpu.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export SYSONE_NVIDIA_SMI="$here/fake-nvidia-smi" SYSONE_GPU_LOCK_DIR="$tmp"
unset SYSONE_GPU SYSONE_GPU_LOCK SYSONE_GPU_HELD CUDA_VISIBLE_DEVICES
fails=0

# expect "<description>" "<expected output>" env... -- args...
expect() {
    local what="$1" want="$2"; shift 2
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    local got
    got=$(env "${envs[@]}" SYSONE_GPU_DRY_RUN=1 bash "$gpu" "$@" true 2>/dev/null | tr '\n' ' ')
    if [ "$got" = "$want" ]; then echo "ok    $what"; else echo "FAIL  $what: got '$got', want '$want'"; fails=1; fi
}
expect_error() {
    local what="$1"; shift
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    if env "${envs[@]}" SYSONE_GPU_DRY_RUN=1 bash "$gpu" "$@" true >/dev/null 2>&1; then
        echo "FAIL  $what: accepted"; fails=1
    else echo "ok    $what"; fi
}

L="$tmp"
# Single GPU (GB10): the legacy lock, whatever is selected.
expect "1 GPU, nothing selected" "CUDA_VISIBLE_DEVICES=<unset> locks=$L/sysone-gpu.lock " FAKE_GPUS=1 --
expect "1 GPU, --gpu 0" "CUDA_VISIBLE_DEVICES=0 locks=$L/sysone-gpu.lock " FAKE_GPUS=1 -- --gpu 0
expect "no nvidia-smi output" "CUDA_VISIBLE_DEVICES=<unset> locks=$L/sysone-gpu.lock " FAKE_GPUS=0 --
# Three GPUs: per-device locks.
expect "3 GPUs, nothing selected -> GPU 0" "CUDA_VISIBLE_DEVICES=0 locks=$L/sysone-gpu0.lock " --
expect "3 GPUs, --gpu 2" "CUDA_VISIBLE_DEVICES=2 locks=$L/sysone-gpu2.lock " -- --gpu 2
expect "3 GPUs, --gpu=1" "CUDA_VISIBLE_DEVICES=1 locks=$L/sysone-gpu1.lock " -- --gpu=1
expect "3 GPUs, SYSONE_GPU=1" "CUDA_VISIBLE_DEVICES=1 locks=$L/sysone-gpu1.lock " SYSONE_GPU=1 --
expect "3 GPUs, CUDA_VISIBLE_DEVICES=1" "CUDA_VISIBLE_DEVICES=1 locks=$L/sysone-gpu1.lock " CUDA_VISIBLE_DEVICES=1 --
expect "--gpu overrides CUDA_VISIBLE_DEVICES" "CUDA_VISIBLE_DEVICES=0 locks=$L/sysone-gpu0.lock " CUDA_VISIBLE_DEVICES=2 -- --gpu 0
expect "two GPUs, locks sorted" "CUDA_VISIBLE_DEVICES=2,0 locks=$L/sysone-gpu0.lock $L/sysone-gpu2.lock " -- --gpu 2,0
expect "UUID selection" "CUDA_VISIBLE_DEVICES=GPU-22222222 locks=$L/sysone-gpu2.lock " -- --gpu GPU-22222222
expect "--gpu all" "CUDA_VISIBLE_DEVICES=<unset> locks=$L/sysone-gpu0.lock $L/sysone-gpu1.lock $L/sysone-gpu2.lock " \
    CUDA_VISIBLE_DEVICES=1 -- --gpu all
expect "SYSONE_GPU_LOCK overrides" "CUDA_VISIBLE_DEVICES=1 locks=/x/y.lock " SYSONE_GPU_LOCK=/x/y.lock -- --gpu 1
expect_error "index out of range" -- --gpu 3
expect_error "ambiguous UUID prefix" -- --gpu GPU-
expect_error "MIG entry" -- --gpu MIG-1234
expect_error "empty CUDA_VISIBLE_DEVICES" CUDA_VISIBLE_DEVICES= --
expect_error "nvidia-smi fails, --gpu 1" SYSONE_NVIDIA_SMI=/nonexistent -- --gpu 1
expect_error "nvidia-smi fails, nothing selected" SYSONE_NVIDIA_SMI=/nonexistent --
expect "nvidia-smi fails, SYSONE_GPU_LOCK set" "CUDA_VISIBLE_DEVICES=1 locks=/x/y.lock " \
    SYSONE_NVIDIA_SMI=/nonexistent SYSONE_GPU_LOCK=/x/y.lock -- --gpu 1
# Nesting: locks already held by the parent are not taken again; others are refused.
expect "nested, same GPU" "CUDA_VISIBLE_DEVICES=1 locks=(held by parent) " SYSONE_GPU_HELD="$L/sysone-gpu1.lock" -- --gpu 1
expect "nested, subset of --gpu all" "CUDA_VISIBLE_DEVICES=2 locks=(held by parent) " \
    SYSONE_GPU_HELD="$L/sysone-gpu0.lock $L/sysone-gpu1.lock $L/sysone-gpu2.lock" -- --gpu 2
expect_error "nested, other GPU" SYSONE_GPU_HELD="$L/sysone-gpu1.lock" -- --gpu 0
out=$(timeout 10 bash "$gpu" --gpu 1 bash "$gpu" --gpu 1 bash -c 'echo inner "$SYSONE_GPU_HELD"' 2>&1)
if [ "$out" = "inner $L/sysone-gpu1.lock" ]; then echo "ok    nested gpu.sh on the same GPU does not deadlock"
else echo "FAIL  nested gpu.sh: got '$out'"; fails=1; fi
if timeout 10 bash "$gpu" --gpu 1 bash "$gpu" --gpu 0 true >/dev/null 2>&1; then echo "FAIL  nested on another GPU: accepted"; fails=1
else echo "ok    nested gpu.sh on another GPU refused"; fi
if SYSONE_GPU_DRY_RUN=1 bash "$gpu" --gpu 1 >/dev/null 2>&1; then echo "FAIL  no command: accepted"; fails=1; else echo "ok    no command"; fi

# The command runs with the lock held, and other GPUs' locks stay free.
out=$(SYSONE_GPU=1 bash "$gpu" bash -c 'echo "$SYSONE_GPU_LOCKED $CUDA_DEVICE_ORDER $CUDA_VISIBLE_DEVICES"; flock -n '"$L"'/sysone-gpu0.lock true && echo free0; flock -n '"$L"'/sysone-gpu1.lock true || echo held1' 2>&1 | tr '\n' ' ')
if [ "$out" = "1 PCI_BUS_ID 1 free0 held1 " ]; then echo "ok    lock held for the command only on its GPU"; else echo "FAIL  lock: got '$out'"; fails=1; fi

[ "$fails" = 0 ] && echo "gpu.sh: all tests passed" || { echo "gpu.sh: FAILED"; exit 1; }
