#!/usr/bin/env bash
# Tests of scripts/x86/check_outgoing.sh on throw-away git repositories (no GPU, no network).
#   scripts/tests/test_check_outgoing.sh
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
check="$here/../x86/check_outgoing.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
export SYSONE_OUTGOING_DENY="acme-secret-project"
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
fails=0
n=0

g() { git -c user.name=Tester -c user.email=tester@users.noreply.github.com -c commit.gpgsign=false "$@"; }

# new_repo: a repo with a main commit and a work branch checked out.
new_repo() {
    n=$((n + 1))
    repo="$tmp/r$n"
    mkdir -p "$repo" && cd "$repo" || exit 1
    g init -q -b main
    mkdir -p benchmarks/results/x86 profiles src tests docs scripts/x86
    echo "base" > README.md
    g add -A && g commit -qm base
    g checkout -qb x86/1-test
}
# add PATH CONTENT: write and commit.
add() { mkdir -p "$(dirname "$1")"; printf '%s\n' "$2" > "$1"; g add -A; g commit -qm "add $1"; }

# expect pass|fail "description" [check args...]: run the check in the current repo.
expect() {
    local want="$1" what="$2"; shift 2
    local out rc
    out=$(bash "$check" --base main "$@" 2>&1); rc=$?
    if { [ "$want" = pass ] && [ "$rc" = 0 ]; } || { [ "$want" = fail ] && [ "$rc" = 1 ]; }; then echo "ok    $what"
    else echo "FAIL  $what (want $want, exit $rc):"; printf '%s\n' "$out" | sed 's/^/        /'; fails=1; fi
}

ok_json='{"gpu_name": "NVIDIA RTX PRO 6000 Blackwell", "compute_capability": "12.0", "memory_total_mib": 97887, "driver": "580.95.05", "tok_s": 12345.678, "mem_bytes": 102641958912}'

new_repo; add benchmarks/results/x86/rtx-pro-6000/2026-11-14/gpu0-single/metadata.json "$ok_json"
add profiles/sm_120-rtx-pro-6000.json '{"mem_bw_gbs": 1612.5, "note": "listen 0.0.0.0:8000, 127.0.0.1"}'
expect pass "clean results PR"

new_repo; add src/engine/x.rs 'fn f() {}'
expect fail "code path in a results PR"
expect pass "code path with --code" --code

new_repo; add tests/tolerances.toml 'x = 1'
expect fail "tolerances.toml never allowed" --safety
new_repo; add src/tolerances.rs '// x'
expect fail "src/tolerances.rs never allowed" --code
new_repo; add scripts/gpu.sh 'echo'
expect fail "safety script with --code" --code
expect pass "safety script with --safety" --safety
new_repo; add benchmarks/results/x86/a/model.safetensors 'x'
expect fail "model file extension"

new_repo; git checkout -q main; add benchmarks/results/x86/a.txt 'x'
expect fail "on main"

bad() {   # bad "description" content
    new_repo; add "benchmarks/results/x86/c/d/out.txt" "$2"
    expect fail "$1"
}
bad "home path" 'model at /home/alice/models'
bad "macOS home path" 'see /Users/bob/x'
bad "IPv4 address" 'server 10.1.2.3:8000'
bad "e-mail address" 'contact alice@example.org'
bad "GPU UUID" 'GPU-1a2b3c4d-1111-2222-3333-444455556666'
bad "MIG UUID" 'MIG-1a2b3c4d-1111'
bad "PCI bus id" 'bus 00000000:41:00.0'
bad "MAC address" 'eth 00:1b:21:aa:bb:cc'
bad "serial field" '"serial": "1320123456789"'
bad "serial in csv" 'name, Serial Number: 132'
bad "bare 13-digit integer" 'board 1320123456789 x'
bad "HF token" 'hf_abcdefghijklmnopqrstuvwxyzABCD'
bad "GitHub token" 'ghp_abcdefghijklmnopqrstuvwxyz0123456789'
bad "fine-grained GitHub token" 'github_pat_11ABCDEFG0123456789_abcdefghij'
bad "private key" '-----BEGIN OPENSSH PRIVATE KEY-----'
bad "denied word" 'tagging run for ACME-SECRET-PROJECT docs'
bad "host name" "built on $(hostname -s)"

new_repo; add benchmarks/results/x86/c/ok.txt 'version 1.2.3, driver 580.95.05, 13.4, 3.1415926535897932, ratio 1.25e-3, sm_120'
expect pass "versions and floats are not IPs or serials"
new_repo; add benchmarks/results/x86/c/ok.txt 'Co-Authored-By: Claude <noreply@anthropic.com>, 123+x@users.noreply.github.com'
expect pass "noreply e-mails allowed"

new_repo; head -c 2000 /dev/urandom > benchmarks/results/x86/blob.dat; g add -A; g commit -qm blob
expect fail "binary file"
new_repo; mkdir -p benchmarks/results/x86; yes 'a line of text' | head -c 1100000 > benchmarks/results/x86/big.txt; g add -A; g commit -qm big
expect fail "file over 1 MB"

new_repo; mkdir -p benchmarks/results/x86; echo 'see /home/carol' > benchmarks/results/x86/staged.txt; g add -A
expect fail "staged, uncommitted change is checked"

new_repo; add benchmarks/results/x86/a.txt 'x'
git -c user.name=T -c user.email="t@box.local" -c commit.gpgsign=false commit -q --allow-empty -m "empty"
expect fail "machine identity in commit e-mail"
new_repo; g commit -q --allow-empty -m "results from 192.168.1.20"
expect fail "IP in commit message"

# --files mode (used by the runners on result directories).
mkdir -p "$tmp/files" "$tmp/Users/x"
printf '%s\n' "$ok_json" > "$tmp/files/good.json"
# The file's own location is not scanned, only its contents.
printf '%s\n' "$ok_json" > "$tmp/Users/x/good.json"
if bash "$check" --files "$tmp/Users/x/good.json" >/dev/null; then echo "ok    --files ignores the file's path"; else echo "FAIL  --files scanned the path"; fails=1; fi
printf 'path /home/dave/x\n' > "$tmp/files/bad.txt"
if bash "$check" --files "$tmp/files/good.json" >/dev/null; then echo "ok    --files clean"; else echo "FAIL  --files clean"; fails=1; fi
if bash "$check" --files "$tmp/files/good.json" "$tmp/files/bad.txt" >/dev/null; then echo "FAIL  --files dirty accepted"; fails=1; else echo "ok    --files dirty"; fi

[ "$fails" = 0 ] && echo "check_outgoing.sh: all tests passed" || { echo "check_outgoing.sh: FAILED"; exit 1; }
