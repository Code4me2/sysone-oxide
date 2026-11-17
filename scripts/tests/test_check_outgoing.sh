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
unset SYSONE_OUTGOING_ALLOW_EMAIL
fails=0
n=0
reason=""

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
    if [ "$want" = fail ] && [ -n "${reason:-}" ] && ! printf '%s\n' "$out" | grep -qE -- "$reason"; then
        echo "FAIL  $what (failed, but not with /$reason/):"; printf '%s\n' "$out" | sed 's/^/        /'; fails=1; reason=""; return
    fi
    reason=""
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

bad() {   # bad "description" content [reason regex]
    new_repo; add "benchmarks/results/x86/c/d/out.txt" "$2"
    reason="${3:-}"; expect fail "$1"
}
good() {  # good "description" content
    new_repo; add "benchmarks/results/x86/c/d/out.txt" "$2"
    expect pass "$1"
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

bad "IPv4 address at the end of a sentence" 'connected to 192.168.1.5.' 'IPv4 address: 192.168.1.5'
bad "IPv4 address in a URL" 'http://10.0.0.12:8000/v1' 'IPv4'
bad "bare UUID" 'job 3f2a9c1e-1b2c-4d5e-8f90-123456789abc' 'UUID'
bad "lowercase gpu- UUID" 'gpu-3f2a9c1e-1b2c-4d5e-8f90-123456789abc' 'UUID'
bad "IPv6 link-local" 'peer fe80::1c2:3ff:fe4d:5a6b' 'IPv6'
bad "IPv6 documentation prefix" 'peer 2001:db8::1' 'IPv6'
bad "IPv6 full form" '2001:0db8:85a3:0000:0000:8a2e:0370:7334' 'IPv6'
bad "absolute path /data" 'read /data/customers/acme/doc1.pdf' 'absolute path'
bad "absolute path /mnt" 'mounted at /mnt/secure' 'absolute path'
bad "absolute path /srv in quotes" '"input": "/srv/x"' 'absolute path'
bad "~ path" 'see ~/docs/x' 'home path variable'
bad "\$HOME path" 'see $HOME/docs/x' 'home path variable'
good "allowed system paths, URLs, rates, Rust paths, times" \
    'lock /tmp/sysone-gpu0.lock, ptxas /usr/local/cuda-13/bin/ptxas, >/dev/null, https://github.com/NVIDIA/cuda-rust, POST /v1/completions, 812.5 tok/s, 1.6 TB/s, u8::MAX, std::fs::read, ::1, at 12:34:56, ~18 GB, version 1.2.3.4.5., </think>, <models>/elephant-1, /path/to/model'
new_repo; add benchmarks/results/x86/c/ok.txt 'version 1.2.3, driver 580.95.05, 13.4, 3.1415926535897932, ratio 1.25e-3, sm_120'
expect pass "versions and floats are not IPs or serials"
new_repo; add benchmarks/results/x86/c/ok.txt 'Co-Authored-By: Claude <noreply@anthropic.com>, 123+x@users.noreply.github.com'
expect pass "noreply e-mails allowed"

# Exactly 1 MiB, well above both the old 8 KiB sample and typical pipe capacity.
# Two-byte glyphs exercise ASCII and valid UTF-8 with identical byte counts.
large_text() {
    awk -v glyph="$2" -v lines="${3:-8192}" 'BEGIN {
        line = ""; for (j = 0; j < 63; j++) line = line glyph
        for (i = 0; i < lines; i++) print line "a"
    }' > "$1"
}
for glyph in aa "$(printf '\303\251')"; do
    new_repo
    large_text benchmarks/results/x86/text.txt "$glyph"
    g add -A
    expect pass "staged 1 MiB text ($glyph)"
    g commit -qm text
    expect pass "committed 1 MiB text ($glyph)"
done

new_repo; : > benchmarks/results/x86/empty.txt; g add -A
expect pass "staged empty blob"
g commit -qm empty
expect pass "committed empty blob"

new_repo; printf '\0plain text\n' > benchmarks/results/x86/nul.dat; g add -A
reason='binary file'; expect fail "staged NUL binary"
g commit -qm binary
reason='binary file'; expect fail "committed NUL binary"

# Earlier matching lines must not hide a NUL beyond both 8 KiB and grep's input buffer.
new_repo; large_text benchmarks/results/x86/late.dat aa 2048
printf '\0tail\n' >> benchmarks/results/x86/late.dat; g add -A
reason='binary file'; expect fail "staged late NUL binary"
g commit -qm binary
reason='binary file'; expect fail "committed late NUL binary"
g rm -q benchmarks/results/x86/late.dat; g commit -qm remove
reason='binary file'; expect fail "late NUL binary added, then deleted"

new_repo; large_text benchmarks/results/x86/oversize.txt aa
printf 'a' >> benchmarks/results/x86/oversize.txt; g add -A
reason='over 1 MB'; expect fail "staged text one byte over 1 MiB"
g commit -qm oversize
reason='over 1 MB'; expect fail "committed text one byte over 1 MiB"

# Inject only a blob-content read failure; metadata and all other git operations still work.
new_repo; printf 'text\n' > benchmarks/results/x86/read.txt; g add -A
# Exported shell functions are inherited by the checker, confined to this throw-away test process.
git() {
    if [ "${1:-}" = cat-file ] && [ "${2:-}" = blob ]; then return 1; fi
    command git "$@"
}
export -f git
reason='could not read blob'; expect fail "staged blob read failure fails closed"
g commit -qm text
reason='could not read blob'; expect fail "committed blob read failure fails closed"
unset -f git

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

# History, not just the net diff: what is added in one commit and removed in a later one is still pushed.
new_repo; add benchmarks/results/x86/c/d/out.txt 'model at /home/alice/secret'; add benchmarks/results/x86/c/d/out.txt 'clean'
reason='home path'; expect fail "leak added, then removed in a later commit"
new_repo; mkdir -p benchmarks/results/x86; yes 'a line of text' | head -c 2100000 > benchmarks/results/x86/big.txt
g add -A; g commit -qm big; add benchmarks/results/x86/big.txt 'small now'
reason='over 1 MB'; expect fail "2 MB file added, then shrunk"
new_repo; add src/engine/x.rs 'fn f() {}'; g rm -q src/engine/x.rs; g commit -qm rm
reason='src/engine/x.rs: outside the allow-list'; expect fail "file outside the allow-list added, then deleted"
new_repo; head -c 3000 /dev/urandom > benchmarks/results/x86/blob.dat; g add -A; g commit -qm blob; g rm -q benchmarks/results/x86/blob.dat; g commit -qm rm
reason='binary'; expect fail "binary added, then deleted"
new_repo; add benchmarks/results/x86/a.txt 'x'; g checkout -q main; add README.md 'base 2'; g checkout -q x86/1-test
g merge -q --no-edit main
reason='merge commit'; expect fail "merge commit in the branch"
# Local diff drivers / textconv must not hide content.
new_repo; git config diff.hide.textconv 'sed d'; echo '*.txt diff=hide' > .git/info/attributes
add benchmarks/results/x86/c/t.txt 'model at /home/alice/x'
reason='home path'; expect fail "textconv driver cannot hide a leak"

# Commit messages and identities.
new_repo; add benchmarks/results/x86/a.txt 'x'
g commit -q --allow-empty -m "$(printf 'results\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com> /home/alice/x 10.1.2.3')"
reason='home path'; expect fail "text after Co-Authored-By on the same line is scanned"
new_repo; add benchmarks/results/x86/a.txt 'x'
g commit -q --allow-empty -m "$(printf 'results\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_014sEtuVEV2A7Tveo3WRGFv6')"
expect pass "exact attribution lines pass"
new_repo; add benchmarks/results/x86/a.txt 'x'
git -c user.name=Alice -c user.email=alice@bigcorp.com -c commit.gpgsign=false commit -q --allow-empty -m "results"
reason='e-mail is not a noreply'; expect fail "real author e-mail not on the allow-list"
export SYSONE_OUTGOING_ALLOW_EMAIL="other@x.org, alice@bigcorp.com"
expect pass "author e-mail on SYSONE_OUTGOING_ALLOW_EMAIL"
unset SYSONE_OUTGOING_ALLOW_EMAIL
new_repo; add benchmarks/results/x86/a.txt 'x'
git -c user.name="$(id -un)" -c user.email=x@users.noreply.github.com -c commit.gpgsign=false commit -q --allow-empty -m "results"
reason='name contains'; expect fail "author name is the machine user name"

# --files mode (used by the runners on result directories).
mkdir -p "$tmp/files" "$tmp/Users/x"
printf '%s\n' "$ok_json" > "$tmp/files/good.json"
# The file's own location is not scanned, only its contents.
printf '%s\n' "$ok_json" > "$tmp/Users/x/good.json"
if bash "$check" --files "$tmp/Users/x/good.json" >/dev/null; then echo "ok    --files ignores the file's path"; else echo "FAIL  --files scanned the path"; fails=1; fi
printf 'path /home/dave/x\n' > "$tmp/files/bad.txt"
if bash "$check" --files "$tmp/files/good.json" >/dev/null; then echo "ok    --files clean"; else echo "FAIL  --files clean"; fails=1; fi
if bash "$check" --files "$tmp/files/good.json" "$tmp/files/bad.txt" >/dev/null; then echo "FAIL  --files dirty accepted"; fails=1; else echo "ok    --files dirty"; fi

# x86_seal (lib.sh) scans every file of a results directory, dotfiles included, and quarantines it on failure.
mkdir -p "$tmp/seal/r1" "$tmp/seal/r2"
printf '%s\n' "$ok_json" > "$tmp/seal/r1/metadata.json"
printf '%s\n' "$ok_json" > "$tmp/seal/r2/metadata.json"; printf 'path /home/alice\n' > "$tmp/seal/r2/.hidden"
( source "$here/../x86/lib.sh"; X86_QUARANTINE="$tmp/quarantine"; x86_seal "$tmp/seal/r1" ) >/dev/null 2>&1 &&
    [ -d "$tmp/seal/r1" ] && echo "ok    x86_seal keeps a clean directory" || { echo "FAIL  x86_seal clean"; fails=1; }
if ( source "$here/../x86/lib.sh"; X86_QUARANTINE="$tmp/quarantine"; x86_seal "$tmp/seal/r2" ) >/dev/null 2>&1 || [ -d "$tmp/seal/r2" ]; then
    echo "FAIL  x86_seal missed a dotfile leak"; fails=1
else echo "ok    x86_seal quarantines a dotfile leak"; fi

[ "$fails" = 0 ] && echo "check_outgoing.sh: all tests passed" || { echo "check_outgoing.sh: FAILED"; exit 1; }
