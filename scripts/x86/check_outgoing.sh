#!/usr/bin/env bash
# Outgoing-data check for the x86 host (docs/x86/AGENT.md, "Security and data hygiene"). Run it before EVERY push;
# a push is allowed only when it prints "check_outgoing: OK".
#
#   scripts/x86/check_outgoing.sh                 results PR: branch diff vs origin/main (+ staged changes)
#   scripts/x86/check_outgoing.sh --code          code PR: also allows source/doc paths (see CODE_PATHS)
#   scripts/x86/check_outgoing.sh --safety        additionally allows the safety scripts (only after the change was
#                                                 agreed in an issue: AGENT.md, "What not to do")
#   scripts/x86/check_outgoing.sh --base REF      compare against REF instead of origin/main
#   scripts/x86/check_outgoing.sh --files F...    scan files (contents and size) without git; used by the runners
#
# Fails when the diff
#   - is on main/master, or touches a path outside the allow-list (results PR: benchmarks/results/x86/**,
#     profiles/**, kernels-simt/artifacts/**), or a gate/tolerance file (never allowed), or a model/trace file;
#   - adds a line with: a home path (/home/, /Users/, /root/), an IPv4 address (except 0.0.0.0, 127.0.0.1), an
#     e-mail address (except noreply ones), a GPU/MIG UUID, a PCI bus id, a MAC address, a serial number (a
#     "serial" field or a bare integer of 13+ digits), an access token (hf_, ghp_/gho_/ghs_/ghu_/ghr_,
#     github_pat_, sk-, AKIA, private keys, Slack), this machine's host name or user name, or a word listed in
#     SYSONE_OUTGOING_DENY (comma-separated, e.g. project or customer names from the sensitive material);
#   - adds a binary file or a file over 1 MB;
#   - has a commit whose message has any of the above, or whose author/committer e-mail is a machine identity
#     (*.local, *.lan, *.localdomain, *.internal, (none), or containing the host name).
# Nothing it reports should be pasted into GitHub: fix the files instead.
set -uo pipefail

RESULT_PATHS='^(benchmarks/results/x86/|profiles/|kernels-simt/artifacts/)'
CODE_PATHS='^(src/|benchmarks/s1/|benchmarks/vllm/|tests/[^/]+\.rs$|docs/|kernels-simt/src/|kernels-simt/Cargo\.(toml|lock)$|Cargo\.(toml|lock)$|build\.rs$|README\.md$|scripts/x86/(run_tests|run_bench|setup)\.sh$)'
SAFETY_PATHS='^(scripts/gpu\.sh$|scripts/doctor\.sh$|scripts/x86/check_outgoing\.sh$|scripts/x86/lib\.sh$|scripts/tests/|CUDA_RUST_REV$|\.gitignore$)'
# Never from the x86 host, under any flag: gates and tolerances (INTEGRITY), model weights, traces, secrets.
DENY_PATHS='(^tests/tolerances\.toml$|^src/tolerances\.rs$|\.(safetensors|gguf|pt|pth|bin|npy|npz|nsys-rep|ncu-rep|sqlite|qdrep|pem|key|p12)$|(^|/)\.env$|(^|/)id_[a-z0-9]+$)'
MAX_BYTES=$((1024 * 1024))

mode=results base="" files=()
while [ $# -gt 0 ]; do
    case "$1" in
        --code) [ "$mode" = safety ] || mode=code ;;
        --safety) mode=safety ;;
        --base) base="$2"; shift ;;
        --files) shift; files=("$@"); mode=files; break ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "check_outgoing: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

problems=0
report() { printf 'check_outgoing: %s\n' "$*"; problems=$((problems + 1)); }

# Words that identify this machine or its user (whole words, case-insensitive, >= 3 characters).
deny_words=()
for w in "$(hostname -s 2>/dev/null)" "$(hostname -f 2>/dev/null)" "${USER:-}" "$(id -un 2>/dev/null)"; do
    [ "${#w}" -ge 3 ] && deny_words+=("$w")
done
IFS=, read -r -a extra <<< "${SYSONE_OUTGOING_DENY:-}"
for w in "${extra[@]}"; do w="${w## }"; w="${w%% }"; [ -n "$w" ] && deny_words+=("$w"); done

# scan: reads "label<TAB>text" lines on stdin and reports every sensitive pattern found.
scan() {
    # Only the text is scanned, never the label (a file path of this machine in --files mode).
    local all tmp labels
    all=$(mktemp) tmp=$(mktemp) labels=$(mktemp)
    cat > "$all"
    awk '{ i = index($0, "\t"); if (i == 0) { print "?" > L; print $0 } else { print substr($0, 1, i - 1) > L; print substr($0, i + 1) } }' \
        L="$labels" "$all" > "$tmp"
    # grep -n output on the text -> "label<TAB>text".
    relabel() { awk 'NR == FNR { lab[NR] = $0; next } { i = index($0, ":"); print lab[substr($0, 1, i - 1)] "\t" substr($0, i + 1) }' "$labels" -; }
    local -a pats=(
        'home path|(/home/|/Users/|/root/)'
        'GPU/MIG UUID|(GPU|MIG)-[0-9A-Fa-f]{8}'
        'PCI bus id|[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]'
        'MAC address|([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}'
        'serial number|[Ss][Ee][Rr][Ii][Aa][Ll]([ _-]?([Nn]umber|[Nn]o|[Nn]um))?"?[[:space:]]*[:=,]|gpu_serial'
        'long integer (serial?)|(^|[^0-9A-Za-z._])[0-9]{13,}([^0-9A-Za-z._]|$)'
        'Hugging Face token|hf_[A-Za-z0-9]{20,}'
        'GitHub token|(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'
        'API key|(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,})'
        'private key|-----BEGIN [A-Z ]*PRIVATE KEY-----'
    )
    local p kind re
    for p in "${pats[@]}"; do
        kind="${p%%|*}" re="${p#*|}"
        grep -nE -- "$re" "$tmp" | head -5 | relabel | while IFS=$'\t' read -r label text; do
            report "$label: $kind: ${text:0:160}"
        done
    done
    # IPv4 (octets <= 255; loopback/any allowed) and e-mail (noreply allowed), checked per match.
    awk '{
            s = $0
            while (match(s, /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/)) {
                ip = substr(s, RSTART, RLENGTH); pre = (RSTART > 1) ? substr(s, RSTART - 1, 1) : ""; post = substr(s, RSTART + RLENGTH, 1)
                s = substr(s, RSTART + RLENGTH)
                if (pre ~ /[0-9.]/ || post ~ /[0-9.]/) continue
                n = split(ip, o, "."); ok = 1
                for (i = 1; i <= 4; i++) if (length(o[i]) > 3 || o[i] + 0 > 255) ok = 0
                if (ok && ip != "0.0.0.0" && ip != "127.0.0.1") { print NR ":IPv4 address: " ip; break }
            }
        }' "$tmp" | head -5 | relabel | while IFS=$'\t' read -r label text; do report "$label: $text"; done
    grep -noE -- '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}' "$tmp" |
        grep -vE ':(noreply@anthropic\.com|[A-Za-z0-9._+-]+@users\.noreply\.github\.com|noreply@github\.com)$' |
        head -5 | relabel | while IFS=$'\t' read -r label m; do report "$label: e-mail address: $m"; done
    local w
    for w in "${deny_words[@]}"; do
        grep -niwF -- "$w" "$tmp" | head -3 | relabel | while IFS=$'\t' read -r label text; do
            report "$label: machine/user/denied word '$w': ${text:0:160}"
        done
    done
    rm -f "$all" "$tmp" "$labels"
}

# scan() reports from pipeline subshells, so count its output lines here. `lastpipe` (set below) runs this in the
# main shell when it ends a pipeline.
run_scan() {
    local out
    out=$(scan)
    if [ -n "$out" ]; then printf '%s\n' "$out"; problems=$((problems + $(printf '%s\n' "$out" | wc -l))); fi
}
shopt -s lastpipe

if [ "$mode" = files ]; then
    [ "${#files[@]}" -gt 0 ] || { echo "check_outgoing: --files needs files" >&2; exit 2; }
    text=()
    for f in "${files[@]}"; do
        [ -f "$f" ] || { report "$f: not a file"; continue; }
        size=$(stat -c %s "$f")
        [ "$size" -le "$MAX_BYTES" ] || report "$f: file over 1 MB ($size bytes)"
        if [ -s "$f" ] && ! grep -qI . "$f"; then report "$f: binary file (results must be small text/JSON)"; else text+=("$f"); fi
    done
    for f in "${text[@]}"; do awk -v f="$f" '{ print f ":" NR "\t" $0 }' "$f"; done | run_scan
    [ "$problems" = 0 ] && { echo "check_outgoing: OK (${#files[@]} files)"; exit 0; }
    echo "check_outgoing: FAILED ($problems problems): fix them; do not push, and do not paste the flagged text anywhere"
    exit 1
fi

git rev-parse --git-dir >/dev/null 2>&1 || { echo "check_outgoing: not in a git repository" >&2; exit 2; }
cd "$(git rev-parse --show-toplevel)"
branch=$(git rev-parse --abbrev-ref HEAD)
case "$branch" in main|master|HEAD) report "on '$branch': work on a branch x86/<issue>-<slug>, never push to main" ;; esac
if [ -z "$base" ]; then
    if git rev-parse -q --verify origin/main >/dev/null; then base=origin/main; else base=main; fi
fi
mb=$(git merge-base "$base" HEAD 2>/dev/null) || { echo "check_outgoing: no merge base with $base (git fetch first)" >&2; exit 2; }

# Paths touched by the branch's commits and by staged changes (unstaged and untracked files are not pushed).
paths=$( { git diff --name-only --no-renames "$mb" HEAD; git diff --name-only --no-renames --cached; } | sort -u)
allowed="$RESULT_PATHS"
[ "$mode" = code ] && allowed="$RESULT_PATHS|$CODE_PATHS"
[ "$mode" = safety ] && allowed="$RESULT_PATHS|$CODE_PATHS|$SAFETY_PATHS"
while IFS= read -r p; do
    [ -n "$p" ] || continue
    if [[ "$p" =~ $DENY_PATHS ]]; then report "$p: never allowed from the x86 host (gate/tolerance, model, trace or secret)"
    elif ! [[ "$p" =~ $allowed ]]; then
        hint="allowed: results paths"; [ "$mode" = results ] && hint="$hint (use --code for a code PR)"
        [[ "$p" =~ $SAFETY_PATHS ]] && hint="safety script: needs agreement in an issue, then --safety"
        report "$p: outside the allow-list ($hint)"
    fi
done <<< "$paths"

# Sizes and binaries of the files as they will be pushed (staged version if staged, else HEAD).
while IFS=$'\t' read -r add del p; do
    [ -n "$p" ] || continue
    if [ "$add" = - ]; then report "$p: binary file (results must be small text/JSON)"; fi
done < <( { git diff --numstat --no-renames "$mb" HEAD; git diff --numstat --no-renames --cached; } | sort -u)
while IFS= read -r p; do
    [ -n "$p" ] || continue
    size=$(git cat-file -s ":$p" 2>/dev/null || git cat-file -s "HEAD:$p" 2>/dev/null || echo 0)
    [ "$size" -le "$MAX_BYTES" ] || report "$p: file over 1 MB ($size bytes)"
done <<< "$paths"

# Added lines of the branch and the index.
{ git diff -U0 --no-color --no-renames "$mb" HEAD; git diff -U0 --no-color --no-renames --cached; } |
    awk '/^\+\+\+ /{ f = substr($0, 7); next } /^\+/{ print f "\t" substr($0, 2) }' | run_scan

# Commit messages and identities.
git log --format='%H%x09%B' "$mb..HEAD" | awk -F'\t' 'NF > 1 { h = substr($1, 1, 10); print "commit " h "\t" $2; next } { print "commit " h "\t" $0 }' |
    grep -v $'\tCo-Authored-By: Claude' | run_scan
host=$(hostname -s 2>/dev/null || true)
while IFS=$'\t' read -r h ae ce; do
    for e in "$ae" "$ce"; do
        if [[ "$e" =~ \.(local|lan|localdomain|internal|home)$ || "$e" == *"(none)"* ]] || { [ -n "$host" ] && [[ "${e,,}" == *"${host,,}"* ]]; }; then
            report "commit ${h:0:10}: machine identity in author/committer e-mail '$e' (set user.name/user.email to the identity the user chose)"
        fi
    done
done < <(git log --format='%H%x09%ae%x09%ce' "$mb..HEAD")

if [ "$problems" = 0 ]; then echo "check_outgoing: OK ($branch vs $base, $(printf '%s\n' "$paths" | grep -c .) paths)"; exit 0; fi
echo "check_outgoing: FAILED ($problems problems): fix them; do not push, and do not paste the flagged text anywhere"
exit 1
