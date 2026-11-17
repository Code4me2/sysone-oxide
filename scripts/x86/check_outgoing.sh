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
# It checks EVERY commit the push would upload (mb..HEAD, where mb is the merge base with the base branch), not
# only the branch's net change: a leak committed and removed again in a later commit is still in the pushed history,
# so it fails here and must be removed by rewriting the unpushed commits (docs/x86/AGENT.md §5.4). It fails when
#   - the branch is main/master, or has a merge commit (rebase on origin/main instead), or any commit or the index
#     touches a path outside the allow-list (results PR: benchmarks/results/x86/**, profiles/**,
#     kernels-simt/artifacts/**), or a gate/tolerance file (never allowed), or a model/trace/secret file;
#   - any commit or the index adds a line with: an absolute path outside a short allow-list (/tmp/sysone*,
#     /usr/local/cuda*, /usr/{bin,lib,include}/, /dev/null, /proc/, /sys/, /v1/ routes, /path/to/; home paths
#     always fail; a path right after '<' or '>' is a tag or placeholder and not checked),
#     ~/ or $HOME, an IPv4 address (except 0.0.0.0, 127.0.0.1) or IPv6 address (except ::1), an e-mail address
#     (except noreply ones and SYSONE_OUTGOING_ALLOW_EMAIL), a UUID (GPU-/MIG- or bare, any case), a PCI bus id,
#     a MAC address, a serial number (a "serial" field or a bare integer of 13+ digits), an access token (hf_,
#     ghp_/gho_/ghs_/ghu_/ghr_, github_pat_, sk-, AKIA, private keys, Slack), this machine's host name or user
#     name, or a word listed in SYSONE_OUTGOING_DENY (comma-separated, e.g. project or customer names from the
#     sensitive material);
#   - any commit or the index adds a binary file or a file over 1 MB (every new blob is checked);
#   - a commit message has any of the above, or an author/committer e-mail is not allowed (only noreply addresses
#     and the comma-separated SYSONE_OUTGOING_ALLOW_EMAIL, set locally by the user, never in the repo) or is a
#     machine identity, or an author/committer name is this machine's host or user name.
# Nothing it reports should be pasted into GitHub: remove it from the unpushed commits instead (AGENT.md §5.4).
# Diffs are taken with --no-ext-diff --no-textconv, so local diff drivers and .gitattributes cannot hide content.
set -uo pipefail

RESULT_PATHS='^(benchmarks/results/x86/|profiles/|kernels-simt/artifacts/)'
CODE_PATHS='^(src/|benchmarks/s1/|benchmarks/vllm/|tests/[^/]+\.rs$|docs/|kernels-simt/src/|kernels-simt/README\.md$|kernels-simt/Cargo\.(toml|lock)$|Cargo\.(toml|lock)$|build\.rs$|README\.md$|scripts/x86/(run_tests|run_bench|run_bulk|setup)\.sh$)'
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
        -h|--help) sed -n '2,31p' "$0"; exit 0 ;;
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

# E-mail addresses that may appear (commit identities, attribution lines): noreply ones, plus the comma-separated
# SYSONE_OUTGOING_ALLOW_EMAIL the user sets locally for the commit identity they chose.
IFS=, read -r -a allow_emails <<< "${SYSONE_OUTGOING_ALLOW_EMAIL:-}"
allowed_email() {
    local e="${1,,}" a
    [[ "$e" =~ ^(noreply@anthropic\.com|[a-z0-9._+-]+@users\.noreply\.github\.com|noreply@github\.com)$ ]] && return 0
    for a in "${allow_emails[@]}"; do a="${a// /}"; [ -n "$a" ] && [ "$e" = "${a,,}" ] && return 0; done
    return 1
}

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
        'GPU/MIG UUID|[GgMm][PpIi][UuGg]-[0-9A-Fa-f]{8}'
        'UUID|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}'
        'home path variable|(~/|\$HOME([^A-Za-z0-9_]|$)|\$\{HOME\}|%USERPROFILE%)'
        'IPv6 address|(^|[^0-9A-Za-z_:])([0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}([^0-9A-Za-z_:]|$)'
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
    # IPv4 (octets <= 255; loopback/any allowed), checked per match. A '.' right after the address ends it unless a
    # digit follows (a sentence end "... 10.0.0.5." is an address; "1.2.3.4.5" is a version string).
    awk '{
            s = $0
            while (match(s, /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/)) {
                ip = substr(s, RSTART, RLENGTH); pre = (RSTART > 1) ? substr(s, RSTART - 1, 1) : ""
                post = substr(s, RSTART + RLENGTH, 1); post2 = substr(s, RSTART + RLENGTH + 1, 1)
                s = substr(s, RSTART + RLENGTH)
                if (pre ~ /[0-9.]/ || post ~ /[0-9]/ || (post == "." && post2 ~ /[0-9]/)) continue
                n = split(ip, o, "."); ok = 1
                for (i = 1; i <= 4; i++) if (length(o[i]) > 3 || o[i] + 0 > 255) ok = 0
                if (ok && ip != "0.0.0.0" && ip != "127.0.0.1") { print NR ":IPv4 address: " ip; break }
            }
        }' "$tmp" | head -5 | relabel | while IFS=$'\t' read -r label text; do report "$label: $text"; done
    # Compressed IPv6 ("fe80::1c2:3", "2001:db8::1"): hex groups around "::", with a digit; "::1" and Rust/C++ paths
    # such as "u8::MAX" (not hex, or glued to an identifier) are not addresses.
    grep -noE -- '(^|[^0-9A-Za-z_:])[0-9A-Fa-f:]*::[0-9A-Fa-f:]*([^0-9A-Za-z_:]|$)' "$tmp" |
        awk '{ i = index($0, ":"); n = substr($0, 1, i - 1); a = substr($0, i + 1); gsub(/[^0-9A-Fa-f:]/, "", a)
               if (a ~ /[0-9]/ && a != "::1" && a ~ /[0-9A-Fa-f]/ && length(a) >= 4) {
                   ok = 1; k = split(a, g, ":"); for (j = 1; j <= k; j++) if (length(g[j]) > 4) ok = 0
                   if (ok && k <= 8) print n ":IPv6 address: " a } }' |
        head -5 | relabel | while IFS=$'\t' read -r label text; do report "$label: $text"; done
    # Absolute paths: anything absolute outside a short allow-list of system/tool paths (results and docs never need
    # a path of this machine; the runners already replace the repo and model paths with <repo> and <model>). A path
    # right after '<' or '>' is not checked: closing tags (</think>) and placeholders (<models>/elephant-1).
    grep -noE -- '(^|[][[:space:]"'"'"'`=(,;:{}|*])/[A-Za-z0-9_+-][A-Za-z0-9._+-]*(/[A-Za-z0-9._+-]*)*' "$tmp" |
        awk '{ i = index($0, ":"); n = substr($0, 1, i - 1); a = substr($0, i + 1); sub(/^[^\/]/, "", a)
               if (a ~ /^\/(home|Users|root)(\/|$)/) next   # reported as "home path" already
               if (a ~ /^\/(tmp\/sysone|usr\/local\/cuda|usr\/(bin|lib|lib64|include)\/|bin\/|dev\/null$|dev\/shm$|proc\/|sys\/|v1\/|v1$|health$|metrics$|path\/to\/)/) next
               print n ":absolute path outside the allow-list: " a }' |
        head -5 | relabel | while IFS=$'\t' read -r label text; do report "$label: $text"; done
    grep -noE -- '[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}' "$tmp" |
        while IFS= read -r m; do allowed_email "${m#*:}" || printf '%s\n' "$m"; done |
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

# Everything below looks at every commit in mb..HEAD (what `git push` uploads) and at the index (what the next
# commit adds), never only at the net diff. No external diff drivers or textconv filters.
D=(--no-ext-diff --no-textconv --no-renames --no-color)

merges=$(git rev-list --merges "$mb..HEAD")
[ -z "$merges" ] || report "merge commit(s) $(printf '%s' "$merges" | cut -c1-10 | tr '\n' ' ')in the branch: rebase on origin/main instead of merging (before the first push)"

# Paths touched by any commit of the branch (also ones a later commit reverted or deleted) and by staged changes.
paths=$( { git log "${D[@]}" --format= --name-only --diff-merges=dense-combined "$mb..HEAD"; git diff "${D[@]}" --name-only --cached; } | sort -u)
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

# Every blob the push would upload (objects of mb..HEAD not reachable from mb), plus staged blobs: size and binary.
check_blob() {   # check_blob OBJECT SIZE PATH
    [ "$2" -le "$MAX_BYTES" ] || { report "$3: file over 1 MB ($2 bytes) in the branch history"; return; }
    local blob_file status nul_prefix
    if ! blob_file=$(mktemp); then
        report "$3: could not create temporary blob file"
        return
    fi
    # Materialize the bounded blob: head/grep in a pipe can SIGPIPE git under pipefail, even for valid text.
    if ! git cat-file blob "$1" > "$blob_file"; then
        report "$3: could not read blob"
    else
        grep -I . "$blob_file" >/dev/null
        status=$?
        if [ "$status" -gt 1 ]; then
            report "$3: could not inspect blob"
        # grep can match an early line before a later NUL. Check every bounded byte, not just the first 8 KiB.
        elif { [ "$2" -gt 0 ] && [ "$status" -ne 0 ]; } || IFS= read -r -d '' nul_prefix < "$blob_file"; then
            report "$3: binary file (results must be small text/JSON)"
        fi
    fi
    rm -f -- "$blob_file" || report "$3: could not remove temporary blob file"
}
while read -r obj type size path; do
    [ "$type" = blob ] && check_blob "$obj" "$size" "$path"
done < <(git rev-list --objects HEAD --not "$mb" | git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)')
while read -r _ obj _ path; do
    [ -n "$obj" ] && [ "$obj" != 0000000000000000000000000000000000000000 ] || continue
    check_blob "$obj" "$(git cat-file -s "$obj")" "$path"
done < <(git diff-index "${D[@]}" --cached HEAD | awk -F'\t' '{ split($1, m, " "); print m[1], m[4], m[5], $2 }')

# Added lines of every commit (merges: their combined diff) and of the index, labelled "<path>@<commit>".
added_lines() {
    awk '
        /^commit [0-9a-z]+$/ { c = substr($2, 1, 10); hdr = 0; next }
        /^diff --cc / { hdr = 1; nc = 2; next }
        /^diff --(git|combined) / { hdr = 1; nc = 1; next }
        hdr && /^\+\+\+ / { f = substr($0, 5); sub(/^b\//, "", f); next }
        /^@@/ { hdr = 0; if ($0 ~ /^@@@/) { match($0, /^@+/); nc = RLENGTH - 1 } next }
        hdr { next }
        { pre = substr($0, 1, nc); if (pre ~ /\+/ && pre !~ /-/) print f "@" c "\t" substr($0, nc + 1) }'
}
{ git log "${D[@]}" -U0 -p --diff-merges=dense-combined --format='commit %H' "$mb..HEAD"
  echo "commit index"; git diff "${D[@]}" -U0 --cached; } | added_lines | run_scan

# Commit messages, every line (the attribution lines carry only noreply addresses, which pass).
git log --format='%H%x09%B' "$mb..HEAD" | awk -F'\t' 'NF > 1 { h = substr($1, 1, 10); print "commit " h "\t" $2; next } { print "commit " h "\t" $0 }' |
    run_scan

# Commit identities: e-mails must be allowed ones, never machine identities; names never the host/user name.
host=$(hostname -s 2>/dev/null || true)
while IFS=$'\t' read -r h an ae cn ce; do
    for e in "$ae" "$ce"; do
        if [[ "${e,,}" =~ \.(local|lan|localdomain|internal|home)$ || "$e" == *"(none)"* ]] || { [ -n "$host" ] && [[ "${e,,}" == *"${host,,}"* ]]; }; then
            report "commit ${h:0:10}: machine identity in author/committer e-mail (set user.name/user.email to the identity the user chose)"
        elif ! allowed_email "$e"; then
            report "commit ${h:0:10}: author/committer e-mail is not a noreply address or in SYSONE_OUTGOING_ALLOW_EMAIL (set locally by the user)"
        fi
    done
    for nm in "$an" "$cn"; do
        for w in "${deny_words[@]}"; do
            [[ "${nm,,}" =~ (^|[^a-z0-9])"${w,,}"([^a-z0-9]|$) ]] && report "commit ${h:0:10}: author/committer name contains a machine/user/denied word"
        done
    done
done < <(git log --format='%H%x09%an%x09%ae%x09%cn%x09%ce' "$mb..HEAD")

if [ "$problems" = 0 ]; then echo "check_outgoing: OK ($branch vs $base, $(printf '%s\n' "$paths" | grep -c .) paths)"; exit 0; fi
echo "check_outgoing: FAILED ($problems problems): fix them; do not push, and do not paste the flagged text anywhere"
exit 1
