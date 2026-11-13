# Shared helpers of the x86 runners (sourced, not run). docs/x86/AGENT.md describes the rules they implement:
# results are small text/JSON under benchmarks/results/x86/<card-slug>/<date>/, with a sanitised metadata.json
# holding ONLY: GPU name, compute capability, memory size, driver and CUDA versions, power limit, clocks, sysone git
# sha and cuda-rust rev (plus the run's own parameters). Never host names, user names, paths outside the repo, IPs,
# serial numbers, UUIDs, bus ids, environment dumps or process lists.

X86_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMI="${SYSONE_NVIDIA_SMI:-nvidia-smi}"
# Local, never committed: stderr of every job (may hold paths and process names). target/ is git-ignored.
X86_LOGS="$X86_ROOT/target/x86-logs"

x86_die() { echo "x86: $*" >&2; exit 2; }

# GPU indices (nvidia-smi order = PCI bus order; gpu.sh makes CUDA use the same order).
x86_gpu_indices() { "$SMI" --query-gpu=index --format=csv,noheader | tr -d ' '; }

# Field of GPU <index>: x86_gpu_field 0 name
x86_gpu_field() { "$SMI" -i "$1" --query-gpu="$2" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//; s/ *$//'; }

# "NVIDIA RTX PRO 6000 Blackwell Workstation Edition" -> "rtx-pro-6000-blackwell-workstation-edition"
x86_card_slug() {
    printf '%s' "$1" | tr 'A-Z' 'a-z' | sed -E 's/^nvidia[ -]+//; s/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
}

# "12.0" -> "sm_120"
x86_sm() { printf 'sm_%s' "$(printf '%s' "$1" | tr -d '.')"; }

# A value that may go into metadata: only plain characters, else "redacted".
x86_safe() {
    if [[ "$1" =~ ^[][A-Za-z0-9\ ._:/()+,\;=-]*$ ]] && [ "${#1}" -le 200 ]; then printf '%s' "$1"; else printf 'redacted'; fi
}

# Fresh results directory benchmarks/results/x86/<slug>/<date>/<run>[-n] for GPU <index> (or for slug `host` when
# the index is `host`: GPU-free tests); prints its path.
x86_results_dir() {
    local idx="$1" run="$2" slug date base d k=1
    if [ "$idx" = host ]; then slug=host; else slug=$(x86_card_slug "$(x86_gpu_field "$idx" name)"); fi
    [ -n "$slug" ] || x86_die "cannot read the name of GPU $idx"
    date=$(date -u +%Y-%m-%d)
    base="$X86_ROOT/benchmarks/results/x86/$slug/$date/$run"
    d="$base"
    while [ -e "$d" ]; do k=$((k + 1)); d="$base-$k"; done
    mkdir -p "$d"
    printf '%s' "$d"
}

x86_cuda_rust_rev() {
    local r
    r=$(git -C "$X86_ROOT/../cuda-rust" rev-parse HEAD 2>/dev/null) || { printf 'missing'; return; }
    [ -z "$(git -C "$X86_ROOT/../cuda-rust" status --porcelain --untracked-files=no 2>/dev/null)" ] || r="$r-dirty"
    printf '%s' "$r"
}

x86_sysone_sha() {
    local r
    r=$(git -C "$X86_ROOT" rev-parse HEAD 2>/dev/null) || { printf 'unknown'; return; }
    [ -z "$(git -C "$X86_ROOT" status --porcelain --untracked-files=no -- . ':!benchmarks/results' ':!profiles' 2>/dev/null)" ] || r="$r-dirty"
    printf '%s' "$r"
}

# x86_write_metadata DIR INDEX KIND NOTE: the sanitised metadata.json (the ONLY machine description that leaves).
x86_write_metadata() {
    local dir="$1" idx="$2" kind="$3" note="$4" f q
    f() { x86_safe "$(x86_gpu_field "$idx" "$1")"; }
    q() { printf '"%s"' "$1"; }
    local cuda_drv toolkit
    cuda_drv=$("$SMI" 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
    toolkit=$("${CUDA_TOOLKIT_PATH:-/usr/local/cuda-13}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    {
        echo "{"
        echo "  \"schema\": \"sysone-x86-metadata/1\","
        echo "  \"kind\": $(q "$(x86_safe "$kind")"),"
        echo "  \"date_utc\": $(q "$(date -u +%Y-%m-%dT%H:%MZ)"),"
        echo "  \"gpu_index\": $(q "$(x86_safe "$idx")"),"
        echo "  \"gpu_name\": $(q "$(f name)"),"
        echo "  \"compute_capability\": $(q "$(f compute_cap)"),"
        echo "  \"memory_total_mib\": $(q "$(f memory.total)"),"
        echo "  \"driver_version\": $(q "$(f driver_version)"),"
        echo "  \"driver_cuda_version\": $(q "$(x86_safe "$cuda_drv")"),"
        echo "  \"cuda_toolkit_version\": $(q "$(x86_safe "$toolkit")"),"
        echo "  \"power_limit_w\": $(q "$(f power.limit)"),"
        echo "  \"power_max_limit_w\": $(q "$(f power.max_limit)"),"
        echo "  \"clocks_max_sm_mhz\": $(q "$(f clocks.max.sm)"),"
        echo "  \"clocks_max_mem_mhz\": $(q "$(f clocks.max.mem)"),"
        echo "  \"clocks_sm_mhz_at_start\": $(q "$(f clocks.sm)"),"
        echo "  \"gpus_on_host\": $(q "$(x86_gpu_indices | wc -l)"),"
        echo "  \"sysone_git_sha\": $(q "$(x86_safe "$(x86_sysone_sha)")"),"
        echo "  \"cuda_rust_rev\": $(q "$(x86_safe "$(x86_cuda_rust_rev)")"),"
        echo "  \"cuda_rust_rev_pinned\": $(q "$(x86_safe "$(grep -v '^#' "$X86_ROOT/CUDA_RUST_REV" | head -1)")"),"
        echo "  \"note\": $(q "$(x86_safe "$note")")"
        echo "}"
    } > "$dir/metadata.json"
}

# Strip local identifiers from a job's stdout in place: the repo root, the model directory, $HOME, host and user
# names. check_outgoing.sh --files then has the last word.
x86_sanitize() {
    local f="$1" host user
    host=$(hostname -s 2>/dev/null || true)
    user=$(id -un 2>/dev/null || true)
    local args=(-e "s#${X86_ROOT//#/\\#}#<repo>#g")
    [ -n "${SYSONE_MODEL_DIR:-}" ] && args+=(-e "s#${SYSONE_MODEL_DIR//#/\\#}#<model>#g")
    [ -n "${SYSONE_NVFP4_DIR:-}" ] && args+=(-e "s#${SYSONE_NVFP4_DIR//#/\\#}#<nvfp4-model>#g")
    [ -n "${HOME:-}" ] && args+=(-e "s#${HOME//#/\\#}#<home>#g")
    [ "${#host}" -ge 3 ] && args+=(-e "s#\\b$host\\b#<host>#g")
    [ "${#user}" -ge 3 ] && args+=(-e "s#\\b$user\\b#<user>#g")
    sed -i "${args[@]}" "$f"
}

# x86_job DIR NAME GPU CMD...: run CMD on GPU (through scripts/gpu.sh, so under that GPU's lock) with stdout to
# DIR/NAME.txt (sanitised) and stderr to the local log; appends "NAME<TAB>exit<TAB>seconds" to DIR/summary.tsv.
x86_job() {
    local dir="$1" name="$2" gpu="$3"; shift 3
    mkdir -p "$X86_LOGS"
    local rel="${dir#"$X86_ROOT"/benchmarks/results/x86/}"
    local log="$X86_LOGS/${rel//\//_}_$name.log"
    local t0 rc
    t0=$(date +%s)
    echo "x86: [gpu $gpu] $name" >&2
    "$X86_ROOT/scripts/gpu.sh" --gpu "$gpu" "$@" > "$dir/$name.txt" 2> "$log"
    rc=$?
    x86_sanitize "$dir/$name.txt"
    printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$dir/summary.tsv"
    [ "$rc" = 0 ] || echo "x86: [gpu $gpu] $name FAILED (exit $rc); local log: target/x86-logs/$(basename "$log")" >&2
    return "$rc"
}

# Final gate on a results directory: every file in it (dotfiles and subdirectories included) must pass
# check_outgoing.sh --files. On failure the directory is moved out of the repo tree (target/x86-quarantine/, or
# X86_QUARANTINE) so it cannot be committed by accident.
x86_seal() {
    local dir="$1" files=()
    mapfile -d '' files < <(find "$dir" -type f -print0)
    if [ "${#files[@]}" -gt 0 ] && "$X86_ROOT/scripts/x86/check_outgoing.sh" --files "${files[@]}" >&2; then return 0; fi
    [ "${#files[@]}" -gt 0 ] || echo "x86: $dir is empty" >&2
    local q="${X86_QUARANTINE:-$X86_ROOT/target/x86-quarantine}/$(date +%s)-$(basename "$dir")"
    mkdir -p "$(dirname "$q")" && mv "$dir" "$q"
    echo "x86: results in $dir failed the outgoing check and were moved to target/x86-quarantine/; fix the leak, rerun" >&2
    return 1
}
