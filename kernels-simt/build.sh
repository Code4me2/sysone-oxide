#!/usr/bin/env bash
# Build the cuda-oxide GDN kernels to PTX and refresh the checked-in artifact (docs/G1.md, "Build").
#
#   kernels-simt/build.sh [ARCH]      # ARCH defaults to sm_121 (GB10); e.g. sm_120, sm_100a
#   kernels-simt/build.sh check       # build + run the mma fragment probe (needs the GPU: run under scripts/gpu.sh)
#
# Writes artifacts/gdn_simt.<ARCH>.ptx and artifacts/gdn_simt.<ARCH>.manifest. sysone's build.rs embeds the PTX
# only if the manifest's source hash matches the current sources and the PTX hash matches the file; otherwise the
# SIMT backend reports the artifact as stale/missing at run time (never a silent fallback to the Tile backend).
# Needs the toolchain pinned in rust-toolchain.toml (1.99.1), cargo-oxide, and the CUDA toolkit; sysone's own
# build does not.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
export PATH="/usr/local/cuda/bin:$PATH"
# Use the published backend (built by `cargo oxide setup` in the cuda-rust checkout) instead of letting cargo-oxide
# build one in place inside ../../cuda-rust, which this repo must not modify.
export CUDA_OXIDE_BACKEND="${CUDA_OXIDE_BACKEND:-$HOME/.cargo/cuda-oxide/librustc_codegen_cuda.so}"

if [[ "${1:-}" == "check" ]]; then
    cd mma_check
    exec cargo oxide run --arch "${2:-sm_121}"
fi

arch="${1:-sm_121}"
[[ "$arch" =~ ^sm_[0-9]+[a-z]?$ ]] || { echo "bad arch '$arch' (want e.g. sm_121)" >&2; exit 2; }

# Source hash: sorted relative paths, each followed by the sha256 of its contents. build.rs computes the same.
source_hash() {
    { echo Cargo.toml; echo Cargo.lock; echo rust-toolchain.toml; find src -type f -name '*.rs'; } | LC_ALL=C sort |
        while read -r f; do printf '%s\n%s\n' "$f" "$(sha256sum < "$f" | cut -c1-64)"; done | sha256sum | cut -c1-64
}

cargo oxide build --arch "$arch"
ptx="gdn_simt.ptx"
[[ -f "$ptx" ]] || { echo "cargo oxide produced no $ptx" >&2; exit 1; }
grep -q "^\.target $arch" "$ptx" || { echo "$ptx does not target $arch:" >&2; grep '^\.target' "$ptx" >&2; exit 1; }
mkdir -p artifacts
cp "$ptx" "artifacts/gdn_simt.$arch.ptx"
{
    echo "source_sha256 $(source_hash)"
    echo "ptx_sha256 $(sha256sum < "artifacts/gdn_simt.$arch.ptx" | cut -c1-64)"
    echo "arch $arch"
    echo "toolchain $(rustc --version)"
    echo "cuda_oxide_backend_rev $(cat "$HOME/.cargo/cuda-oxide/source-rev.txt" 2>/dev/null || echo unknown)"
} > "artifacts/gdn_simt.$arch.manifest"
cat "artifacts/gdn_simt.$arch.manifest"

# Register / spill report (informational).
if command -v ptxas >/dev/null; then
    ptxas -arch="$arch" -v "artifacts/gdn_simt.$arch.ptx" -o /dev/null 2>&1 |
        grep -E "Compiling entry|registers|spill" | sed 's/^ptxas info *: //' || true
fi
