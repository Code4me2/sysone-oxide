#!/usr/bin/env bash
# Check the build and run prerequisites for sysone (docs/PLATFORM.md) and print the fix for anything missing.
# Usage: scripts/doctor.sh [--simt] [--core]
#   --simt  also checks the cuda-oxide toolchain for SIMT kernels
#   --core  only build/test/benchmark prerequisites: Docker and Hugging Face auth are reported, not required
set -u
simt=0 core=0
for a in "$@"; do
  case "$a" in --simt) simt=1 ;; --core) core=1 ;; *) echo "doctor: unknown argument $a" >&2; exit 2 ;; esac
done
fail=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FIX   %s\n        -> %s\n' "$1" "$2"; fail=1; }
info() { printf '  info  %s\n' "$1"; }

arch=$(uname -m)
cuda=${CUDA_TOOLKIT_PATH:-/usr/local/cuda-13}
echo "sysone doctor ($arch, toolkit $cuda)"

# CUDA toolkit >= 13.2 with the Tile IR assembler (cuTile on Blackwell); 13.3+ for block-scaled FP4 MMA.
if [ -x "$cuda/bin/nvcc" ]; then
  v=$("$cuda/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
  if printf '13.3\n%s\n' "$v" | sort -V -C; then ok "CUDA toolkit $v"; else
    bad "CUDA toolkit $v (need >= 13.3 for NVFP4 block-scaled MMA)" "sudo apt-get install -y cuda-toolkit-13-4 cuda-tileiras-13-4"; fi
  if [ -x "$cuda/bin/tileiras" ]; then ok "tileiras"; else bad "tileiras missing" "sudo apt-get install -y cuda-tileiras-${v/./-}"; fi
else
  bad "no nvcc under $cuda" "install a CUDA 13.x toolkit, or set CUDA_TOOLKIT_PATH"
fi

# Driver: R580+ (CUDA 13.0 driver API). Newer toolkits run on it via minor-version compatibility.
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
if [ -n "$drv" ] && [ "${drv%%.*}" -ge 580 ]; then ok "driver $drv"; else bad "driver ${drv:-not found}" "install an R580+ NVIDIA driver"; fi
here=$(cd "$(dirname "$0")/.." && pwd)
gpus=$(nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null)
[ -n "$gpus" ] || info "GPU: none"
while IFS=, read -r i name cc; do
    [ -n "$i" ] || continue
    cc="${cc// /}"; arch="sm_${cc/./}"
    # The SIMT GDN backend needs a PTX artifact built for exactly this architecture (docs/G1.md).
    if [ -f "$here/kernels-simt/artifacts/gdn_simt.$arch.ptx" ] || [ -f "$here/kernels-simt/artifacts/gdn_simt.${arch}a.ptx" ]; then
        simt_note="SIMT artifact $arch present"
    else simt_note="no SIMT artifact for $arch (GdnBackend::Simt unavailable; kernels-simt/build.sh $arch)"; fi
    info "GPU $i:${name} ($arch); $simt_note"
done <<< "$gpus"
# scripts/gpu.sh serializes GPU jobs with flock (util-linux).
if command -v flock >/dev/null; then ok "flock"; else bad "flock missing" "sudo apt-get install -y util-linux"; fi

# bindgen (cuda-bindings) needs libclang plus clang's resource headers (stddef.h); a bare libclang1 is not enough.
inc=$(clang -print-resource-dir 2>/dev/null)/include
if command -v clang >/dev/null && [ -f "$inc/stddef.h" ]; then ok "clang $(clang -dumpversion) with resource headers"; else
  bad "clang / libclang resource headers missing" "sudo apt-get install -y clang libclang-dev cmake build-essential pkg-config"; fi

# Rust: the pinned toolchain is installed on demand by rustup from rust-toolchain.toml.
if command -v rustup >/dev/null || [ -x "$HOME/.cargo/bin/rustup" ]; then ok "rustup"; else
  bad "rustup missing" "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none"; fi

# cuda-rust checkout next to this repo, at the pinned revision (CUDA_RUST_REV).
pin=$(grep -v '^#' "$here/CUDA_RUST_REV" 2>/dev/null | head -1 | tr -d ' ')
if [ -d "$here/../cuda-rust/cutile-rs" ]; then
  have=$(git -C "$here/../cuda-rust" rev-parse HEAD 2>/dev/null)
  if [ -z "$pin" ]; then bad "CUDA_RUST_REV missing or empty" "restore it from git: git checkout -- CUDA_RUST_REV"
  elif [ "$have" != "$pin" ]; then
    bad "../cuda-rust is at ${have:-an unknown revision}, sysone is pinned to $pin (CUDA_RUST_REV)" \
        "git -C ../cuda-rust fetch origin && git -C ../cuda-rust checkout $pin"
  elif [ -n "$(git -C "$here/../cuda-rust" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    bad "../cuda-rust at the pinned ${pin:0:12} but has local modifications" "git -C ../cuda-rust stash (or check out a clean copy); sysone must build against the pinned upstream source"
  else ok "../cuda-rust at the pinned revision ${pin:0:12}"; fi
else
  bad "../cuda-rust missing" "git clone https://github.com/NVIDIA/cuda-rust.git ../cuda-rust && git -C ../cuda-rust checkout $pin"; fi

# With --core, Docker and HF auth are reported but not required (build, tests and benchmarks need neither).
opt() { if [ "$core" = 1 ]; then info "optional: $1 (fix: $2)"; else bad "$1" "$2"; fi; }

# Docker (reference runtime and vLLM baseline in NGC containers).
if docker info >/dev/null 2>&1; then ok "docker access"; else
  if id -nG "$USER" | grep -qw docker || grep -q "^docker:.*\b$USER\b" /etc/group; then
    opt "docker group not active in this session" "log out and back in (or run: newgrp docker)"
  else opt "no docker access" "sudo usermod -aG docker $USER, then log in again"; fi
fi

# Hugging Face auth (private checkpoints such as the NVFP4 artifact).
if hf auth whoami >/dev/null 2>&1; then ok "hf auth"; else
  opt "not logged in to Hugging Face" "hf auth login"; fi

if [ "$simt" = 1 ]; then
  # cuda-oxide: pinned Rust 1.99.1 with rustc-dev/llvm-tools, and llc 21+ with the NVPTX target.
  llc=$(command -v llc-23 || command -v llc-22 || command -v llc-21 || true)
  if [ -n "$llc" ] && "$llc" --version | grep -q nvptx; then ok "$(basename "$llc") with NVPTX"; else
    bad "LLVM 21+ (llc) not found; Ubuntu $(. /etc/os-release; echo "$VERSION_ID") does not package it" \
        "wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 21"; fi
fi

[ "$fail" = 0 ] && echo "all checks passed" || { echo "some checks need fixing (see FIX lines)"; exit 1; }
