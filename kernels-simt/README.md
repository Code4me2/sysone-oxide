# kernels-simt: cuda-oxide GDN kernels

SIMT (warp-level, `mma.sync`) kernels for the GatedDeltaNet mixer, written in Rust with cuda-oxide
(`../../cuda-rust/cuda-oxide`). sysone loads them as PTX at run time; see docs/G1.md for the design and numbers.

- `src/main.rs`: the `#[kernel]` functions (device code only).
- `build.sh [ARCH]`: `cargo oxide build --arch ARCH` (default `sm_121`), then copies the PTX to
  `artifacts/gdn_simt.<ARCH>.ptx` and writes `artifacts/gdn_simt.<ARCH>.manifest` (hash of the kernel sources,
  hash of the PTX, toolchain, cuda-oxide backend revision) and prints the ptxas register/spill report.
- `build.sh check`: builds and runs `mma_check/`, the numeric check of the tf32/bf16 `mma.sync` fragments
  (run it under `scripts/gpu.sh`).
- `artifacts/`: checked in. sysone's `build.rs` embeds an artifact only if its manifest matches the current sources
  and the PTX; otherwise selecting the SIMT backend fails at run time with the rebuild command.

Needs: the toolchain in `rust-toolchain.toml` (1.99.1 with rust-src, rustc-dev, llvm-tools), `cargo-oxide`
(`cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide`, then `cargo oxide setup` in the
cuda-rust checkout), and the CUDA toolkit. sysone's own build (1.98.0) needs none of these.

After editing `src/`, run `kernels-simt/build.sh` and commit the refreshed artifact with the source change.
