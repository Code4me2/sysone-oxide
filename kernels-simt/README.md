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
  and the PTX; otherwise selecting the SIMT backend fails at run time with the rebuild command. At run time
  `src/kernels/simt.rs` picks the artifact built for the device's exact architecture (`sm_<major><minor>`, with or
  without the `a` suffix); there is no fallback to another architecture's PTX.
  - `gdn_simt.sm_121`: GB10, validated (docs/G1.md).
  - `gdn_simt.sm_120`: RTX 5090 / RTX PRO 6000 (x86). Built on GB10 from the same sources; the PTX differs from
    sm_121's only in `.target sm_120` and `.version 8.7` (8.8 for sm_121). ptxas for sm_120 reports
    the same registers, spills and shared memory as sm_121: scan 255 registers / 296 B spills / 47,120 B smem, tinv
    56 / 0 / 37,888 B). **Never run yet**: its first run is docs/x86/TASKS.md T2.

After editing `src/`, rebuild **every** artifact (`build.sh sm_121`, `build.sh sm_120`) and commit them with the
source change: an artifact whose source hash no longer matches is disabled by `build.rs`.

Needs: the toolchain in `rust-toolchain.toml` (1.99.1 with rust-src, rustc-dev, llvm-tools), `cargo-oxide`
(`cargo +1.99.1 install --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide`, then `cargo oxide setup` in the
cuda-rust checkout), and the CUDA toolkit. sysone's own build (1.98.0) needs none of these.

After editing `src/`, run `kernels-simt/build.sh` and commit the refreshed artifact with the source change.
