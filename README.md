# sysone

Inference server for System One decision models (`baselabs/elephant-1`) on
[cuda-rust](../cuda-rust): a drop-in, higher-throughput replacement for OneJev's `qev` server.

- Design: [docs/DESIGN.md](docs/DESIGN.md)
- Phase 0 report (environment, GB10 measurements): [docs/PHASE0.md](docs/PHASE0.md)
- Platforms (GB10 vs discrete x86) and prerequisites: [docs/PLATFORM.md](docs/PLATFORM.md)
- GDN SIMT kernels (cuda-oxide): [docs/G1.md](docs/G1.md); `kernels-simt/build.sh` rebuilds their checked-in PTX
  (needs Rust 1.99.1 + cargo-oxide; the main build does not).
- Forked from grout; its Qwen3 engine is kept read-only under [legacy/grout](legacy/grout) as porting reference.

Requires `../cuda-rust` checked out next to this repo. Run `scripts/doctor.sh` to check every prerequisite (CUDA ≥ 13.3 with
tileiras, R580+ driver, clang/libclang headers, Docker, HF auth); it prints the fix for anything missing.

```bash
cargo test --lib
cargo run --release --bin hwprobe          # GB10 bandwidth + bf16 GEMM roofline
cargo run --release -- --model <dir>       # loads and checks config.json (serving not implemented yet)
```
