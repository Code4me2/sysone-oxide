# Documentation

sysone serves text decisions (`noul`, `choice`, `score`), not chat or multimodal generation.
For a first run, use the current user guides below rather than the historical phase reports.

## User guides

| Document | Purpose |
|---|---|
| [Getting started](GETTING_STARTED.md) | Clone/layout, pinned dependencies, public model, launch, HTTP API, limits and troubleshooting |
| [Performance](PERFORMANCE.md) | Completed optimized-qev serving comparison, absolute rates, correctness and caveats |
| [Bulk tagging](BULK.md) | JSONL jobs, shared question sets, multi-GPU workers and resume; its GB10 measurements are historical |
| [Platforms](PLATFORM.md) | Build prerequisites, memory/GPU differences and tested scope |
| [Exact eager modes](x86/README.md) | BF16/Tile opt-in configuration, bounded warmup and qualification limits |
| [Project landing page](../README.md) | Overview and a minimal request |

## Developer references

- [Architecture and code guide](ARCHITECTURE.md): request lifecycle, packed trees, module map, precision and
  [testing layers](ARCHITECTURE.md#6-correctness-and-testing). Its phase-status sections and old measurements
  should be read with their follow-ups; use the current guides for launch instructions and qualification scope.
- [Design](DESIGN.md): rationale, reference semantics, correctness strategy and original development plan.
- [Reference harness](../reference/README.md) and [evaluation tools](../reference/eval/README.md): developer-only
  golden generation and evaluation, with extra Python dependencies, fixtures and machine-specific defaults.
  They are **not prerequisites for serving the public BF16 model**.
- [x86 results index](../benchmarks/results/x86/README.md): standard runner layout and the
  [curated accepted-comparison summary](../benchmarks/results/x86/qev-serving-optimized/summary.json).
- Source: [`src/`](../src), [`tests/`](../tests), [`benchmarks/s1/`](../benchmarks/s1), and
  [`scripts/`](../scripts). Per-module Rust documentation can be generated with `cargo doc --no-deps`.

### Historical implementation and measurement reports

These files are preserved in place. Their numbers, open items and defaults describe their stated revision and
hardware, not an automatically current support matrix or a new optimized-qev comparison.

| Reports | Subject |
|---|---|
| [Phase 0](PHASE0.md), [Phase 1](PHASE1.md), [Phase 2a](PHASE2A.md) | Initial probes, frontend/server implementation, engine gates and follow-up analyses |
| [G1](G1.md), [E2](E2.md) | SIMT GatedDeltaNet kernels and engine integration on GB10 |
| [N1](N1.md) | NVFP4 engine and checkpoint profiles; not the public BF16 qev comparison |
| [L1](L1.md) | Pinned GEMMs, batch invariance and last-layer trimming |
| [Profiling](PROFILE.md), [F1](F1.md) | GB10 profiling, fused NVFP4 gate/up and kernel tuning |
| [C1](C1.md), [S2](S2.md) | NVFP4 cuBLASLt down projection, gated norm and residual epilogue |

### Historical operational collaboration material

The original [x86 agent operating manual](x86/AGENT.md) and [task list](x86/TASKS.md) record host-specific agent
collaboration and safety procedures. They remain unchanged and contain dated status statements; they are not a
new-user setup prerequisite. Their GPU-lock, data-hygiene and no-tolerance-changing safeguards still matter for
work in that environment. Current exact-mode and serving evidence is summarized in
[the x86 guide](x86/README.md) and [Performance](PERFORMANCE.md).
