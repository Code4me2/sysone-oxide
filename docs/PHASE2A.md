# Phase 2a report (2026-11-13)

Five workstreams (I1, P1, P2, E1, S1), each implemented on its own branch, reviewed independently, fixed where the
review blocked (E1), and merged into `main`. G1 (GDN on cuda-oxide) started after this and is not part of this report.

## Results

| Workstream | Result |
|---|---|
| S1 server fixes | Merged earlier (W5 review items: forward_s timing, empty request id, admission backpressure, ...). |
| I1 bf16 engine | Full GPU forward over packed trees with T-bucket CUDA graphs; replay bit-identical to eager. End-to-end gate (b) passes: vs bf16 reference dp p99 1.32e-2, max 2.27e-2, 0 flips; vs fp32 dp p99 1.24e-2 (the bf16 reference's own is 1.23e-2). HTTP replay of the 52-request corpus matches qev (5 error bodies byte-identical). Batch invariance max \|dp\| 1.20e-2 (tolerances.toml gate 0.01822). |
| P1 GDN speed (cuTile) | Fused scan with recomputed WY operands, blocked chunk inverse (fla solve_tril style), L2-friendly launch order, no stored f32 q/k. Mixer per layer: 2.28 ms at 1k (was 3.9), 8.30 at 4k (15.1), 32.6 at 16k (58.0); targets 5 / 20 ms missed with the precise default. The TF32 variant meets the time but is ~500x outside the fp32 tolerance and stays opt-in. |
| P2 attention | p_split=3 reaches ~38 useful TFLOP/s (target 50 missed; ~59 is the arithmetic ceiling of the 3-part P split). The engine uses one tile config at every T (bit-exact across buckets); `AttnTiles::for_tokens` is an opt-in documented as not bit-identical. Attention is 1.3% of the forward, so this matters little. |
| E1 eval set | 278 labeled records (128 BANKING77, 128 LEDGAR, 22 LongBench-v2; 834 questions) with 8 reference runs (incl. fp32). Paired-bootstrap gate in reference/eval/stats.py, limits in tests/tolerances.toml [eval.bf16]/[eval.nvfp4], validated by leave-one-out, cross-tier, tempering and minimal-flip probes. |

## Merged tree

`cargo test --release --all-targets` (via scripts/gpu.sh): everything passes except `engine_layers::hidden_states_vs_fp32_truth`
(gate (a)): layer31 p99 5.65e-2 > 3.66e-2 and norm p99 9.21e-2 > 6.67e-2 (layer15 now passes after P1). The I1 review
reproduced the failure and found it is heavy-tailed, row-specific sensitivity shared with the reference: over all 113 rows
of m019 the per-row sysone/reference error ratio has median 0.94-1.00 with symmetric tails, and on a fresh 63-row dump
(m008) sysone is worse than the reference on only 21-27 rows per tap. The gate is computed from 100 sampled rows, so its
p99 is the second-largest row. Proposed redefinition (not yet done; needs a decision): dumps over >= 1000 rows, threshold
and statistic on the same rows, plus a paired per-row criterion (median log ratio <= log 1.5, sign test on > 1.5x rows).

## Forward speed (bench_forward, merged main, GPU lock held)

| bucket | replay ms | tok/s | GEMM | GDN | norm + pointwise | attention |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,041 | 3,480 packed (3,935 bucket) | 61.3% | 29.5% | 7.8% | 1.3% |
| 16,384 | 3,732 | 4,389 | 61.9% | 27.8% | 8.9% | 1.3% |

Before P1 the 16k forward was 4,393 ms (3,730 tok/s). GDN at 16k is now 1,042 ms (scan_prefix 409, scan_suffix 308,
chunk_tinv 165). GEMMs run at 93-101 TFLOP/s. JIT warm-up is still ~24 s on first step.

## Open items

1. Gate (a) redefinition (above).
2. GDN: G1 (cuda-oxide SIMT, tensor-core tf32) in progress; target ~0.25 s at 16k.
3. Fuse silu_mul and norms (0.34 s at 16k); pin cuBLASLt algorithms per bucket (would also settle the batch-invariance attribution).
4. JIT disk cache (cuTile has one; enable it).
5. NVFP4 end to end with the E1 gate; allowed_token_mass; engine_busy_frac > 1.
6. Review nits: GdnPlan now carries mutable per-batch data (qk_inv) - not shareable across concurrent streams; stale pad rows in
   bucket arenas; split_batch assumes push_request batches; score.py subset verdict needs a distinct exit code; more
   interleaved graph-replay cases.
7. vLLM / reference throughput baselines (needs docker group).
