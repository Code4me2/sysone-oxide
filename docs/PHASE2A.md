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

**Gate (a) redefined (A1, branch ws/a1-gate-a).** Dense dumps (`reference/hidden_dense.py`): 3,829 rows per tap
(every prefix row, every `P // 1024`-th for the 4k/16k requests, every answer row); threshold and statistic on the
same rows; per tap (i) p99 <= 1.5 x reference p99, (ii) median ln(sysone/ref) <= ln 1.5, (iii) one-sided binomial
sign test (alpha 0.01) on rows > 1.5x worse vs > 1.5x better. Validated on reference-only data before sysone ran
(`tests/golden/dev/model/hidden_gate_report.json`): five bf16 realizations (served, per-question, three
hidden-basis permutations) pass in every pairing, out of sample for the design effect, and leaving out any request;
every-row error x1.75/x2, 5% of rows x3+ (strided or one block) and 2x-median noise fail; so do x1.1-x1.6 and an
RMSNorm computed in Llama order or fully in bf16 (reported). One change from the proposal: the plain row binomial
failed equivalent bf16 realizations at layer31/norm (p 1e-3, 3.6e-4) because rows of a request move together, so
counts are deflated by a per-tap design effect (1.5 x the largest request-clustered dispersion among the bf16
realizations: 3.0, 3.1, 2.4, 3.9, 21.3, 18.4).

sysone against it (all taps PASS):

| tap | sysone p99 | limit (1.5 x ref p99) | max (ref) | median ln ratio | rows worse / better | sign p |
|---|---:|---:|---:|---:|---:|---:|
| layer0 | 7.52e-3 | 1.12e-2 | 1.21e-2 (1.21e-2) | -0.000 | 23 / 22 | 0.50 |
| layer3 | 8.99e-3 | 1.31e-2 | 1.44e-2 (1.22e-2) | -0.002 | 207 / 206 | 0.53 |
| layer4 | 8.19e-3 | 1.22e-2 | 1.47e-2 (1.40e-2) | -0.003 | 208 / 180 | 0.22 |
| layer15 | 1.84e-2 | 2.45e-2 | 4.73e-2 (4.78e-2) | +0.002 | 122 / 96 | 0.21 |
| layer31 | 2.47e-2 | 3.56e-2 | 5.25e-1 (6.31e-1) | +0.020 | 326 / 150 | 0.067 |
| norm | 3.65e-2 | 5.15e-2 | 5.15e-1 (7.52e-1) | +0.021 | 222 / 75 | 0.038 |

Caveat: at layer31 and norm sysone's worse/better asymmetry (row z = 8.1 and 8.5) is larger than any pair of bf16
reference realizations showed (max 4.1 and 3.2); it passes only because of the design effect, and the median shift
(+2.0%) is at the edge of the nulls (<= 1.6% on the full row set, up to 2.1% leaving one request out). Small, but
systematic: worth locating (which requests/rows) before the next precision change.

## Forward speed (bench_forward, merged main, GPU lock held)

| bucket | replay ms | tok/s | GEMM | GDN | norm + pointwise | attention |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,041 | 3,480 packed (3,935 bucket) | 61.3% | 29.5% | 7.8% | 1.3% |
| 16,384 | 3,732 | 4,389 | 61.9% | 27.8% | 8.9% | 1.3% |

Before P1 the 16k forward was 4,393 ms (3,730 tok/s). GDN at 16k is now 1,042 ms (scan_prefix 409, scan_suffix 308,
chunk_tinv 165). GEMMs run at 93-101 TFLOP/s. JIT warm-up is still ~24 s on first step.

## Open items

1. Gate (a): redefined and passing (above); locate sysone's deep-layer worse/better asymmetry.
2. GDN: G1 (cuda-oxide SIMT, tensor-core tf32) in progress; target ~0.25 s at 16k.
3. Fuse silu_mul and norms (0.34 s at 16k); pin cuBLASLt algorithms per bucket (would also settle the batch-invariance attribution).
4. JIT disk cache (cuTile has one; enable it).
5. NVFP4 end to end with the E1 gate; allowed_token_mass; engine_busy_frac > 1.
6. Review nits: GdnPlan now carries mutable per-batch data (qk_inv) - not shareable across concurrent streams; stale pad rows in
   bucket arenas; split_batch assumes push_request batches; score.py subset verdict needs a distinct exit code; more
   interleaved graph-replay cases.
7. vLLM / reference throughput baselines (needs docker group).
