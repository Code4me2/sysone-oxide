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

## Follow-up: sysone bf16 on the E1 eval gate (A2, ws/a2-eval-sysone)

The full 278-record set through the real sysone server (`--engine bf16`, default buckets 256…65,536 with CUDA graphs,
one request at a time over HTTP; `reference/eval/run_sysone.sh`) passes the committed `[eval.bf16]` gate against all
four bf16-tier paths: 1/834 top-1 change above the 0.02 reference margin (limit 2/834; `ledgar-04056`, a labeled gain),
mean |dp| 1.36-1.54e-4 (limit 2.20e-4), p99 |dp| 3.48e-3-4.13e-3 (limit 5.82e-3), accuracy 228/278 vs 227/278 (paired CI
lower bound 0, limit -1.08pp), Brier CI upper bound <= 6.0e-4 (limit 2.36e-3). 181 s of request time for 550k rendered
tokens (3,034 tok/s at concurrency 1; the HF eager reference took 492 s). Details: reference/eval/README.md.

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

## Follow-up: the deep-layer asymmetry (A3, ws/a3-deep-layer)

**Finding: no numeric deviation from the reference. The layer31/norm asymmetry is one unlucky bf16 realization,
concentrated in one request; it is not systematic.** Nothing in the forward was changed. Evidence:

1. *Where.* The per-request breakdown (now printed by `hidden_states_vs_fp32_truth`) puts most of the excess in one
   request: rows > 1.5x worse / better at layer31 m040 (4k) 166/19, m044 (16k) 67/43, m026 22/7, m019 16/5, all
   others balanced or better (m037 25/36); at norm m040 132/5 of 222/75. Dumping every layer (`hidden_dump_all_layers`,
   `reference/deep_layer.py dump`) shows layers 0-15 balanced (median ln ratio -0.004..+0.002), and in m040 the
   excess sits in prefix positions 128-1024 (224 sampled rows), appearing at layer 18 (mean ln ratio +0.11) and
   growing to +0.41 at layer 31; positions 0-128 and 1024-4141 reach only +0.12 and +0.05. The bf16 null realizations show the
   same region as sensitive (perm1 vs served on m040: 117/43 at layer31): bf16_served happens to be a lucky
   realization there.
2. *Per layer, same input (teacher forcing).* The reference model run with its residual stream replaced at layer L
   by sysone's own stream after layer L-1 (`deep_layer.py forced`, bf16 and fp32, every layer, all prefix rows of
   m019 and m037, every second row of m040), compared op group by op group with sysone's layer L on the same input:
   sysone/reference local error ratios are 0.986-1.005 for every layer, mixer, post-mixer residual, MLP and layer
   output (mean ln ratio of the layer delta -0.0001 on GDN layers, -0.002..+0.0003 on attention layers; on m040's
   rows 128-1024 alone +0.0005). The GDN mixer output is bit-identical to the reference's on 52-95% of elements and
   sysone's is slightly closer to fp32 (ratio 0.986-1.000); the attention mixer is 0.995-0.998 (slightly closer).
   No row-coherent bias between sysone and the reference (coherence of sysone - reference ~1 for attention; for the
   GDN mixer 1-7, matched by the reference's own coherence vs fp32, as expected from the shared recurrent state).
3. *Other sysone realizations.* Packing a request with others leaves its rows bit-identical (no new realization).
   Running sysone in three permuted hidden bases (`EngineOptions::debug_permute_hidden`, the same construction as the
   reference's bf16_perm runs) gives three independent sysone realizations; all are on the good side of the
   reference: layer31 median ln ratio -0.013 / -0.014 / -0.010, rows worse/better 176/190, 173/206, 176/185; norm
   -0.012 / -0.011 / -0.010, 90/97, 73/90, 75/81 (all six taps pass every criterion with large margins). Over all 20
   sysone x bf16 realization pairs the layer31 median ln ratio averages -0.004 (range -0.025..+0.032); bf16 x bf16
   pairs span +-0.016 and sysone x sysone pairs +-0.040, so the original run's +0.020 is inside sysone's own
   realization spread. On m040 the permuted sysone runs have geo-mean errors 7.82 / 7.53 / 7.66e-3 vs the bf16
   realizations' 7.63-8.23e-3 (the original basis: 8.72e-3).

Consequence for later precision work (NVFP4, faster GDN): a single run's deep-layer paired statistics carry a
realization component of about +-0.02-0.04 in median ln ratio and a few hundred rows in the sign counts, mostly from
one or two long requests; a precision change should be judged on several realizations (hidden-basis permutations
of the candidate are a cheap way to get them) or with the design effect, not on one run's counts. The per-layer
teacher-forced comparison (`reference/deep_layer.py forced` + `reference/a3/forced_stats.py`) is the sharper tool
for checking that a kernel change keeps the reference's numerics.

## Forward speed (bench_forward, merged main, GPU lock held)

| bucket | replay ms | tok/s | GEMM | GDN | norm + pointwise | attention |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 1,041 | 3,480 packed (3,935 bucket) | 61.3% | 29.5% | 7.8% | 1.3% |
| 16,384 | 3,732 | 4,389 | 61.9% | 27.8% | 8.9% | 1.3% |

Before P1 the 16k forward was 4,393 ms (3,730 tok/s). GDN at 16k is now 1,042 ms (scan_prefix 409, scan_suffix 308,
chunk_tinv 165). GEMMs run at 93-101 TFLOP/s. JIT warm-up is still ~24 s on first step.

## Open items

1. Gate (a): redefined and passing (above); the deep-layer worse/better asymmetry is a realization effect, not a deviation (A3).
2. GDN: G1 (cuda-oxide SIMT, tensor-core tf32) in progress; target ~0.25 s at 16k.
3. Fuse silu_mul and norms (0.34 s at 16k); pin cuBLASLt algorithms per bucket (would also settle the batch-invariance attribution).
4. JIT disk cache (cuTile has one; enable it).
5. NVFP4 end to end with the E1 gate; allowed_token_mass; engine_busy_frac > 1.
6. Review nits: GdnPlan now carries mutable per-batch data (qk_inv) - not shareable across concurrent streams; stale pad rows in
   bucket arenas; split_batch assumes push_request batches; score.py subset verdict needs a distinct exit code; more
   interleaved graph-replay cases.
7. vLLM / reference throughput baselines (needs docker group).
