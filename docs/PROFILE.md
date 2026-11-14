# Profile of the forward on GB10 (F1, 2026-11-14)

Where one packed forward of elephant-1 spends its time on the GB10 (sm_121, 48 SMs, unified LPDDR5X), for the
nvfp4 engine (`--engine nvfp4`, the NVFP4-MLP checkpoint, fused quantization, SIMT GDN) and the bf16 engine for
comparison, at the 4k and 16k buckets of `bench_forward` (the same synthetic requests as N1: 3 questions, ~600
tokens each; 16,377 packed tokens in the 16k bucket). Measured on an idle GPU under `scripts/gpu.sh`, main at
`6183295`, before any F1 change. This file is the measurement; [F1.md](F1.md) has what was changed because of it.

Summary: **the CUDA-graph replay is kernel-bound end to end.** Launch gaps, tails and the step boundary together
are below 0.1% of a 16k forward, so programmatic dependent launch (PDL) cannot reach the 2% bar and was not
implemented. The time is in the kernels: the FP4 GEMMs (29%, limited by L2 throughput, a quarter of which is
wasted on the e4m3 block-scale loads), the bf16 cuBLAS projections (35%), the fused SiLU·mul + quantize (10%,
latency-bound at one tile block per SM), and the GDN scan.

## Method

### (a) Timeline: Nsight Systems

```bash
# nvfp4 (and bf16 with --model .../elephant-1); 2 timed eager forwards + 5 replays per bucket, no event profile.
scripts/gpu.sh /usr/local/bin/nsys profile --trace=cuda-sw,osrt --sample=none --cpuctxsw=none \
    -o $OUT/sw_nvfp4 --force-overwrite true \
    ./target/release/bench_forward --model $HF/elephant-1-nvfp4-mlp --tokens 4096,16384 --iters 5 --eager-iters 2 --no-profile
/usr/local/bin/nsys export --type sqlite -o $OUT/sw_nvfp4.sqlite $OUT/sw_nvfp4.nsys-rep
python3 -I scripts/profile/nsys_forward.py $OUT/sw_nvfp4.sqlite --eager-iters 2
```

**Software trace, not node-level graph trace.** On this GB10 (driver 580.178, nsys 2026.3.2) the hardware CUDA
trace that `--cuda-graph-trace=node` requires drops almost every kernel record ("incomplete CUPTI events dropped";
a `--trace=cuda` run keeps 14 of ~1,700 kernels, and none with `--cuda-event-trace=false`), so the kernels inside
a replayed graph cannot be timed one by one here. The software trace records every kernel of the *eager* forwards
and each graph replay as one GPU interval. Kernel durations do not depend on how a kernel was launched, so
`scripts/profile/nsys_forward.py` derives:

- **kernel time** per forward: the sum of kernel durations of an eager forward (median of the two timed ones);
- **bubbles inside a replay** (launch gaps + tails between graph nodes): replay GPU span − kernel time;
- **the step boundary**: GPU idle time between two replays, and the copies that run in it (metadata upload,
  logits readback). bench_forward's step is `stage` (pack + upload metadata, rewrite GDN/attention plans) →
  graph launch → logits readback → `stream.synchronize()`, the same as the server's and bulk's `GpuEngine::step`.

On a discrete x86 GPU the hardware trace may work: try `--trace=cuda --cuda-graph-trace=node` first there (then
`nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace` gives the in-graph gaps directly).

### (b) Kernels: Nsight Compute

```bash
# Sections for the top kernels (first matching launches of the eager 16k forward: layers 0-3 cover every kernel).
scripts/gpu.sh /usr/local/cuda/bin/ncu -k 'regex:gemm_nvfp4_raw|nvjet|Kernel2|silu_mul|gdn_simt_scan|add_zc_rmsnorm|seg_attn|tree_conv|gated_rmsnorm|gdn_simt_tinv|qk_norm' \
    --launch-count 40 --section SpeedOfLight --section Occupancy --section LaunchStats --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats -o $OUT/ncu_nvfp4_16k -f \
    ./target/release/bench_forward --model $HF/elephant-1-nvfp4-mlp --tokens 16384 --iters 1 --eager-iters 1 --no-profile
/usr/local/cuda/bin/ncu -i $OUT/ncu_nvfp4_16k.ncu-repz --page details --csv > $OUT/details.csv
python3 -I scripts/profile/ncu_summary.py $OUT/details.csv
# Raw counters (L2 bytes, tensor pipe, stall reasons) for the MLP kernels:
scripts/gpu.sh /usr/local/cuda/bin/ncu -k 'regex:gemm_nvfp4_raw|silu_mul_nvfp4|nvjet|add_zc_rmsnorm' --launch-count 6 \
    --metrics gpu__time_duration.sum,lts__t_bytes.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct \
    --csv -f ./target/release/bench_forward --model $HF/elephant-1-nvfp4-mlp --tokens 16384 --iters 1 --eager-iters 1 --no-profile
```

GPU performance counters are open to normal users on this machine (`NVreg_RestrictProfilingToAdminUsers=0`,
`/proc/driver/nvidia/params: RmProfilingAdminOnly: 0`). GB10 reports no `dram__*` metrics (unified memory): the
memory side below is L2 (`lts__*`) plus bytes over time against the 243 GB/s that hwprobe measures for LPDDR5X.
Nsight Compute ran with its default clock control; the SM clock it reports (2.32-2.41 GHz) is the same range the
GPU sustains under load in bench_forward (2.29-2.42 GHz logged by `nvidia-smi`; GB10 is power-capped, "SW Power
Cap" active, so the 3.0 GHz maximum is never reached in a long forward).

Reports, traces and SQLite exports stay outside git (`/home/velvet/Desktop/sysone-prof/` here); only these
summaries are committed.

## (a) Timeline

| | nvfp4 4k | nvfp4 16k | bf16 4k | bf16 16k |
|---|---:|---:|---:|---:|
| kernel launches per forward | 420 | 420 | 420 | 420 |
| kernel time per forward (eager, ms) | 578.9 | 2,157.8 | 929.1 | 3,166.5 |
| replay GPU span (ms, median of 5) | 572.4 | 2,154.0 | 922.6 | 3,163.5 |
| implied bubbles inside a replay (ms) | < 0 (noise) | < 0 (noise) | < 0 (noise) | < 0 (noise) |
| median gap between eager kernels (us) | 2.2 | 2.4 | 2.0 | 2.0 |
| idle inside one eager 16k forward (ms) | - | 1.0 | - | 0.8 |
| step boundary: GPU idle between replays (ms) | 0.23 | 0.39 | 0.29 | 0.39 |
| ... copies in it: 9 HtoD (kB, us) + 1 DtoH (us) | 165 kB, 14 us + 4 us | 307 kB, 17 us + 5 us | 165 kB, 15 us + 4 us | 307 kB, 17 us + 5 us |

- The replay span is *shorter* than the eager kernel sum by 0.1-1.1% (run-to-run clock variation under the power
  cap is of that size), i.e. the gaps between graph nodes are not measurable against it. The eager forward, with
  a host launch per kernel, has 1.0 ms of idle at 16k (0.05%); a replay has at most that.
- The step boundary (metadata upload, plan rewrite, logits readback, host sync) costs 0.39 ms per 16k step
  (0.02%) and 0.23 ms at 4k (0.04%).

Per kernel group, nvfp4, 16k (eager kernel durations, ms per forward; TFLOP/s from 2·T·N·K at T = 16,384):

| kernel | grid | launches | ms | % | what | rate |
|---|---|---:|---:|---:|---|---|
| `gemm_nvfp4_raw` | 24576 | 32 | 420.6 | 19.5 | FP4 MLP gate/up `[24576, 4096]` | 251 TFLOP/s |
| `nvjet_sm121_tst_mma_128x208x64` | 632x13 | 24 | 410.5 | 19.0 | bf16 GDN in-projection `[12352, 4096]` (cuBLAS) | 97 TFLOP/s |
| cutlass `Kernel2` (sm80 128x256) | 512x4 | 32 | 215.3 | 10.0 | bf16 GDN out_proj + attention o_proj `[4096, 4096]` | 82 TFLOP/s |
| `gemm_nvfp4_raw` | 4096 | 32 | 210.7 | 9.8 | FP4 MLP down `[4096, 12288]` | 250 TFLOP/s |
| `silu_mul_nvfp4` | 16384x12 | 32 | 208.4 | 9.7 | SiLU·mul + quantize | 141 GB/s |
| `gdn_simt_scan` | 64x2049 | 48 | 188.8 | 8.8 | GDN prefix + suffix scans (out of F1's scope) | |
| cutlass `Kernel2` | 512x8 | 8 | 106.0 | 4.9 | bf16 attention q_proj `[8192, 4096]` | 83 TFLOP/s |
| `add_zc_rmsnorm` | 16384 | 32 | 77.3 | 3.6 | residual add + input norm (before the mixers) | 223 GB/s |
| `add_zc_rmsnorm_nvfp4` | 16384 | 32 | 75.6 | 3.5 | residual add + post-attention norm + quantize | 190 GB/s |
| `tree_conv1d_silu` | 2305x16 | 24 | 59.0 | 2.7 | GDN conv | |
| `gated_rmsnorm` | 1024x32 | 24 | 58.2 | 2.7 | GDN gated norm | 166 GB/s |
| `gdn_simt_tinv` | 32x2305 | 24 | 44.4 | 2.1 | GDN chunk inverse | |
| `seg_attn` | 3073x4 | 8 | 36.1 | 1.7 | segment attention | |
| cutlass `Kernel2` | 512x2 | 8 | 28.2 | 1.3 | bf16 attention kv_proj `[2048, 4096]` | 78 TFLOP/s |
| `qk_norm_rope` | 8192 | 8 | 15.6 | 0.7 | | |
| rest (embedding, gates, slot guard, gather, slot head, first norm) | | 76 | 3.1 | 0.1 | | |
| **total** | | **420** | **2,157.8** | | | |

By class: bf16 cuBLAS GEMMs 760 ms (35%), FP4 GEMMs 631 ms (29%), GDN 350 ms (16%), SiLU·mul+quantize 208 ms
(10%), norms 153 ms (7%), attention 52 ms (2%). The bf16 engine at 16k (3,166.5 ms) spends 1,064 ms in the gate/up
and 603 ms in the down cuBLAS GEMMs (`nvjet` 632x24 and 632x4), 170 ms in `silu_mul` and 157 ms in its 64
`add_zc_rmsnorm`; the rest is the same kernels as nvfp4. The 4k tables (same kernels, same order of cost) are in
the script's output.

## (b) Kernels (Nsight Compute, nvfp4 16k)

| kernel | us | SM % | L2 % | tensor pipe active | occupancy (achieved / theoretical) | regs | limiter |
|---|---:|---:|---:|---:|---|---:|---|
| `gemm_nvfp4_raw` gate/up | 12,900 | 55 | **83** | 56% | 19 / 25% (1 block of 12 warps) | 168 | **L2 throughput**: 19.0 GB through L2 per launch, of which ~4.7 GB are e4m3 scale sectors (below); stalls: wait 21%, math-pipe throttle 18%, long scoreboard 17% |
| `gemm_nvfp4_raw` down | 6,430-7,330 | 49-55 | 72-82 | 56% | 21-23 / 25% | 168 | L2 throughput, as gate/up |
| `nvjet` (cuBLAS bf16) in-proj | 16,540 | 86 | 59 | 82% | 16 / 17% | 255 | **tensor pipe** (math-pipe throttle 60%): at the bf16 MMA roofline |
| cutlass `Kernel2` (cuBLAS bf16) out_proj / o_proj | 6,330-6,640 | 71-74 | 46-48 | - | 14-16 / 17% | 222 | tensor pipe, ~85% of nvjet's efficiency (an older sm80 kernel cuBLAS picks for N = 4096) |
| `silu_mul_nvfp4` | 7,030-7,130 | 13 | 10 | 0 | 13 / 17% (1 block of 8 warps) | 255 | **latency**: 920 MB per launch at 130 GB/s (53% of DRAM bandwidth); register-limited to one tile block per SM, issue 13%, IPC 0.6 |
| `silu_mul` (bf16 engine) | 5,435 | 9 | 13 | 0 | 77% | 46 | DRAM bandwidth (1.21 GB at 222 GB/s) |
| `add_zc_rmsnorm_nvfp4` | 2,190-2,260 | 10 | 12 | 0 | 11 / 17% | 255 | ~DRAM bandwidth (576 MB at 260 GB/s through L2; 440 MB from DRAM) |
| `add_zc_rmsnorm` | 2,390-2,460 | 4 | 13 | 0 | 11 / 17% | 255 | DRAM bandwidth (671 MB through L2 at 280 GB/s) |
| `gdn_simt_scan` (prefix / suffix) | 5,450 / 3,070 | 23 / 29 | 16 / 19 | - | 20 / 17% | 255 | latency (serial chunk scan), out of scope |
| `gdn_simt_tinv` | 1,920 | 55 | 6 | - | 16 / 17% | 56 | shared-memory bound (L1 57%) |
| `tree_conv1d_silu` | 2,510 | 17 | 12 | - | 33% | 110 | latency |
| `gated_rmsnorm` | 2,500 | 6 | 9 | - | 13 / 17% | 255 | latency, 166 GB/s |
| `seg_attn` | 4,470 | 48 | 24 | - | 21 / 25% | 168 | compute (fp32 softmax, split-P MMA) |
| `qk_norm_rope` | 1,660 | 38 | 15 | - | 16 / 25% | 168 | issue-bound |

**The FP4 GEMM is L2-bound, and a quarter of its L2 traffic is scale sectors.** One 128x128 output tile with
K = 4096 reads 256 KiB of packed activations, 256 KiB of packed weights and 2 x 32 KiB of e4m3 block scales.
Over 24,576 tiles that is 14.2 GB. ncu measures 19.0 GB (564.7 M read sectors) per gate/up launch. The
difference is the scales: a tile's scales for one k-step are `[128 rows, 8 bytes]` at a row stride of K/16
bytes, so every 8-byte row segment costs a whole 32-byte L2 sector (4x): 2 x 128 KiB per tile instead of
2 x 32 KiB, +4.7 GB per launch, exactly the measured excess. L2 throughput is at 83% of peak while the tensor
pipe is active 56% of the time; the down GEMM shows the same split. W3c's tile sweep (and F1's re-run at 1k-32k,
`bench_nvfp4 --sweep`) finds 128x128x128 with grouped-M 8 the fastest cuTile configuration at every M: larger
tiles lose more to registers and staging than they save in traffic.

**`silu_mul_nvfp4` is slower than plain `silu_mul` because it runs one tile block per SM.** Its `[1, 1024]` tile
compiles to 255 registers x 256 threads, the whole register file, so 8 warps per SM wait on two IEEE divisions,
an exponential and a group reduction per element with nothing to hide the latency; plain `silu_mul` runs at 46
registers and 77% occupancy.

## (c) PDL (programmatic dependent launch)

What PDL could recover is the time between the end of one kernel's last CTAs and the start of the next kernel's
first useful work: the launch gap plus the ramp-down tail of the previous kernel.

- **Gaps:** not measurable inside a replay (span ≤ kernel sum); an eager forward with 419 host launches between
  its kernels idles 1.0 ms at 16k (2.4 us median per boundary). A replay's gaps are at most that: ≤ 0.05% of the
  forward.
- **Tails:** every large kernel runs many waves at 16k (gate/up GEMM 24,576 tiles on 48 SMs at one tile per SM =
  512 waves; norms 16,384 blocks; scans 64x2049), so the partially occupied last wave is ≤ 1/20 of a kernel and
  overlapping it with the next kernel's prologue saves a few microseconds per boundary: ~1-2 ms per forward
  (< 0.1%). At 4k the down GEMM has 21 waves (≤ 1.6% of that kernel's 6.6 ms in the tail).
- **Total recoverable: ≤ 3 ms of 2,158 ms (≤ 0.15%) at 16k, ≤ 0.3% at 4k**, below the 2% bar. **PDL is not
  implemented.**
- Which kernels could take it: the cuda-oxide SIMT kernels (`gdn_simt_*`: cuda-oxide has
  `griddepcontrol.wait/launch_dependents`; sysone's own launcher would set the launch attribute); the cuTile
  kernels (cuda-rust at `CUDA_RUST_REV` has `gdc_wait_tko` / `gdc_launch_dependents_tko` and a host-side
  `.programmatic_dependent_launch()`, Tile IR ≥ 13.4, sm_90+ including sm_120/121: `cutile-examples/examples/pdl.rs`;
  PLATFORM.md P5's "PDL is not listed for sm_12x" predates it); not the cuBLAS bf16 GEMMs (35% of the forward:
  no API makes a cuBLAS kernel wait on `griddepcontrol` or trigger its dependents), so every cuBLAS boundary stays
  a full serialization point. Stream capture keeps programmatic edges in the graph (CUDA ≥ 12.3), so a replay
  would retain them. With < 0.2% to gain, nothing is worth changing.

## (d) Opportunities, ranked (estimated ms saved per 16k nvfp4 forward)

| # | opportunity | evidence | estimate at 16k | bit-identical? | in F1 scope |
|---|---|---|---:|---|---|
| 1 | **SiLU·mul + quantize in the gate/up GEMM epilogue** (two accumulators per tile: gate and up rows of the same output columns; the `[T, 24576]` bf16 gate/up output is never written) | `silu_mul_nvfp4` 208 ms; gate/up writes and re-reads 805 MB per layer (51 GB per forward) | 150-230 | yes (same MMA chain per element, same epilogue ops) | yes |
| 2 | **Block-scale layout of the FP4 GEMM operands** (k-tile-major scales so a tile's `[128, 8]` scales are 1 KiB contiguous) | L2-bound at 83%, 25% of L2 bytes are wasted scale sectors, tensor pipe 56% | 100-150 | yes (same values, other layout) | yes |
| 3 | `silu_mul_nvfp4` tile shape / occupancy (only matters where #1 is not used) | one block per SM, 141 GB/s vs 243 GB/s | 60-90 | yes | yes |
| 4 | Residual add after the down GEMM in its epilogue | saves writing + re-reading the MLP output (4.3 GB per forward) | ~18 | yes for the FP4 down GEMM; not for cuBLAS (its `beta` epilogue rounds once, not twice) | yes, low value |
| 5 | A cuda-oxide `mma.sync` e2m1 GEMM (larger tiles, explicit cp.async/ldmatrix pipeline) where cuTile tops out | cuTile's best tile is 128x128 at every M; L2-bound; after #2 the remaining limiter decides | 50-200, after #2 | yes if the MMA chain per element is kept (same instruction, same K order) | yes, large effort |
| 6 | FP4 GEMM tile per M | 128x128x128 g8 is the best cuTile config at 1k, 4k, 16k and 32k (`bench_nvfp4 --sweep`) | 0 | (BM/BN/group-M do not change a bit: tested) | done: nothing to change |
| 7 | Norm + quantize in the gate/up GEMM prologue | the norm needs whole-row statistics; the prologue would re-normalise and re-quantize each row once per N tile (192x at gate/up) | negative | - | no |
| 8 | PDL / launch gaps / step boundary | (a), (c) | ≤ 3 | - | no (below 2%) |
| - | bf16 cuBLAS projections | 760 ms; nvjet at 82-88% tensor pipe, the sm80 `Kernel2` that cuBLAS picks for N = 4096 at ~71% | (pinning faster algorithms: workstream L1) | must stay cuBLAS | no (L1) |
| - | GDN scan / tinv / conv / gated norm | 350 ms | (separate work) | - | no |

The plan F1 follows from this: #1 and #2 first (NVFP4, bit-identical, the two largest), #3 for the unfused-SiLU
paths, then #4/#5 as the remaining profile allows.

## Addendum (after F1's experiments, same day)

Two of the readings above did not survive testing; F1.md has the numbers.

- **The FP4 GEMM is not L2-bound after all.** A k-tile-major scale layout (`[K/128, rows, 8]`, so a tile's scales
  for one k-step are one contiguous block) removed the scale-sector waste exactly as predicted (gate/up at 4k: L2
  bytes 4.74 → 3.85 GB per launch, L2 throughput 78% → 53%), and the kernel did not get faster (3.31 vs 3.24 ms;
  down 1.88 vs 1.74 ms; tensor pipe still 54-56%). The limiter is the kernel's own pipeline (one 12-warp tile
  block per SM, waits on its staged loads: stall "wait" 21%, long scoreboard 17%), not L2 bandwidth. Opportunity
  #2 is worth 0 ms in cuTile, and the layout change was not kept. cuTile compile hints on the plain GEMM (warps per
  CTA 4 or 8, occupancy 1 or 2) are no faster either (at best +2%, within noise); 16 warps is rejected. More FP4
  GEMM throughput needs a different kernel structure (#5).
- **#1 paid more than estimated once its epilogue was overlapped.** Fusing SiLU·mul + quantize into the gate/up
  GEMM only pays when the main loop stays the plain GEMM's (one accumulator over one 128-row weight tile, which
  takes interleaving the gate and up rows at load) and two tile blocks share an SM (cuTile `occupancy` 2), so one
  tile's epilogue runs under the other's MMAs: the fused launch then costs what the plain gate/up GEMM costs.
