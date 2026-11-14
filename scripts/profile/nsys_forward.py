#!/usr/bin/env python3
"""Summarise an Nsight Systems trace of `bench_forward` (docs/PROFILE.md).

Input: the SQLite export of a `nsys profile --trace=cuda-sw,osrt` run of
`bench_forward --tokens A,B,... --iters N --eager-iters E --no-profile`. Run it with `python3 -I`.

Why software trace: on GB10 (driver R580, nsys 2026.3) the hardware CUDA trace that `--cuda-graph-trace=node`
needs drops almost every kernel record, so per-node timing inside a replayed CUDA graph is not available. The
software trace records (a) every kernel of the eager forwards and (b) each graph replay as one GPU interval
(CUPTI_ACTIVITY_KIND_GRAPH_TRACE). Kernel durations do not depend on the launch mode, so

    replay bubbles (launch gaps + tails inside the graph) = replay GPU span - sum of kernel durations of one forward

with the kernel sum taken from the eager forwards of the same bucket (the last `--eager-iters` forwards before
the replays; the first eager step of a bucket is the JIT one and is skipped).

Prints, per bucket: per-kernel-group table (name + grid, launches per forward, ms per forward), the kernel sum, the
eager span and its idle gaps, the replay span and its implied bubbles, and the step boundary between replays
(GPU idle between two replays, with the copies that happen in it).
"""
import argparse
import sqlite3
import statistics
from collections import OrderedDict, defaultdict


def load(db):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    kernels = [
        (s, e, names[n], (gx, gy, gz), (bx, by, bz), regs, smem)
        for s, e, n, gx, gy, gz, bx, by, bz, regs, smem in c.execute(
            "select start, end, shortName, gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread, "
            "staticSharedMemory + dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL order by start")
    ]
    graphs = list(c.execute("select start, end, graphExecId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start"))
    copies = list(c.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start"))
    return kernels, graphs, copies


def forwards(kernels):
    """Split the kernel stream into eager forwards, each starting at the embedding kernel."""
    out, cur = [], None
    for k in kernels:
        if k[2].startswith("embedding_batch"):
            cur = [k]
            out.append(cur)
        elif cur is not None:
            cur.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--eager-iters", type=int, default=2, help="timed eager forwards per bucket (after the JIT one)")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    kernels, graphs, copies = load(a.sqlite)
    fw = forwards(kernels)
    # Graph replays grouped by executable graph (one per bucket), in bucket order.
    by_exec = OrderedDict()
    for s, e, gx in graphs:
        by_exec.setdefault(gx, []).append((s, e))
    for gx, reps in by_exec.items():
        first = reps[0][0]
        # Eager forwards of this bucket: the last `eager_iters` that ended before the first replay.
        eager = [f for f in fw if f[-1][1] <= first][-a.eager_iters:]
        if not eager:
            continue
        ksum = [sum(k[1] - k[0] for k in f) / 1e6 for f in eager]
        span = [(f[-1][1] - f[0][0]) / 1e6 for f in eager]
        n_launch = len(eager[0])
        groups = defaultdict(lambda: [0.0, 0, None])
        for f in eager:
            for k in f:
                key = (k[2], k[3])
                g = groups[key]
                g[0] += (k[1] - k[0]) / 1e6 / len(eager)
                g[1] += 1
                g[2] = (k[4], k[5], k[6])
        gaps = []
        for f in eager:
            gaps.append([(f[i + 1][0] - f[i][1]) / 1e3 for i in range(len(f) - 1)])
        rspan = [(e - s) / 1e6 for s, e in reps]
        bounds = [(reps[i + 1][0] - reps[i][1]) / 1e6 for i in range(len(reps) - 1)]
        ks = statistics.median(ksum)
        print(f"== graph exec {gx}: {len(reps)} replays, {n_launch} kernel launches per forward")
        print(f"kernel sum per forward (eager, median of {len(eager)}): {ks:.1f} ms")
        print(f"eager span {statistics.median(span):.1f} ms; idle inside eager forward "
              f"{statistics.median(span) - ks:.1f} ms; median gap {statistics.median(gaps[-1]):.1f} us")
        rs = statistics.median(rspan)
        print(f"replay GPU span median {rs:.1f} ms (min {min(rspan):.1f}, max {max(rspan):.1f}); "
              f"implied bubbles inside replay {rs - ks:.1f} ms ({(rs - ks) / rs * 100:.2f}%), "
              f"{(rs - ks) / max(n_launch - 1, 1) * 1e3:.1f} us per kernel boundary")
        if bounds:
            b = statistics.median(bounds)
            between = [c for c in copies if reps[0][0] <= c[0] <= reps[-1][1]]
            h2d = [c for c in between if c[3] == 1]
            d2h = [c for c in between if c[3] == 2]
            print(f"step boundary (GPU idle between replays) median {b:.2f} ms = {b / (rs + b) * 100:.2f}% of a step; "
                  f"per step: {len(h2d) / len(reps):.0f} HtoD copies "
                  f"{sum(c[2] for c in h2d) / len(reps) / 1e3:.1f} kB {sum(c[1] - c[0] for c in h2d) / len(reps) / 1e3:.1f} us, "
                  f"{len(d2h) / len(reps):.0f} DtoH {sum(c[1] - c[0] for c in d2h) / len(reps) / 1e3:.1f} us")
        print(f"{'kernel':<44} {'grid':>16} {'block':>14} {'regs':>4} {'smem':>6} {'n':>4} {'ms':>9} {'%':>5}")
        rows = sorted(groups.items(), key=lambda kv: -kv[1][0])
        for (name, grid), (ms, n, (blk, regs, smem)) in rows[: a.top]:
            print(f"{name[:44]:<44} {str(grid):>16} {str(blk):>14} {regs:>4} {smem:>6} "
                  f"{n // len(eager):>4} {ms:>9.2f} {ms / ks * 100:>5.1f}")
        print()


if __name__ == "__main__":
    main()
