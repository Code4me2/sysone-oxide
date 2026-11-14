#!/usr/bin/env python3
"""One line per profiled kernel from an Nsight Compute details CSV (docs/PROFILE.md).

    ncu -i REPORT.ncu-repz --page details --csv > details.csv
    python3 -I scripts/profile/ncu_summary.py details.csv [--names]

Columns: duration, SM clock, compute (SM) and memory throughput as % of peak, L2 (lts) throughput, achieved and
theoretical occupancy, registers, the occupancy limiter, issue slot use, the top warp stall reason, and ncu's own
bound classification where present. GB10 reports no DRAM metrics (unified memory): memory-side numbers are the
L2 / lts ones.
"""
import argparse
import csv
from collections import OrderedDict

COLS = [
    ("dur us", "Duration"),
    ("GHz", "SM Frequency"),
    ("SM %", "Compute (SM) Throughput"),
    ("Mem %", "Memory Throughput"),
    ("L1 %", "L1/TEX Cache Throughput"),
    ("L2 %", "L2 Cache Throughput"),
    ("occ ach", "Achieved Occupancy"),
    ("occ th", "Theoretical Occupancy"),
    ("regs", "Registers Per Thread"),
    ("blk/SM reg", "Block Limit Registers"),
    ("blk/SM smem", "Block Limit Shared Mem"),
    ("issue %", "Issue Slots Busy"),
    ("IPC", "Executed Ipc Active"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--names", action="store_true", help="list every metric name and exit")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    if a.names:
        for n in sorted(set((r["Section Name"], r["Metric Name"]) for r in rows)):
            print(n)
        return
    ks = OrderedDict()
    for r in rows:
        k = ks.setdefault(r["ID"], {"name": r["Kernel Name"], "grid": r["Grid Size"], "block": r["Block Size"],
                                    "rules": []})
        if r["Metric Name"]:
            k[r["Metric Name"]] = (r["Metric Value"], r["Metric Unit"])
        if r.get("Rule Name") and r.get("Estimated Speedup"):
            k["rules"].append((r["Rule Name"], r["Estimated Speedup"]))
    print("| id | kernel | grid | " + " | ".join(c[0] for c in COLS) + " |")
    print("|---|---|---|" + "---:|" * len(COLS))
    for i, k in ks.items():
        vals = []
        for _, m in COLS:
            v = k.get(m)
            if v is None:
                vals.append("-")
            else:
                val, unit = v
                if m == "Duration":
                    x = float(val.replace(",", ""))
                    x = x / 1e3 if unit == "ns" else (x * 1e3 if unit == "ms" else x)
                    vals.append(f"{x:.0f}")
                else:
                    vals.append(val)
        print(f"| {i} | {k['name'][:48]} | {k['grid']} | " + " | ".join(vals) + " |")


if __name__ == "__main__":
    main()
