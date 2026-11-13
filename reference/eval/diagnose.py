"""Information only (not a gate): where a candidate run differs from the reference paths of a tier.

    python3 -I reference/eval/diagnose.py tests/golden/dev/eval/candidates/sysone_bf16.json --tier bf16 [--json out]

Prints, per reference path: the gate's comparison metrics per source (BANKING77, LEDGAR, LongBench-v2: questions,
top-1 changes, mean/p99/max |dp|, labeled gains/losses); every top-1 change with both runs' top-1 margins and
whether the reference margin is above the tier's disagree_margin (only those count for the gate); the records with
the largest |dp|; and throughput from the candidate's per-record seconds and the set's token counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalset  # noqa: E402
import stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--tier", choices=["nvfp4", "bf16"], required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cand = evalset.load_run(args.run)
    st = evalset.load_set()
    man = {m["id"]: m for m in st["manifest"]}
    lq = st["labeled_qid"]
    margin_lim = stats.load_limits(args.tier)["disagree_margin"]
    ca = stats.answered(cand)
    out = {"run": cand["name"], "tier": args.tier, "per_ref": {}}

    # throughput (information)
    rs = [r for r in cand["results"] if r["status"] == 200]
    secs = sum(r["seconds"] for r in rs)
    toks = sum(man[r["id"]]["total"] for r in rs)
    by_src = {}
    for r in rs:
        s = man[r["id"]]["source"]
        d = by_src.setdefault(s, {"records": 0, "seconds": 0.0, "tokens": 0})
        d["records"] += 1
        d["seconds"] += r["seconds"]
        d["tokens"] += man[r["id"]]["total"]
    out["throughput"] = {"records": len(rs), "request_seconds_sum": round(secs, 2), "tokens": toks,
                         "tokens_per_s": round(toks / secs, 1) if secs else None,
                         "first_record_seconds": rs[0]["seconds"] if rs else None,
                         "per_source": by_src, "provenance": cand.get("provenance", {})}
    print(f"== {cand['name']}: {len(rs)} records, sum of request seconds {secs:.1f} s, {toks} rendered tokens "
          f"(set.json `total`), {toks / secs:.0f} tok/s at concurrency 1")
    for s, d in by_src.items():
        print(f"   {s:14s} {d['records']:4d} records {d['seconds']:8.1f} s  {d['tokens']:8d} tok  "
              f"{d['tokens'] / d['seconds']:7.0f} tok/s")

    for name in stats.TIERS[args.tier]:
        ref = evalset.load_run(name)
        ra = stats.answered(ref)
        ids = [i for i in man if i in ca and i in ra]
        per_src, flips, recdp = {}, [], []
        for rid in ids:
            src = man[rid]["source"]
            d = per_src.setdefault(src, {"questions": 0, "flips": 0, "flips_counted": 0, "dp": [],
                                         "labeled_gains": 0, "labeled_losses": 0})
            worst = 0.0
            for qid, vc in ca[rid].items():
                vr = ra[rid][qid]
                dp = [abs(x - y) for x, y in zip(vc, vr)]
                d["dp"].extend(dp)
                worst = max(worst, max(dp))
                d["questions"] += 1
                ac, ar = stats.argmax(vc), stats.argmax(vr)
                if ac != ar:
                    mr, mc = stats.margin(vr), stats.margin(vc)
                    counted = margin_lim <= 0 or mr > margin_lim
                    d["flips"] += 1
                    d["flips_counted"] += counted
                    f = {"id": rid, "qid": qid, "ref_margin": round(mr, 6), "cand_margin": round(mc, 6),
                         "counted": counted, "ref_top": ar, "cand_top": ac,
                         "p_ref_top": [round(vr[ar], 6), round(vc[ar], 6)],
                         "p_cand_top": [round(vr[ac], 6), round(vc[ac], 6)]}
                    if qid == lq:
                        gold = st["manifest"][[m["id"] for m in st["manifest"]].index(rid)]["gold"][qid]
                        names = evalset.option_names(evalset.request_by_id(rid)["request"]["questions"][qid])
                        g = names.index(gold)
                        f["labeled"] = "gain" if ac == g else ("loss" if ar == g else "both wrong")
                        d["labeled_gains"] += ac == g
                        d["labeled_losses"] += ar == g
                    flips.append(f)
            recdp.append((worst, rid))
        summ = {}
        for src, d in per_src.items():
            dp = d.pop("dp")
            summ[src] = {**d, "dp_mean": sum(dp) / len(dp), "dp_p99": stats.q_nearest(dp, 0.99), "dp_max": max(dp)}
        recdp.sort(reverse=True)
        out["per_ref"][name] = {"per_source": summ, "flips": flips,
                                "largest_dp_records": [{"id": r, "dp_max": w} for w, r in recdp[: args.top]]}
        print(f"\n-- vs {name}")
        for src, d in summ.items():
            print(f"   {src:14s} q={d['questions']:4d} flips={d['flips']} (counted {d['flips_counted']}) "
                  f"mean|dp|={d['dp_mean']:.3g} p99={d['dp_p99']:.3g} max={d['dp_max']:.3g} "
                  f"labeled +{d['labeled_gains']}/-{d['labeled_losses']}")
        for f in flips:
            print(f"   flip {f['id']:24s} {f['qid']:8s} ref margin {f['ref_margin']:.4f} cand margin "
                  f"{f['cand_margin']:.4f} {'COUNTED' if f['counted'] else 'near tie'} {f.get('labeled', '')}")
        print("   largest |dp| records: " + ", ".join(f"{r} {w:.3g}" for w, r in recdp[: args.top]))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
