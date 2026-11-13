"""Score any /v1/systemone server (sysone or qev) on the labeled eval set and apply the statistical gate.

    python3 -I reference/eval/score.py --server http://127.0.0.1:8080 --tier nvfp4
    python3 -I reference/eval/score.py --server http://127.0.0.1:8080 --tier bf16 --model elephant-1
    python3 -I reference/eval/score.py --run-file saved.json --tier nvfp4         # re-score a saved run

Each record of tests/golden/dev/eval/set.json (requests from <sysone-ref>/golden/dev/eval/requests.jsonl,
sha256-checked; override with SYSONE_EVAL_REQUESTS) is POSTed as `/v1/systemone?debias=1`. The answers become
answer-level probability vectors (evalset.py), the representation the committed reference runs use. Then:

  * accuracy per source and overall (bootstrap 95% CI), Brier, ECE; next to every reference run of the tier;
  * the gate of stats.py against EVERY reference path of the tier (nvfp4: the NVFP4 QDQ paths; bf16: the bf16
    qev paths and the fp32 qev path), with the thresholds of tests/tolerances.toml [eval.<tier>].

A subset run (--limit/--only/--sources) is compared on the records it answered only. The committed thresholds
were calibrated on the full set, and the two CI-based checks are not subset-invariant (a 28-record CI is ~3x
wider than a 278-record one, so even a reference path fails them on a subset). For a subset the verdict therefore
uses SUBSET-CALIBRATED limits: the same rule (stats.thresholds: max(factor x envelope, floor)) applied to the
reference paths' pairwise envelope on exactly the candidate's records. The committed-limit checks are still
printed. A subset verdict is indicative; only a full-set run is the gate, so a subset PASS has its own exit code.
Exit status: 0 gate passed on the full set, 1 gate failed, 2 errors/incomplete (a non-200 answer counts as an
error), 4 subset passed with subset-calibrated limits (indicative only, NOT a gate pass).
Stdlib only; the server needs no GPU access from here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalset  # noqa: E402
import stats  # noqa: E402


def post(url: str, body: dict, timeout: float) -> tuple[int, dict, float]:
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read())
        except Exception:
            payload = {"error": str(e)}
        return e.code, payload, time.time() - t0


def run_server(args) -> dict:
    recs = list(evalset.load_requests())
    if args.sources:
        keep = set(args.sources.split(","))
        recs = [r for r in recs if r["source"] in keep]
    if args.only:
        keep = set(args.only.split(","))
        recs = [r for r in recs if r["id"] in keep]
    if args.limit is not None:
        recs = recs[: args.limit]
    url = args.server.rstrip("/") + "/v1/systemone?debias=1"

    def one(item):
        body = dict(item["request"])
        if args.model:
            body["model"] = args.model
        status, payload, dt = post(url, body, args.timeout)
        e = {"id": item["id"], "status": status, "seconds": round(dt, 3)}
        if status == 200:
            try:
                e["answers"] = evalset.answer_vectors(item["request"], payload)
            except (KeyError, ValueError, TypeError) as exc:
                e["status"] = -1
                e["error"] = f"malformed response: {exc}"
        else:
            e["error"] = json.dumps(payload)[:300]
        return e

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, e in enumerate(ex.map(one, recs)):
            results.append(e)
            if args.verbose or e["status"] != 200:
                print(f"  {e['id']:40s} {e['status']} {e['seconds']:.2f}s {e.get('error', '')}", flush=True)
            elif (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(recs)} records, {time.time() - t0:.0f}s", flush=True)
    return {"name": args.name, "config": {"server": args.server, "model": args.model,
                                          "concurrency": args.concurrency, "wall_seconds": round(time.time() - t0, 1)},
            "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--server", help="base URL of a /v1/systemone server")
    src.add_argument("--run-file", type=Path, help="a run saved by --save (no server)")
    ap.add_argument("--model", default=None, help="override the request `model` field (default: jev-latest)")
    ap.add_argument("--tier", choices=["nvfp4", "bf16"], required=True, help="which reference paths/gate to use")
    ap.add_argument("--name", default="candidate")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--sources", default=None, help="comma-separated: banking77,ledgar,longbench_v2")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--save", type=Path, default=None, help="write the candidate run (JSON) here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cand = evalset.load_run(args.run_file) if args.run_file else run_server(args)
    cand["name"] = cand.get("name") or args.name
    if args.save:
        args.save.write_text(json.dumps(cand, separators=(",", ":")) + "\n")
    errors = [r for r in cand["results"] if r["status"] != 200]
    n_set = len(evalset.load_set()["manifest"])

    refs = {p: evalset.load_run(p) for p in stats.TIERS[args.tier] if (evalset.RUNS / f"{p}.json").exists()}
    if not refs:
        raise SystemExit(f"no reference runs for tier {args.tier} under {evalset.RUNS}")
    limits = stats.load_limits(args.tier)

    print(f"\n== {cand['name']}: {len(cand['results'])}/{n_set} records, {len(errors)} errors")
    ids = {r["id"] for r in cand["results"] if r["status"] == 200}

    def subset(run):  # references restricted to the candidate's records, so accuracies are comparable
        return {**run, "results": [r for r in run["results"] if r["id"] in ids]}

    print(f"{'run':22s} {'overall':>18s} " + " ".join(f"{s:>18s}" for s in evalset.load_set()["per_source"])
          + "   brier    ece")
    for name, run in [(cand["name"], cand), *refs.items()]:
        s = stats.run_summary(subset(run))
        cells = []
        for g in ["overall", *evalset.load_set()["per_source"]]:
            if g in s:
                cells.append(f"{s[g]['correct']}/{s[g]['n']} {100 * s[g]['accuracy']:5.1f}%")
            else:
                cells.append("-")
        tail = f"  {s['overall']['brier']:.4f} {s['overall']['ece']:.4f}" if "overall" in s else ""
        print(f"{name:22s} " + " ".join(f"{c:>18s}" for c in cells) + tail)
        if name == cand["name"] and "overall" in s:
            lo, hi = s["overall"]["ci95"]
            print(f"{'':22s} overall 95% CI [{100 * lo:.1f}, {100 * hi:.1f}]%")

    partial = len(ids) < n_set
    sub_refs = {k: subset(v) for k, v in refs.items()}
    g = stats.gate(cand, sub_refs, limits)
    verdicts = [(f"limits from tests/tolerances.toml [eval.{args.tier}]", g)]
    if partial:
        names = list(sub_refs)
        pairs = stats.tier_pairs(sub_refs, names, args.tier)
        anyc = next(iter(pairs.values()))
        th = stats.thresholds(stats.envelope(pairs), anyc["questions"], anyc["records"], args.tier)
        sub_limits = stats.tier_limits(args.tier, th)
        verdicts.append((f"subset-calibrated limits on these {len(ids)} records: "
                         + ", ".join(f"{k}={v:.4g}" for k, v in sub_limits.items()),
                         stats.gate(cand, sub_refs, sub_limits)))
    for title, g in verdicts:
        show(args.tier, list(refs), title, g)
    g = verdicts[-1][1]
    verdict = "PASS" if g["pass"] else "FAIL"
    print(f"\nGATE {verdict}" + (" (subset-calibrated; indicative only)" if partial else "")
          + (f"; {len(errors)} request errors" if errors else ""))
    sys.exit(2 if errors else (1 if not g["pass"] else (4 if partial else 0)))


def show(tier: str, refs: list[str], title: str, g: dict) -> None:
    print(f"\n-- gate [{tier}] vs {refs} ({title}): {'PASS' if g['pass'] else 'FAIL'}")
    for c in g["checks"]:
        if c["metric"] == "info":
            v = c["value"]
            lo, hi = v["acc_diff_ci95"]
            print(f"   {c['ref']:18s} info: labeled agreement {100 * v['agreement_labeled']:.2f}%, top-1 changes"
                  f" {v['disagree_all_count']} (any margin), max|dp| "
                  f"{v['dp_max']:.3f}, dAcc {100 * v['acc_diff']:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}],"
                  f" gains {v['gains']} losses {v['losses']}")
        else:
            op = ">=" if c["metric"] == "acc_ci_lo" else "<="
            print(f"   {c['ref']:18s} {c['metric']:11s} {c['value']:+.4g} {op} {c['limit']:+.4g}  "
                  f"{'ok' if c['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
