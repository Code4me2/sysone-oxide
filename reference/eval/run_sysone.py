"""Run the labeled eval set through a live sysone server over HTTP, in resumable time-boxed chunks, and merge the
parts into a candidate run in exactly the format of the committed reference runs (stdlib only, no GPU access
from here; the server is the GPU job, started by run_sysone.sh under scripts/gpu.sh).

Every record is POSTed with its unchanged request body as `/v1/systemone?debug=1&debias=1`, the query the
reference runs used (run_eval.py), one request at a time by default (the reference runs' request pattern; the
server may still batch internally). A non-200 answer is kept as an error record (never dropped): the merged run
then has status != 200 for it and score.py exits 2.

    python3 -I reference/eval/run_sysone.py --name sysone_bf16 --server http://127.0.0.1:8766 --build build.json
    python3 -I reference/eval/run_sysone.py --name sysone_bf16 --merge

Parts (full response JSON incl. the debug slot logits): <sysone-ref>/golden/dev/eval/candidates/<name>/part-*.json.
Merged run (committed): tests/golden/dev/eval/candidates/<name>.json, sha256 of the parts in
tests/golden/dev/eval/candidates/manifest.json. Candidate runs live outside runs/ on purpose: stats.py reads every
file in runs/ as a reference path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evalset  # noqa: E402

PARTS = evalset.REF / "golden" / "dev" / "eval" / "candidates"
OUT = evalset.REPO / "tests" / "golden" / "dev" / "eval" / "candidates"
QUERY = "/v1/systemone?debug=1&debias=1"


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
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return -2, {"error": f"transport: {e}"}, time.time() - t0


def done_ids(name: str) -> set[str]:
    out = set()
    for p in sorted((PARTS / name).glob("part-*.json")):
        out |= {r["id"] for r in json.loads(p.read_text())["results"]}
    return out


def run(args) -> None:
    t0 = time.time()
    records = list(evalset.load_requests())
    if args.only:
        only = set(args.only.split(","))
        records = [r for r in records if r["id"] in only]
    if args.limit is not None:
        records = records[: args.limit]
    done = done_ids(args.name)
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        print(f"{args.name}: complete ({len(done)} records)")
        return
    build = json.loads(args.build.read_text()) if args.build else {}
    url = args.server.rstrip("/") + QUERY
    results = []
    for item in todo:
        if time.time() - t0 > args.max_seconds:
            break
        status, body, dt = post(url, item["request"], args.timeout)
        entry = {"id": item["id"], "status": status, "response": body, "seconds": round(dt, 3)}
        meta = body.get("qev") if isinstance(body, dict) else None
        if isinstance(meta, dict):
            entry["prefix_tokens"] = meta.get("prefix_tokens")
        results.append(entry)
        print(f"{item['id']:36s} {status} {dt:7.2f}s prefix={entry.get('prefix_tokens', '-')}", flush=True)
    if not results:
        print("time budget spent; nothing run")
        return
    part = {
        "name": args.name,
        "config": {"server": "sysone", "query": QUERY, "concurrency": 1,
                   "requests_sha256": evalset.load_set()["requests_sha256"], **build.get("config", {})},
        "provenance": build.get("provenance", {}),
        "chunk_wall_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    out = PARTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    idx = len(list(out.glob("part-*.json")))
    path = out / f"part-{idx:03d}.json"
    path.write_text(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
    left = len(todo) - len(results)
    errs = sum(r["status"] != 200 for r in results)
    print(f"wrote {path}: {len(results)} records ({errs} errors) in {time.time() - t0:.0f}s; {left} left")


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def crosscheck(rid: str, answers: dict, response: dict) -> float:
    """With debias 1 and no chunking the answer vector is the softmax of the question's slot logits (debug
    diagnostics), to qev's 1e-6 rounding. Returns the worst deviation (the harness reads what it thinks)."""
    diag = (response.get("qev") or {}).get("diagnostics") or {}
    worst = 0.0
    for qid, v in answers.items():
        sl = diag[qid]["slot_logits"]
        rows = sl if sl and isinstance(sl[0], list) else [sl]
        assert len(rows) == 1, (rid, qid, "chunked question")
        p = softmax([float(x) for x in rows[0]])
        assert len(p) == len(v), (rid, qid, len(p), len(v))
        worst = max(worst, max(abs(x - y) for x, y in zip(p, v)))
    return worst


def merge(args) -> None:
    st = evalset.load_set()
    parts = [json.loads(p.read_text()) for p in sorted((PARTS / args.name).glob("part-*.json"))]
    if not parts:
        raise SystemExit(f"no parts for {args.name}")
    configs = {json.dumps(p["config"], sort_keys=True) for p in parts}
    if len(configs) != 1:
        raise SystemExit(f"{args.name}: parts disagree on config: {configs}")
    builds = {json.dumps({k: v for k, v in p["provenance"].items() if k != "load_seconds"}, sort_keys=True)
              for p in parts}
    if len(builds) != 1:
        raise SystemExit(f"{args.name}: parts disagree on provenance (engine build): {builds}")
    by_id = {}
    for p in parts:
        for r in p["results"]:
            if r["id"] in by_id:
                raise SystemExit(f"{args.name}: {r['id']} appears twice")
            by_id[r["id"]] = r
    want = [m["id"] for m in st["manifest"]]
    extra = sorted(set(by_id) - set(want))
    if extra:
        raise SystemExit(f"{args.name}: records not in the set: {extra[:5]}")
    missing = [i for i in want if i not in by_id]
    results, worst = [], 0.0
    for rid in want:
        if rid not in by_id:
            continue
        r = by_id[rid]
        e = {"id": rid, "status": r["status"], "seconds": r["seconds"]}
        if r["status"] == 200:
            e["prefix_tokens"] = r["prefix_tokens"]
            try:
                e["answers"] = evalset.answer_vectors(evalset.request_by_id(rid)["request"], r["response"])
                worst = max(worst, crosscheck(rid, e["answers"], r["response"]))
            except (KeyError, ValueError, TypeError) as exc:
                e["status"] = -1
                e["error"] = f"malformed response: {exc}"
                e.pop("answers", None)
        else:
            e["error"] = json.dumps(r["response"])[:300]
        results.append(e)
    if worst >= 2e-6:
        raise SystemExit(f"{args.name}: answer vectors differ from softmax(slot_logits) by {worst:.3g}")
    provs = [p["provenance"] for p in parts]
    out = {
        "name": args.name,
        "config": parts[0]["config"],
        "provenance": {**{k: v for k, v in provs[0].items() if k != "load_seconds"},
                       "load_seconds": [p.get("load_seconds") for p in provs], "parts": len(parts),
                       "chunk_wall_seconds": [p.get("chunk_wall_seconds") for p in parts]},
        "batched_fallbacks": [],
        "missing": missing,
        "results": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man[args.name] = {str(p.relative_to(evalset.REF)): {"bytes": p.stat().st_size,
                                                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted((PARTS / args.name).glob("part-*.json"))}
    man_path.write_text(json.dumps(dict(sorted(man.items())), indent=1) + "\n")
    errs = sum(r["status"] != 200 for r in results)
    print(f"merged {args.name}: {len(results)} records from {len(parts)} parts, {len(missing)} missing, "
          f"{errs} errors; answers vs softmax(slot_logits) max {worst:.2g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8766")
    ap.add_argument("--build", type=Path, default=None, help="JSON {config, provenance} written by run_sysone.sh")
    ap.add_argument("--max-seconds", type=float, default=420.0,
                    help="stop starting new records after this much wall time of this chunk")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--only", default=None, help="comma-separated record ids")
    ap.add_argument("--limit", type=int, default=None, help="first N records of the set only")
    ap.add_argument("--merge", action="store_true", help="merge the parts into the committed candidate run")
    args = ap.parse_args()
    merge(args) if args.merge else run(args)


if __name__ == "__main__":
    main()
