"""Run the labeled eval set through the patched qev reference on one path, in resumable time-boxed chunks.

Paths (`--path`), all through qev's own FastAPI app with `?debug=1&debias=1`:
  batched               qev DecisionEngine fork_mode=batched: shared prefix, forked cache, padded suffix batch
  sequential            qev DecisionEngine fork_mode=sequential: shared prefix, one suffix forward per branch
  per_question          one independent full forward per branch, batch 1, use_cache=False (the NVFP4 card's
                        evaluation path; qevref.per_question_mode)
  per_question_batched  every branch of a request as an independent full sequence (prefix + suffix, no cache),
                        right-padded into ONE batch of N rows: the per-question math with a batch-N kernel shape

Every GPU invocation must go through scripts/gpu.sh. One invocation loads the model, then runs records until
--max-seconds of wall time (load included) have passed, writes one part file and exits; rerun the same command
to resume (records already in a part file are skipped). `run_chunks.sh` loops this until a run is complete, so
each locked command stays under ~10 minutes and other GPU users get the lock in between.

    scripts/gpu.sh venv/bin/python -I reference/eval/run_eval.py --name nvfp4_perq --nvfp4 --path per_question
    venv/bin/python -I reference/eval/run_eval.py --name nvfp4_perq --merge     # CPU: parts -> committed run

Parts (full slot logits, full response JSON): <sysone-ref>/golden/dev/eval/runs/<name>/part-*.json.
Merged run (committed, compact): tests/golden/dev/eval/runs/<name>.json; answer-level probability vectors (the
same representation reference/eval/score.py reads from any server), rounded as qev rounds them (1e-6).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import qevref  # noqa: E402
import evalset  # noqa: E402

PARTS = qevref.GOLDEN_BIN / "eval" / "runs"
RUNS = qevref.REPO / "tests" / "golden" / "dev" / "eval" / "runs"


def per_question_batched_mode(engine) -> None:
    """All branches of a request as independent full sequences (no cache), right-padded into one batch. Same
    block-bidirectional mask and position ids as the per-question path; only the batch shape differs."""
    import torch
    from qev import readout as ro
    from qev.engine import BranchResult

    def run(state, rendered):
        prefix_ids, suffixes, _ = engine._split(state, rendered)
        P = len(prefix_ids)
        dev = engine.device
        lens = [P + len(s) for s in suffixes]
        width = max(lens)
        n = len(suffixes)
        ids = torch.full((n, width), engine.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros((n, width), dtype=torch.long, device=dev)
        for row, s in enumerate(suffixes):
            ids[row, : lens[row]] = torch.tensor(prefix_ids + s, device=dev)
            mask[row, : lens[row]] = 1
        ends = [m - 1 for m in lens]
        keep = sorted(set(ends))
        out = []
        with torch.inference_mode():
            with ro.blocks(ro.block_ids([P] * n, lens, width, dev)):
                o = engine.model(input_ids=ids, attention_mask=mask,
                                 position_ids=torch.arange(width, device=dev)[None].expand(n, width),
                                 use_cache=False, return_dict=True,
                                 logits_to_keep=torch.tensor(keep, dtype=torch.long, device=dev))
            for row, (r, suf) in enumerate(zip(rendered, suffixes)):
                logits, mass = engine._readout(o.logits[row, keep.index(ends[row])], r.n_slots)
                out.append(BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, len(suf)))
            del o
        return out

    engine.info.fork_mode = "per_question_batched"
    engine._run_sequential = run
    engine._run_batched = run


def done_ids(name: str) -> set[str]:
    out = set()
    for p in sorted((PARTS / name).glob("part-*.json")):
        out |= {r["id"] for r in json.loads(p.read_text())["results"]}
    return out


def run(args) -> None:
    import torch
    from fastapi.testclient import TestClient
    from qev.calibrate import softmax
    from qev.server import create_app

    t0 = time.time()
    records = evalset.load_requests()
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

    model = qevref.NVFP4_MODEL if args.nvfp4 else qevref.BF16_MODEL
    fork = {"batched": "batched", "sequential": "sequential"}.get(args.path, "sequential")
    engine = qevref.load_engine(model, args.dtype, fork_mode=fork, nvfp4=args.nvfp4)
    if args.path == "per_question":
        qevref.per_question_mode(engine)
    elif args.path == "per_question_batched":
        per_question_batched_mode(engine)
    load_s = time.time() - t0
    client = TestClient(create_app(engine, model_name=model.name))

    record: dict = {}

    def wrap(fn):
        def inner(state, rendered):
            prefix_ids, suffixes, _ = engine._split(state, rendered)
            out = fn(state, rendered)
            record.update(prefix_ids=prefix_ids, suffixes=suffixes, branches=out)
            return out
        return inner

    ran = {"batched": "batched", "sequential": "sequential"}.get(args.path, args.path)
    engine._run_batched = wrap(engine._run_batched)
    engine._run_sequential = wrap(engine._run_sequential)
    # decide() falls back from batched to sequential on OOM; detect it so a run never silently mixes paths
    fallbacks = []

    results = []
    for item in todo:
        if time.time() - t0 > args.max_seconds:
            break
        record.clear()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        r = client.post("/v1/systemone?debug=1&debias=1", json=item["request"])
        dt = time.time() - t1
        body = r.json()
        if r.status_code != 200:  # every record of the set is eligible: anything else is a harness error
            raise SystemExit(f"{item['id']}: HTTP {r.status_code}: {json.dumps(body)[:500]}")
        meta = body.get("qev") or {}
        if isinstance(meta, dict):
            meta.pop("timing", None)
            if args.path == "batched" and meta.get("fork_mode") != "batched":
                fallbacks.append(item["id"])
        entry = {"id": item["id"], "status": r.status_code, "response": body, "seconds": round(dt, 3),
                 "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if record:
            entry["prefix_tokens"] = len(record["prefix_ids"])
            entry["prefix_sha256"] = qevref.sha256_ids(record["prefix_ids"])
            entry["branches"] = [
                {"qid": b.qid, "kind": b.kind, "n_slots": len(b.logits), "suffix_tokens": b.suffix_tokens,
                 "suffix_sha256": qevref.sha256_ids(s), "logits": [float(x) for x in b.logits],
                 "probs": softmax(b.logits), "allowed_mass": b.allowed_mass}
                for b, s in zip(record["branches"], record["suffixes"])]
        results.append(entry)
        print(f"{item['id']:36s} {r.status_code} {dt:7.2f}s prefix={entry.get('prefix_tokens', '-')} "
              f"peak={entry['peak_gpu_gb']}GB", flush=True)
        record.clear()
        gc.collect()

    if not results:
        print("time budget spent on loading; nothing run")
        return
    part = {
        "name": args.name,
        "config": {"model": model.name, "dtype": args.dtype, "nvfp4": args.nvfp4, "path": args.path, "ran": ran,
                   "head_dtype": engine.head_dtype, "engine_fork_mode": engine.info.fork_mode,
                   "requests_sha256": evalset.load_set()["requests_sha256"]},
        "provenance": qevref.provenance({"model_files_sha256": qevref.model_files_sha(model),
                                         "load_seconds": round(load_s, 1)}),
        "batched_fallbacks": fallbacks,
        "results": results,
    }
    out = PARTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    idx = len(list(out.glob("part-*.json")))
    path = out / f"part-{idx:03d}.json"
    path.write_text(json.dumps(part, ensure_ascii=False, separators=(",", ":")))
    left = len(todo) - len(results)
    print(f"wrote {path}: {len(results)} records in {time.time() - t0:.0f}s (load {load_s:.0f}s); {left} left")


def merge(args) -> None:
    """Parts -> the committed compact run: answer-level vectors per question plus enough provenance."""
    st = evalset.load_set()
    parts = [json.loads(p.read_text()) for p in sorted((PARTS / args.name).glob("part-*.json"))]
    if not parts:
        raise SystemExit(f"no parts for {args.name}")
    configs = {json.dumps(p["config"], sort_keys=True) for p in parts}
    if len(configs) != 1:
        raise SystemExit(f"{args.name}: parts disagree on config: {configs}")
    by_id = {}
    for p in parts:
        for r in p["results"]:
            if r["id"] in by_id:
                raise SystemExit(f"{args.name}: {r['id']} appears twice")
            by_id[r["id"]] = r
    want = [m["id"] for m in st["manifest"]]
    missing = [i for i in want if i not in by_id]
    results = []
    for rid in want:
        if rid not in by_id:
            continue
        r = by_id[rid]
        e = {"id": rid, "status": r["status"], "seconds": r["seconds"]}
        if r["status"] == 200:
            e["prefix_tokens"] = r["prefix_tokens"]
            e["answers"] = evalset.answer_vectors(evalset.request_by_id(rid)["request"], r["response"])
            # cross-check: with debias 1 and no chunking the answer vector is the branch softmax (to rounding)
            for b in r["branches"]:
                v = e["answers"][b["qid"]]
                assert len(v) == len(b["probs"]) and max(abs(x - y) for x, y in zip(v, b["probs"])) < 2e-6, rid
        results.append(e)
    provs = [p["provenance"] for p in parts]
    out = {
        "name": args.name,
        "config": parts[0]["config"],
        "provenance": {**{k: v for k, v in provs[0].items() if k != "load_seconds"},
                       "load_seconds": [p.get("load_seconds") for p in provs], "parts": len(parts)},
        "batched_fallbacks": sorted({x for p in parts for x in p.get("batched_fallbacks", [])}),
        "missing": missing,
        "results": results,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
    # sha256 manifest of the out-of-repo part files (full logits / responses) behind every committed run
    man_path = RUNS.parent / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    man[args.name] = {str(p.relative_to(qevref.GOLDEN_BIN)): {"bytes": p.stat().st_size,
                                                               "sha256": qevref.sha256_file(p)}
                      for p in sorted((PARTS / args.name).glob("part-*.json"))}
    man_path.write_text(json.dumps(dict(sorted(man.items())), indent=1) + "\n")
    print(f"merged {args.name}:{len(results)} records from {len(parts)} parts, {len(missing)} missing, "
          f"{len(out['batched_fallbacks'])} batched fallbacks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--nvfp4", action="store_true")
    ap.add_argument("--path", choices=["batched", "sequential", "per_question", "per_question_batched"])
    ap.add_argument("--max-seconds", type=float, default=480.0,
                    help="stop starting new records after this much wall time (load included)")
    ap.add_argument("--only", default=None, help="comma-separated record ids")
    ap.add_argument("--limit", type=int, default=None, help="first N records of the set only")
    ap.add_argument("--merge", action="store_true", help="merge the parts into the committed run (no GPU)")
    args = ap.parse_args()
    if args.merge:
        merge(args)
    else:
        if not args.path:
            ap.error("--path is required")
        run(args)


if __name__ == "__main__":
    main()
