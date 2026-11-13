"""Run the model-side golden corpus through the patched qev reference and record every branch (DESIGN.md §9.1).

Paths (`--path`):
  batched       qev DecisionEngine, fork_mode=batched (shared prefix, forked cache, padded suffix batch)
  sequential    qev DecisionEngine, fork_mode=sequential (shared prefix, one suffix forward per branch)
  per_question  one independent full forward per branch, batch 1, use_cache=False (the NVFP4 evaluation path)

Every request goes through qev's own FastAPI app (`qev.server.create_app`) with `?debug=1&debias=k`, so the
recorded response JSON (and every 422) is byte-for-byte what the reference server returns, minus the
`qev.timing` field. Per branch it records the slot logits (first n_slots, fp32 from the fp32 LM head), the
probabilities (softmax, T=1) and the token counts.

    venv/bin/python -I reference/run_golden.py --dtype bfloat16 --path batched --name bf16_batched
    venv/bin/python -I reference/run_golden.py --nvfp4 --path per_question --name nvfp4_perq_r1

Output: tests/golden/dev/model/runs/<name>.json (committed) and, for requests marked dump_hidden (unless
--no-hidden), <golden bin>/<name>/hidden_<id>.safetensors via layer_dump.dump_tree.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_dump  # noqa: E402
import qevref  # noqa: E402

RUNS = qevref.REPO / "tests" / "golden" / "dev" / "model" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--nvfp4", action="store_true")
    ap.add_argument("--path", required=True, choices=["batched", "sequential", "per_question"])
    ap.add_argument("--corpus", type=Path, default=qevref.REPO / "reference" / "corpus_model" / "requests.jsonl")
    ap.add_argument("--only", default=None, help="comma-separated request ids (default: all)")
    ap.add_argument("--max-prefix-tokens", type=int, default=None,
                    help="skip requests whose prefix is longer (logged as dropped); for slow fp32 runs")
    ap.add_argument("--no-hidden", action="store_true")
    ap.add_argument("--hidden-mode", default=None, choices=["served", "per_question"])
    args = ap.parse_args()

    import torch
    from fastapi.testclient import TestClient
    from qev.calibrate import softmax
    from qev.server import create_app

    model = qevref.NVFP4_MODEL if args.nvfp4 else qevref.BF16_MODEL
    t0 = time.time()
    fork = "sequential" if args.path == "per_question" else args.path
    engine = qevref.load_engine(model, args.dtype, fork_mode=fork, nvfp4=args.nvfp4)
    if args.path == "per_question":
        qevref.per_question_mode(engine)
    load_s = time.time() - t0
    app = create_app(engine, model_name=model.name)
    client = TestClient(app)

    # record what each decide() actually ran (decide() may fall back from batched to sequential on OOM)
    record: dict = {}

    def wrap(name, fn):
        def run(state, rendered):
            prefix_ids, suffixes, _ = engine._split(state, rendered)
            out = fn(state, rendered)
            record.update(ran=name, prefix_ids=prefix_ids, suffixes=suffixes, rendered=rendered, branches=out)
            return out
        return run

    engine._run_batched = wrap("batched" if args.path != "per_question" else "per_question", engine._run_batched)
    engine._run_sequential = wrap("sequential" if args.path != "per_question" else "per_question",
                                  engine._run_sequential)

    corpus = qevref.load_corpus(args.corpus)
    only = set(args.only.split(",")) if args.only else None
    bin_dir = qevref.GOLDEN_BIN / args.name
    results, dropped = [], []
    for item in corpus:
        if only and item["id"] not in only:
            continue
        if args.max_prefix_tokens is not None and item["expect"] == "ok":
            prefix_ids, _ = layer_dump.tree_ids(engine, item["request"], item["debias"])
            if len(prefix_ids) > args.max_prefix_tokens:
                dropped.append({"id": item["id"], "prefix_tokens": len(prefix_ids),
                                "reason": f"prefix > {args.max_prefix_tokens} tokens"})
                continue
        record.clear()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        r = client.post(f"/v1/systemone?debug=1&debias={item['debias']}", json=item["request"])
        dt = time.time() - t1
        body = r.json()
        if isinstance(body.get("qev"), dict):
            body["qev"].pop("timing", None)
        entry = {"id": item["id"], "status": r.status_code, "response": body, "seconds": round(dt, 3),
                 "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if record:
            entry.update(ran=record["ran"], prefix_tokens=len(record["prefix_ids"]),
                         prefix_sha256=qevref.sha256_ids(record["prefix_ids"]))
            entry["branches"] = [
                {"qid": b.qid, "kind": b.kind, "labels": b.labels, "n_slots": len(b.logits),
                 "suffix_tokens": b.suffix_tokens, "suffix_sha256": qevref.sha256_ids(s),
                 "logits": [float(x) for x in b.logits], "probs": softmax(b.logits),
                 "allowed_mass": b.allowed_mass}
                for b, s in zip(record["branches"], record["suffixes"])]
            if item.get("dump_hidden") and not args.no_hidden:
                mode = args.hidden_mode or ("per_question" if args.path == "per_question" else "served")
                tensors = layer_dump.dump_tree(engine, record["prefix_ids"], record["suffixes"], mode)
                path = bin_dir / f"hidden_{item['id']}.safetensors"
                layer_dump.save(tensors, path, {"request": item["id"], "model": model.name, "dtype": args.dtype,
                                                "nvfp4": args.nvfp4, "mode": mode,
                                                "prefix_rows": tensors["prefix_rows"].tolist()})
                entry["hidden"] = {"file": str(path.relative_to(qevref.GOLDEN_BIN)), "mode": mode}
        results.append(entry)
        print(f"{item['id']:28s} {r.status_code} {dt:7.2f}s prefix={entry.get('prefix_tokens', '-')} "
              f"branches={len(entry.get('branches', []))} peak={entry['peak_gpu_gb']}GB", flush=True)
        record.clear()
        gc.collect()

    out = {
        "name": args.name,
        "config": {"model": model.name, "dtype": args.dtype, "nvfp4": args.nvfp4, "path": args.path,
                   "head_dtype": engine.head_dtype, "engine_fork_mode": engine.info.fork_mode,
                   "corpus": str(args.corpus.relative_to(qevref.REPO)),
                   "corpus_sha256": qevref.sha256_file(args.corpus)},
        "provenance": qevref.provenance({"model_files_sha256": qevref.model_files_sha(model),
                                         "load_seconds": round(load_s, 1)}),
        "dropped": dropped,
        "results": results,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=None, separators=(",", ":")))
    print(f"wrote {RUNS / (args.name + '.json')} ({len(results)} requests, {len(dropped)} dropped) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
