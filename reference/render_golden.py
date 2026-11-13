"""Render goldens: run every corpus request through the reference frontend (no weights) and record what it renders.

    python -I reference/render_golden.py [--corpus reference/corpus/requests.jsonl] [--out tests/golden/dev/render]
                                         [--model /home/velvet/Desktop/hf_models/elephant-1]

Each request is POSTed to the reference server itself (`qev.server.create_app`, via Starlette's TestClient), whose
engine is the weightless `qev_shim.ShimEngine`, so schema validation, routing, `decide()`'s planning, limit checks,
serve.patch's `_split` and the error responses are all the reference's own code. Nothing is reimplemented here;
this script only serializes what those code paths produced.

Output (deterministic: a rerun is byte-identical):
    <out>/<group>.jsonl     one line per corpus case, in corpus order (format: reference/README.md)
    <out>/provenance.json   versions, hashes, slot table, limits, per-file manifest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

DEFAULT_MODEL = "/home/velvet/Desktop/hf_models/elephant-1"
SERVED_NAME = "elephant-1"
ONEJEV_COMMIT = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
FORMAT = "sysone-render-golden-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def render_record(shim, state, plan: dict) -> dict | None:
    """The plan `decide()` built (and serve.patch's `_split`, if it ran), with each branch's full prompt.

    Every branch prompt is `prompt_head + suffix_text + prompt_tail` (checked here for each branch); the head (system
    turn + state block) and the tail (end of the user turn + generation prompt) are stored once per request."""
    rendered = plan.get("rendered")
    if not rendered:
        return None
    split = plan.get("split")
    prefix_ids = plan["prefix_ids"]
    suffixes = plan["suffixes"]
    owners = plan["owners"]
    branches = []
    head = tail = None
    for i, r in enumerate(rendered):
        qid, order, chunk = owners[i]
        b = {"qid": r.qid, "kind": r.kind, "chunk": chunk, "order": list(order), "labels": list(r.labels),
             "n_slots": r.n_slots, "descriptions": list(r.descriptions), "suffix_text": r.suffix,
             "option_char_spans": [list(s) for s in r.option_spans] if r.option_spans is not None else None}
        prompt = shim._render_prompt(state, r.suffix)
        start = prompt.rindex(r.suffix)  # as serve.patch's _split locates the suffix
        h, t = prompt[:start], prompt[start + len(r.suffix):]
        if head is None:
            head, tail = h, t
        elif (h, t) != (head, tail):
            raise AssertionError(f"branch {i}: prompt is not prompt_head + suffix_text + prompt_tail")
        if prefix_ids is not None:
            b["suffix_ids"] = list(suffixes[i])
            # Self-check of the recording, not of the reference: the branch is the prefix plus its suffix.
            if shim._encode(prompt) != list(prefix_ids) + list(suffixes[i]):
                raise AssertionError(f"branch {i}: prompt tokenization != prefix_ids + suffix_ids")
        if split is not None:
            if split["prefix_ids"] != prefix_ids or split["suffixes"][i] != suffixes[i]:
                raise AssertionError("_split and _prefix_and_suffix_ids disagree")
            sp = split["spans"][i]
            b["option_token_spans"] = [list(s) for s in sp] if sp is not None else None
        branches.append(b)
    return {"prefix_ids": list(prefix_ids) if prefix_ids is not None else None,
            "prefix_tokens": len(prefix_ids) if prefix_ids is not None else None,
            "longest": plan.get("longest"), "total": plan.get("total"),
            "split_ran": split is not None, "error": plan.get("error"),
            "prompt_head": head, "prompt_tail": tail, "branches": branches}


def http_record(resp) -> dict:
    body = resp.json()
    if resp.status_code == 200:
        # Answers come from the shim's zero logits and timings from the clock: neither is a golden.
        body.pop("answers", None)
        body.get("qev", {}).pop("timing", None)
    return {"status": resp.status_code, "body": body}


def engine_direct(shim, body: dict, debias: int) -> dict:
    """Plan a request the schema rejects (choice > 255 options) by calling the engine with an unvalidated request,
    the way a non-HTTP caller of DecisionEngine.decide could."""
    from qev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

    qs = {}
    for qid, q in body["questions"].items():
        cls = {"noul": NoulQuestion, "choice": ChoiceQuestion, "score": ScoreQuestion}[q["type"]]
        qs[qid] = cls.model_construct(**q)
    req = SystemOneRequest.model_construct(state=body.get("state"), questions=qs)
    try:
        resp, meta = shim.decide(req, debias=debias)
        outcome = {"ok": True, "usage": resp.usage.model_dump(),
                   "qev": {k: v for k, v in meta.items() if k != "timing"}}
    except ValueError as exc:
        outcome = {"ok": False, "error": str(exc)}
    rec = render_record(shim, req.state, shim.last or {})
    return {"outcome": outcome, "render": rec}


def versions() -> dict:
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for pkg in ("transformers", "tokenizers", "torch", "pydantic", "pydantic_core", "fastapi", "starlette", "httpx", "jinja2", "qev"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=str(HERE / "corpus" / "requests.jsonl"))
    ap.add_argument("--out", default=str(REPO / "tests" / "golden" / "dev" / "render"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only", default=None, help="comma-separated case ids (debugging; writes nothing)")
    ap.add_argument("--check", action="store_true", help="regenerate into a temp dir and fail unless byte-identical to --out")
    args = ap.parse_args(argv)
    if args.check:
        import filecmp
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            rest = [a for a in (argv if argv is not None else sys.argv[1:]) if a != "--check"]
            main(rest + ["--out", tmp])
            want = sorted(p.name for p in Path(args.out).iterdir())
            got = sorted(p.name for p in Path(tmp).iterdir())
            same = want == got and all(filecmp.cmp(Path(tmp) / n, Path(args.out) / n, shallow=False) for n in got)
        print(("OK: " if same else "MISMATCH: ") + f"regenerated goldens vs {args.out}", file=sys.stderr)
        return 0 if same else 1

    import qev
    from fastapi.testclient import TestClient
    from qev.prompt import PROMPT_VERSION, SLOT_LABELS, SYSTEM_PROMPT
    from qev.server import create_app
    from qev_shim import ShimEngine

    model = Path(args.model)
    shim = ShimEngine(str(model))
    client = TestClient(create_app(shim, SERVED_NAME))
    only = set(args.only.split(",")) if args.only else None

    # split on "\n" only: str.splitlines() would also split on U+2028 etc. inside JSON strings
    lines = [l for l in Path(args.corpus).read_text(encoding="utf-8").split("\n") if l]
    groups: dict[str, list[str]] = {}
    counts = {"cases": 0, "status": {}}
    for line in lines:
        case = json.loads(line)
        if only is not None and case["id"] not in only:
            continue
        body, debias = case["body"], case["debias"]
        payload = dumps(body).encode("utf-8")
        shim.last = None
        resp = client.post(f"/v1/systemone?debias={debias}", content=payload, headers={"content-type": "application/json"})
        state = body.get("state")
        rec = {"id": case["id"], "group": case["group"], "debias": debias,
               "request_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
               "http": http_record(resp),
               "render": render_record(shim, state, shim.last) if shim.last else None}
        if case.get("engine_direct"):
            rec["engine_direct"] = engine_direct(shim, body, debias)
        out_line = dumps(rec)
        groups.setdefault(case["group"], []).append(out_line)
        counts["cases"] += 1
        st = str(resp.status_code)
        counts["status"][st] = counts["status"].get(st, 0) + 1
        if only is not None:
            print(out_line[:4000])

    if only is not None:
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for group, recs in groups.items():
        text = "".join(r + "\n" for r in recs)
        (out / f"{group}.jsonl").write_text(text, encoding="utf-8", newline="\n")
        manifest[f"{group}.jsonl"] = {"cases": len(recs), "sha256": sha256_text(text)}

    patched = Path(qev.__file__).resolve().parent.parent
    files = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json", "readout.json", "serve.patch")
    prov = {
        "format": FORMAT,
        "generator": {"script": "reference/render_golden.py", "shim": "reference/qev_shim.py",
                      "corpus": str(Path(args.corpus).resolve().relative_to(REPO)) if Path(args.corpus).resolve().is_relative_to(REPO) else args.corpus,
                      "corpus_sha256": sha256_file(Path(args.corpus)),
                      "sha256": {name: sha256_file(HERE / name) for name in ("render_golden.py", "qev_shim.py", "corpus/generate.py")}},
        "reference": {"onejev_commit": ONEJEV_COMMIT, "serve_patch": "elephant-1/serve.patch",
                      "qev_path": str(patched), "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
                      "served_model_name": SERVED_NAME, "server_entry": "qev.server.create_app via starlette TestClient",
                      "fork_mode": shim.info.fork_mode, "attention": shim.readout.config.attention,
                      "readout": shim.readout.config.readout},
        "checkpoint": {"path": str(model), "sha256": {f: sha256_file(model / f) for f in files if (model / f).exists()}},
        "versions": versions(),
        "limits": {"max_branch_tokens": shim.max_branch_tokens, "max_request_tokens": shim.max_request_tokens},
        "tokenizer": {"pad_id": shim.pad_id, "eos_id": shim.tokenizer.eos_token_id, "max_slots": shim.max_slots,
                      "template_kwargs": shim.template_kwargs},
        "slots": [{"label": l, "id": i} for l, i in zip(SLOT_LABELS, shim.slot_ids)],
        "counts": counts,
        "files": manifest,
    }
    (out / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {counts['cases']} cases in {len(groups)} files to {out}; status {counts['status']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
