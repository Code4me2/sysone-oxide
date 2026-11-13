"""The labeled eval set (stdlib only): manifest, requests, and the answer-level probability vectors every
comparison uses.

`tests/golden/dev/eval/set.json` is the committed manifest; the requests themselves (LongBench-v2 contexts make
them ~MBs) live in `<sysone-ref>/golden/dev/eval/requests.jsonl` and are checked against the manifest's sha256
before use. Both are produced by build_set.py.

Answer vectors: for each question of a response, its probabilities in criteria order:
  choice  [probabilities[name] for name in criteria]
  score   [probabilities[str(i)] for i in range(levels)]
  noul    [noul, 1 - noul]
This is what any /v1/systemone server returns (rounded to 1e-6 by qev), so reference runs and candidate servers
are compared in exactly the same representation.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SET_JSON = REPO / "tests" / "golden" / "dev" / "eval" / "set.json"
RUNS = REPO / "tests" / "golden" / "dev" / "eval" / "runs"
REF = Path(os.environ.get("SYSONE_REF", "/home/velvet/Desktop/sysone-ref"))
REQUESTS = Path(os.environ.get("SYSONE_EVAL_REQUESTS", REF / "golden" / "dev" / "eval" / "requests.jsonl"))


@lru_cache(maxsize=1)
def load_set() -> dict:
    return json.loads(SET_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_requests() -> tuple[dict, ...]:
    raw = REQUESTS.read_bytes()
    want = load_set()["requests_sha256"]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SystemExit(f"{REQUESTS}: sha256 {got} does not match set.json ({want}); rerun build_set.py")
    # split on \n only: request strings may contain U+2028/U+2029
    out = tuple(json.loads(line) for line in raw.decode("utf-8").split("\n") if line)
    ids = [m["id"] for m in load_set()["manifest"]]
    assert [r["id"] for r in out] == ids, "requests.jsonl and set.json disagree on record order"
    return out


def request_by_id(rid: str) -> dict:
    return _by_id()[rid]


@lru_cache(maxsize=1)
def _by_id() -> dict:
    return {r["id"]: r for r in load_requests()}


def option_names(question: dict) -> list[str]:
    if question["type"] == "choice":
        return list(question["criteria"].keys())
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["yes", "no"]


def answer_vectors(request: dict, response: dict) -> dict[str, list[float]]:
    """Answer-level probability vector per question id (criteria order). Raises if the response is malformed."""
    out = {}
    answers = response["answers"]
    for qid, q in request["questions"].items():
        a = answers[qid]
        if a["type"] != q["type"]:
            raise ValueError(f"{qid}: answer type {a['type']} for a {q['type']} question")
        if q["type"] == "noul":
            p = float(a["noul"])
            out[qid] = [p, 1.0 - p]
        else:
            probs = a["probabilities"]
            names = option_names(q)
            if set(probs) != set(names):
                raise ValueError(f"{qid}: probability keys differ from the options")
            out[qid] = [float(probs[n]) for n in names]
    return out


def load_run(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = RUNS / f"{name_or_path}.json"
    return json.loads(p.read_text(encoding="utf-8"))
