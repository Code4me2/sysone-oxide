"""Build the labeled eval set for the statistical NVFP4 gate (DESIGN §9.2) from public sources.

The NVFP4 checkpoint's card evaluated a held-out set of 278 labeled tasks (128 BANKING77, 128 LEDGAR, 22 eligible
LongBench-v2), each rendered as one Decision-Index-style request with the labeled question plus two unlabeled
support questions (556 = 2 x 278). Its records were not published, so this script rebuilds an *equivalent* set
from the same public sources with a fixed seed. It is not, and cannot be, identical to the card's records:
the sampling, the question wording and the two support questions are ours. Every choice is listed in
reference/eval/README.md.

    bash reference/eval/fetch.sh                                  # pinned downloads + parquet -> jsonl
    venv/bin/python -I reference/eval/build_set.py                # CPU only, ~1 min (tokenizes LongBench-v2)

Outputs (byte-deterministic on rerun):
  <sysone-ref>/golden/dev/eval/requests.jsonl    one request per line (LongBench contexts make it ~3 MB)
  tests/golden/dev/eval/set.json                 manifest: per record source/index/gold/token counts/sha256,
                                                 the sha256 of requests.jsonl and of every source file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import qevref  # noqa: E402

DATA = qevref.REF / "eval-data"
OUT_BIN = qevref.GOLDEN_BIN / "eval"
OUT_SET = qevref.REPO / "tests" / "golden" / "dev" / "eval" / "set.json"

SEED = 20261113
N_PER_SOURCE = 128
# The NVFP4 card's documented trained/request limit (16,384 rendered tokens per evaluated question: one
# independent forward of prefix + that question). qev's own limits (32768 per branch, 65536 per request) are
# looser; both are applied, and how many LongBench-v2 items only qev's limits would admit is recorded.
CARD_BRANCH_LIMIT = 16384
QEV_BRANCH_LIMIT = 32768
QEV_REQUEST_LIMIT = 65536

SOURCE_FILES = {
    "banking77": "banking77/data/test-00000-of-00001.parquet",
    "ledgar": "lex_glue/ledgar/test-00000-of-00001.parquet",
    "longbench_v2": "longbench_v2/data.json",
}
SOURCE_REVISIONS = {
    "banking77": "PolyAI/banking77@1846984189f2bc0ac51c6fc3a600fa9e1714e0c6 (test split)",
    "ledgar": "coastalcph/lex_glue@c23fdff1a6bf74e0e1a71cb86f1e781d37da888c (ledgar, test split)",
    "longbench_v2": "zai-org/LongBench-v2 (= THUDM/LongBench-v2)@2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
}

LABELED_QID = "task"


def support_questions(source: str) -> dict:
    """Two unlabeled support questions per record (one noul, one score), as in the card's 2 x 278 layout. They
    carry no labels: they only add branches (and probability vectors) to compare between paths."""
    if source == "banking77":
        return {
            "is_problem": {"type": "noul", "instructions": "Is the customer reporting a problem (something went "
                           "wrong, is missing or did not arrive), rather than asking for information?"},
            "urgency": {"type": "score", "instructions": "How urgent is this message for the bank?",
                        "criteria": ["Not urgent: general information or a routine request",
                                     "Should be handled today",
                                     "Urgent: the customer's money or account access is at risk now"]},
        }
    if source == "ledgar":
        return {
            "has_quantity": {"type": "noul", "instructions": "Does this provision state a specific amount of "
                             "money, number, date or time period?"},
            "boilerplate": {"type": "score", "instructions": "How standard (boilerplate) is this provision?",
                            "criteria": ["Specific to this agreement", "Partly standard",
                                         "Standard boilerplate found in most agreements"]},
        }
    if source == "longbench_v2":
        return {
            "single_document": {"type": "noul", "instructions": "Does the state consist of a single document "
                                "(rather than several documents, a dialogue history or a code repository)?"},
            "technicality": {"type": "score", "instructions": "How technical is the content of the state?",
                             "criteria": ["Not technical", "Somewhat technical", "Highly technical"]},
        }
    raise KeyError(source)


def labeled_request(source: str, state: str, instructions: str, options: dict) -> dict:
    # "jev-latest" is qev's schema default and an alias every qev server (and sysone) accepts, whatever the served
    # model name; score.py --model overrides it.
    return {"model": "jev-latest", "state": state,
            "questions": {LABELED_QID: {"type": "choice", "instructions": instructions, "criteria": options},
                          **support_questions(source)}}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def classification_records(source: str, labels: list[str], instructions: str) -> list[dict]:
    rows = load_jsonl(DATA / f"{source}_test.jsonl")
    rng = random.Random(f"{SEED}:{source}")
    picked = sorted(rng.sample(range(len(rows)), N_PER_SOURCE))
    out = []
    # Option names are the dataset's label names verbatim, in label-id order, with no description (rendered
    # "A. activate_my_card"); the state is the raw text.
    options = {name: None for name in labels}
    assert len(options) == len(labels), "duplicate label names"
    for i in picked:
        row = rows[i]
        assert row["index"] == i
        out.append({"id": f"{source}-{i:05d}", "source": source, "source_index": i,
                    "gold": {LABELED_QID: labels[row["label"]]},
                    "request": labeled_request(source, row["text"], instructions, dict(options))})
    return out


def longbench_candidates() -> list[dict]:
    items = json.loads((DATA / SOURCE_FILES["longbench_v2"]).read_text(encoding="utf-8"))
    out = []
    for it in items:
        choices = [it[f"choice_{c}"] for c in "ABCD"]
        # The four choices become the option names (rendered "A. <choice text>", so slot letter = original
        # letter); a record whose choice texts are not distinct non-empty strings cannot be a choice question.
        ok = all(isinstance(c, str) and c.strip() for c in choices) and len({c for c in choices}) == 4
        out.append({"id": f"longbench_v2-{it['_id']}", "source": "longbench_v2", "source_index": it["_id"],
                    "skip": None if ok else "choice texts not distinct non-empty strings",
                    "meta": {k: it[k] for k in ("domain", "sub_domain", "difficulty", "length")},
                    "gold": {LABELED_QID: choices["ABCD".index(it["answer"])]} if ok else None,
                    "request": labeled_request("longbench_v2", it["context"], it["question"],
                                               {c: None for c in choices}) if ok else None,
                    "context_chars": len(it["context"])})
    return out


def plan(shim, request: dict) -> dict:
    """Token counts the reference computes for this request (qev's own decide() planning, no weights)."""
    from qev.schema import SystemOneRequest

    req = SystemOneRequest.model_validate(request)
    try:
        shim.decide(req, debias=1)
    except ValueError:
        pass
    last = shim.last
    return {"prefix_tokens": len(last["prefix_ids"]), "longest": last["longest"], "total": last["total"],
            "error": last.get("error")}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_BIN / "requests.jsonl")
    args = ap.parse_args()

    from qev_shim import ShimEngine

    # A shim with the loosest limits, so decide() always reaches planning; eligibility is applied here.
    shim = ShimEngine(str(qevref.BF16_MODEL), max_branch_tokens=10**9, max_request_tokens=10**9)
    tok = shim.tokenizer
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    records = (classification_records("banking77", labels["banking77"],
                                      "Which intent does the customer's message express?")
               + classification_records("ledgar", labels["ledgar"],
                                        "Which heading best describes this contract provision?"))

    lb = longbench_candidates()
    lb_stats = {"items": len(lb), "skipped_bad_choices": 0, "context_over_limit": 0, "qev_eligible": 0,
                "card_eligible": 0}
    for c in lb:
        if c["skip"]:
            lb_stats["skipped_bad_choices"] += 1
            continue
        # cheap necessary condition first: the context alone already exceeds the qev branch limit
        n_ctx = len(tok(c["request"]["state"], add_special_tokens=False)["input_ids"])
        if n_ctx > QEV_BRANCH_LIMIT:
            lb_stats["context_over_limit"] += 1
            continue
        c["plan"] = plan(shim, c["request"])
        if c["plan"]["longest"] <= QEV_BRANCH_LIMIT and c["plan"]["total"] <= QEV_REQUEST_LIMIT:
            lb_stats["qev_eligible"] += 1
            if c["plan"]["longest"] <= CARD_BRANCH_LIMIT:
                lb_stats["card_eligible"] += 1
                records.append(c)
    lb_stats["qev_only_eligible_ids"] = sorted(
        c["source_index"] for c in lb if c.get("plan") and CARD_BRANCH_LIMIT < c["plan"]["longest"] <= QEV_BRANCH_LIMIT
        and c["plan"]["total"] <= QEV_REQUEST_LIMIT)

    lines, manifest = [], []
    for r in records:
        p = r.get("plan") or plan(shim, r["request"])
        assert p["error"] is None, (r["id"], p)
        assert p["longest"] <= CARD_BRANCH_LIMIT and p["total"] <= QEV_REQUEST_LIMIT, (r["id"], p)
        line = json.dumps({"id": r["id"], "source": r["source"], "debias": 1, "gold": r["gold"],
                           "request": r["request"]}, ensure_ascii=False)
        lines.append(line)
        m = {"id": r["id"], "source": r["source"], "source_index": r["source_index"], "gold": r["gold"],
             "n_options": len(r["request"]["questions"][LABELED_QID]["criteria"]),
             "prefix_tokens": p["prefix_tokens"], "longest": p["longest"], "total": p["total"],
             "request_sha256": sha256_bytes(line.encode())}
        if "meta" in r:
            m["meta"] = r["meta"]
        manifest.append(m)
    body = ("\n".join(lines) + "\n").encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)

    counts = {}
    for m in manifest:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    out = {
        "schema": 1,
        "seed": SEED,
        "labeled_qid": LABELED_QID,
        "records": len(manifest),
        "labeled_tasks": len(manifest),
        "support_questions": 2 * len(manifest),
        "per_source": counts,
        "limits": {"card_branch_limit": CARD_BRANCH_LIMIT, "qev_branch_limit": QEV_BRANCH_LIMIT,
                   "qev_request_limit": QEV_REQUEST_LIMIT},
        "longbench_v2_eligibility": lb_stats,
        "requests_file": str(args.out),
        "requests_sha256": sha256_bytes(body),
        "sources": {k: {"revision": SOURCE_REVISIONS[k], "file": v, "sha256": qevref.sha256_file(DATA / v)}
                    for k, v in SOURCE_FILES.items()},
        "builder_sha256": qevref.sha256_file(Path(__file__)),
        "tokenizer_sha256": qevref.sha256_file(qevref.BF16_MODEL / "tokenizer.json"),
        "manifest": manifest,
    }
    OUT_SET.parent.mkdir(parents=True, exist_ok=True)
    OUT_SET.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tot = [m["longest"] for m in manifest]
    print(f"{len(manifest)} records {counts}; longest branch max {max(tot)}; longbench {lb_stats | {'qev_only_eligible_ids': len(lb_stats['qev_only_eligible_ids'])}}")
    print(f"wrote {args.out} ({len(body)} bytes, sha256 {out['requests_sha256'][:16]}) and {OUT_SET}")


if __name__ == "__main__":
    main()
