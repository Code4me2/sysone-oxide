"""Noise statistics between the W1b golden runs, and the measured numbers behind tests/tolerances.toml (§9.2).

Reads tests/golden/dev/model/runs/*.json (and, for per-layer hidden states, the safetensors those runs point to),
compares runs pairwise branch by branch, and writes tests/golden/dev/model/noise_report.json. Every statistic is
reported as p50 / p99 / max (DESIGN §9.2: "always report the measured distribution").

Per pair of runs (a, b), over the branches both ran (same request, same branch index, identical token ids):
  dp_option      |p_a - p_b| for every option of every branch (softmax of the slot logits, T = 1)
  dp_branch_max  max over a branch's options of |p_a - p_b|
  dp_answer      |p_a - p_b| on the final answer probabilities of the response JSON (after debias averaging)
  dlogit         |l_a - l_b| on the raw slot logits
  top1 flips     branches whose argmax differs, with b's top-1 margin at each flip
Per-layer hidden states (pairs whose runs both dumped them): rel = ||h_a - h_b||_2 / ||h_b||_2 per row, b = truth.

    venv/bin/python -I reference/noise_stats.py              # report only
    venv/bin/python -I reference/noise_stats.py --write-tolerances
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qevref  # noqa: E402

MODEL_DIR = qevref.REPO / "tests" / "golden" / "dev" / "model"
RUNS = MODEL_DIR / "runs"
TOLERANCES = qevref.REPO / "tests" / "tolerances.toml"

# (name, run a, run b): b is the reference side ("truth") of the pair.
PAIRS = [
    ("bf16_vs_fp32.batched", "bf16_batched", "fp32_batched"),
    ("bf16_vs_fp32.sequential", "bf16_sequential", "fp32_sequential"),
    ("bf16_vs_fp32.per_question", "bf16_perq", "fp32_sequential"),
    ("bf16.batched_vs_sequential", "bf16_batched", "bf16_sequential"),
    ("fp32.batched_vs_sequential", "fp32_batched", "fp32_sequential"),
    ("bf16.per_question_vs_sequential", "bf16_perq", "bf16_sequential"),
    ("bf16.per_question_vs_batched", "bf16_perq", "bf16_batched"),
    ("nvfp4.per_question_vs_batched", "nvfp4_batched", "nvfp4_perq_r1"),
    ("nvfp4.per_question_vs_sequential", "nvfp4_sequential", "nvfp4_perq_r1"),
    ("nvfp4.batched_vs_sequential", "nvfp4_batched", "nvfp4_sequential"),
    ("nvfp4.cross_process.r1_r2", "nvfp4_perq_r2", "nvfp4_perq_r1"),
    ("nvfp4.cross_process.r1_r3", "nvfp4_perq_r3", "nvfp4_perq_r1"),
    ("nvfp4.cross_process.r2_r3", "nvfp4_perq_r3", "nvfp4_perq_r2"),
    ("nvfp4_vs_bf16.per_question", "nvfp4_perq_r1", "bf16_perq"),
    ("nvfp4_vs_bf16.sequential", "nvfp4_sequential", "bf16_sequential"),
    ("nvfp4_vs_fp32.per_question", "nvfp4_perq_r1", "fp32_sequential"),
]
# pooled groups: the envelopes the tolerance rules use
GROUPS = {
    "bf16_vs_fp32": ["bf16_vs_fp32.batched", "bf16_vs_fp32.sequential"],
    "bf16_fork": ["bf16.batched_vs_sequential"],
    "bf16_path": ["bf16.per_question_vs_sequential", "bf16.per_question_vs_batched"],
    "nvfp4_cross_process": ["nvfp4.cross_process.r1_r2", "nvfp4.cross_process.r1_r3", "nvfp4.cross_process.r2_r3"],
    "nvfp4_perq_vs_shared_prefix": ["nvfp4.per_question_vs_batched", "nvfp4.per_question_vs_sequential"],
    "nvfp4_vs_bf16": ["nvfp4_vs_bf16.per_question", "nvfp4_vs_bf16.sequential"],
}
# hidden-state pairs (a, b): b = truth
HIDDEN_PAIRS = [
    ("bf16_vs_fp32", "bf16_batched", "fp32_batched"),
    ("bf16_rerun", "bf16_batched", "bf16_sequential"),
    ("fp32_rerun", "fp32_batched", "fp32_sequential"),
    ("bf16_served_vs_per_question", "bf16_sequential", "bf16_perq_hidden"),
    ("nvfp4_served_vs_per_question", "nvfp4_sequential", "nvfp4_perq_r1"),
    ("nvfp4_vs_bf16", "nvfp4_sequential", "bf16_sequential"),
    ("nvfp4_vs_fp32", "nvfp4_sequential", "fp32_sequential"),
]


def quantiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)

    def q(p):  # nearest-rank
        return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]

    return {"n": len(s), "p50": q(0.50), "p99": q(0.99), "max": s[-1], "mean": sum(s) / len(s)}


def load_runs() -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))}


def answer_probs(resp: dict) -> dict:
    out = {}
    for qid, a in resp.get("answers", {}).items():
        if a["type"] == "noul":
            out[(qid, "yes")] = a["noul"]
        else:
            for k, v in a["probabilities"].items():
                out[(qid, k)] = v
    return out


def compare(a: dict, b: dict) -> dict:
    ra = {r["id"]: r for r in a["results"]}
    rb = {r["id"]: r for r in b["results"]}
    dp, dbr, dl, dans = [], [], [], []
    flips, mism, n_br, n_req = [], [], 0, 0
    for rid in sorted(ra.keys() & rb.keys()):
        x, y = ra[rid], rb[rid]
        if x["status"] != y["status"]:
            mism.append(f"{rid}: status {x['status']} vs {y['status']}")
            continue
        if "branches" not in x or "branches" not in y:
            continue
        if x["prefix_sha256"] != y["prefix_sha256"] or len(x["branches"]) != len(y["branches"]):
            mism.append(f"{rid}: prefix or branch count differs")
            continue
        n_req += 1
        for i, (u, v) in enumerate(zip(x["branches"], y["branches"])):
            if u["suffix_sha256"] != v["suffix_sha256"] or u["n_slots"] != v["n_slots"]:
                mism.append(f"{rid}#{i}: suffix differs")
                continue
            n_br += 1
            d = [abs(p - q) for p, q in zip(u["probs"], v["probs"])]
            dp += d
            dbr.append(max(d))
            dl += [abs(p - q) for p, q in zip(u["logits"], v["logits"])]
            ia = max(range(len(u["probs"])), key=u["probs"].__getitem__)
            ib = max(range(len(v["probs"])), key=v["probs"].__getitem__)
            if ia != ib:
                top = sorted(v["probs"], reverse=True)
                flips.append({"id": rid, "branch": i, "ref_margin": top[0] - top[1], "dp_max": max(d)})
        pa, pb = answer_probs(x["response"]), answer_probs(y["response"])
        dans += [abs(pa[k] - pb[k]) for k in sorted(pa.keys() & pb.keys())]
    return {"requests": n_req, "branches": n_br, "dp_option": quantiles(dp), "dp_branch_max": quantiles(dbr),
            "dp_answer": quantiles(dans), "dlogit": quantiles(dl), "top1_flips": len(flips),
            "top1_flip_max_ref_margin": max((f["ref_margin"] for f in flips), default=0.0),
            "flips": flips[:20], "mismatches": mism[:20], "n_mismatches": len(mism),
            "_raw": {"dp_option": dp, "dp_branch_max": dbr, "dp_answer": dans, "dlogit": dl}}


def hidden_compare(a: dict, b: dict) -> dict | None:
    import torch
    from safetensors.torch import load_file

    ha = {r["id"]: r["hidden"]["file"] for r in a["results"] if "hidden" in r}
    hb = {r["id"]: r["hidden"]["file"] for r in b["results"] if "hidden" in r}
    common = sorted(ha.keys() & hb.keys())
    if not common:
        return None
    rel: dict[str, dict[str, list]] = {}
    for rid in common:
        ta, tb = load_file(str(qevref.GOLDEN_BIN / ha[rid])), load_file(str(qevref.GOLDEN_BIN / hb[rid]))
        for key, tv in tb.items():
            if not key.startswith(("layer", "norm")):
                continue
            level, where = key.split("/")
            r = (ta[key] - tv).norm(dim=-1) / tv.norm(dim=-1).clamp_min(1e-30)
            rel.setdefault(level, {"prefix": [], "answer": []})[where] += r.tolist()
    order = [lv for lv in qevref.LEVELS if lv in rel]
    return {"requests": common,
            "rel_err": {lv: {"prefix": quantiles(rel[lv]["prefix"]), "answer": quantiles(rel[lv]["answer"]),
                             "all": quantiles(rel[lv]["prefix"] + rel[lv]["answer"])} for lv in order}}


def pooled(results: dict, names: list[str]) -> dict | None:
    have = [n for n in names if n in results]
    if not have:
        return None
    out = {"pairs": have}
    for key in ("dp_option", "dp_branch_max", "dp_answer", "dlogit"):
        out[key] = quantiles([x for n in have for x in results[n]["_raw"][key]])
    out["top1_flips"] = sum(results[n]["top1_flips"] for n in have)
    out["top1_flip_max_ref_margin"] = max(results[n]["top1_flip_max_ref_margin"] for n in have)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-tolerances", action="store_true")
    args = ap.parse_args()
    runs = load_runs()
    pairs = {}
    for name, a, b in PAIRS:
        if a in runs and b in runs:
            pairs[name] = {"a": a, "b": b, **compare(runs[a], runs[b])}
    groups = {g: pooled(pairs, names) for g, names in GROUPS.items()}
    for p in pairs.values():
        del p["_raw"]
    hidden = {}
    for name, a, b in HIDDEN_PAIRS:
        if a in runs and b in runs and (h := hidden_compare(runs[a], runs[b])):
            hidden[name] = {"a": a, "b": b, **h}
    report = {"runs": {k: {"config": v["config"], "requests": len(v["results"]), "dropped": v["dropped"]}
                       for k, v in runs.items()},
              "groups": groups, "pairs": pairs, "hidden": hidden}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "noise_report.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, p in pairs.items():
        o, bm = p["dp_option"], p["dp_branch_max"]
        print(f"{name:38s} br={p['branches']:4d} dp_opt p50={o.get('p50', 0):.2e} p99={o.get('p99', 0):.2e} "
              f"max={o.get('max', 0):.2e} | br_max p99={bm.get('p99', 0):.2e} | flips={p['top1_flips']} "
              f"(max margin {p['top1_flip_max_ref_margin']:.3f}) mism={p['n_mismatches']}")
    for name, h in hidden.items():
        print(name, {lv: f"{v['all']['p50']:.2e}/{v['all']['p99']:.2e}/{v['all']['max']:.2e}"
                     for lv, v in h["rel_err"].items()})
    if args.write_tolerances:
        import tolerances

        old = TOLERANCES.read_text() if TOLERANCES.exists() else ""
        TOLERANCES.write_text(tolerances.keep_generated_blocks(old, tolerances.render(report)))
        print(f"wrote {TOLERANCES}")


if __name__ == "__main__":
    main()
