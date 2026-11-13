"""Statistics over eval-set runs and the statistical gates (DESIGN §9.2). Stdlib only.

Per run (labeled question of each record):
  accuracy per source and overall, with a record-bootstrap 95% CI; multi-class Brier score; ECE (top-1
  confidence, 10 equal-width bins).
Per pair of runs (a, b), over the records both answered:
  top-1 agreement over ALL questions (labeled + support) and over the labeled ones; |dp| = |p_a - p_b| per option
  of every question (answer-level vectors, see evalset.py): mean, p50, p99 (nearest rank), max; accuracy
  difference a - b with a paired record-bootstrap 95% CI; Brier difference a - b with a paired record-bootstrap
  95% CI (labeled questions).

THE GATE (one rule for both tiers; thresholds live in tests/tolerances.toml [eval.*]):
  The reference paths R = {r_1..r_k} of a tier are compared pairwise; the ENVELOPE of each metric is its worst
  value over those k(k-1)/2 pairs:
    disagree   fraction of ALL questions whose top-1 differs while the reference side's top-1 margin exceeds
               disagree_margin (both orders; max over pairs). disagree_margin = 0 for nvfp4 (every question
               counts) and [e2e.bf16].top1_margin (0.02, DESIGN §9.2 end-to-end row: "top choice agrees wherever
               the reference margin is > 0.02") for bf16, so near-tie flips the design allows never count
    dp_mean    mean |dp|                                         (max over pairs)
    dp_p99     p99 |dp|                                          (max over pairs)
    acc_drop   max(0, -lo) of the paired-bootstrap 95% CI of acc(r_i) - acc(r_j), both orders (max over pairs):
               the largest accuracy drop the reference's own path noise cannot rule out
    brier_rise max(0, hi, -lo) of the paired-bootstrap 95% CI of Brier(r_i) - Brier(r_j) (max over pairs):
               the largest calibration loss the reference's own path noise cannot rule out
  threshold = max(FACTOR x envelope, floor) with FACTOR = tolerances.toml `factor` (1.5), floors only where a
  discrete count makes the envelope too coarse:
    floor_disagree = 2 questions / all questions;
    floor_acc_drop = on the CI scale: -(lower bound of the paired-bootstrap 95% CI) of a candidate that differs
      from the reference by ONE labeled loss, or by one loss plus one gain (net zero), whichever is lower. It
      is computed exactly (multinomial resampling counts), not by simulation: 3/278 on the full set.
  Limits are written to tolerances.toml rounded UP (6 significant digits), never below the rule.
  A candidate C PASSES iff for EVERY reference path r in R:
    disagree(C, r) <= disagree (margin of r);  dp_mean(C, r) <= dp_mean;  dp_p99(C, r) <= dp_p99;
    lower bound of the paired-bootstrap 95% CI of acc(C) - acc(r) >= -acc_drop      (non-inferiority);
    upper bound of the paired-bootstrap 95% CI of Brier(C) - Brier(r) <= brier_rise.
  (The first version gated the point Brier difference on bf16 only; its leave-one-out validation failed because
  two bf16 paths had identical point Briers, an envelope of 0. The CI form is robust to that and applies to both
  tiers.)
  Tiers: nvfp4 (R = the NVFP4 QDQ paths; C = sysone NVFP4) and bf16 (R = the bf16 qev paths plus the fp32 qev
  path, C = sysone bf16). DESIGN §9.2's binding rule bounds sysone's error by the envelope the bf16 reference shows
  vs fp32 truth AND vs its own batched/sequential paths, so the fp32 path is a reference path of the bf16 tier: the
  bf16~fp32 pairs carry the "vs truth" envelope, and the candidate is also gated against fp32 directly.
  NVFP4 vs bf16 is reported as quality information only.

Bootstrap: B = 10,000 resamples of the records (labeled tasks), seeded (SEED), percentile interval. Paired:
the same resampled records for both runs.

    python3 -I reference/eval/stats.py                      # report (prints, writes tests/golden/dev/eval/report.json)
    python3 -I reference/eval/stats.py --write-tolerances   # also rewrites the [eval.*] block of tolerances.toml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalset  # noqa: E402

SEED = 20261113
B = 10_000
REPO_TOL = evalset.REPO / "tests" / "tolerances.toml"


def _top_level_factor() -> float:
    for raw in REPO_TOL.read_text().split("\n"):
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            break
        if line.startswith("factor") and "=" in line:
            return float(line.split("=", 1)[1])
    raise SystemExit("tests/tolerances.toml lacks the top-level `factor`")


FACTOR = _top_level_factor()
TIERS = {
    "nvfp4": ["nvfp4_perq", "nvfp4_perq_bN", "nvfp4_batched", "nvfp4_sequential"],
    "bf16": ["bf16_batched", "bf16_sequential", "bf16_perq", "fp32_batched"],
}
# the truth path of a tier: part of the envelope, never left out in leave-one-out validation
TRUTH = {"bf16": "fp32_batched"}


def _table_value(table: str, key: str) -> float:
    cur = None
    for raw in REPO_TOL.read_text().split("\n"):
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            cur = line.strip("[]").strip()
        elif cur == table and "=" in line and line.split("=", 1)[0].strip() == key:
            return float(line.split("=", 1)[1])
    raise SystemExit(f"tests/tolerances.toml lacks [{table}] {key}")


# top-1 margin of the reference at or below which a top-1 change is allowed (DESIGN §9.2 end-to-end row)
MARGIN = {"nvfp4": 0.0, "bf16": _table_value("e2e.bf16", "top1_margin")}
SIG = 6  # significant digits of the written limits (rounded up)
REPORT = evalset.REPO / "tests" / "golden" / "dev" / "eval" / "report.json"
TOLERANCES = evalset.REPO / "tests" / "tolerances.toml"
BEGIN = "# >>> [eval] generated by reference/eval/stats.py --write-tolerances (do not edit by hand) >>>"
END = "# <<< [eval] end <<<"


def q_nearest(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]


def argmax(v: list[float]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def answered(run: dict) -> dict[str, dict]:
    return {r["id"]: r["answers"] for r in run["results"] if r.get("status") == 200 and "answers" in r}


def labeled_rows(run: dict) -> list[dict]:
    """Per labeled record: correct (0/1), top-1 confidence, Brier."""
    st = evalset.load_set()
    lq = st["labeled_qid"]
    ans = answered(run)
    rows = []
    for m in st["manifest"]:
        if m["id"] not in ans:
            continue
        req = evalset.request_by_id(m["id"])["request"]
        names = evalset.option_names(req["questions"][lq])
        v = ans[m["id"]][lq]
        gold = names.index(m["gold"][lq])
        top = argmax(v)
        brier = sum((p - (1.0 if i == gold else 0.0)) ** 2 for i, p in enumerate(v))
        rows.append({"id": m["id"], "source": m["source"], "correct": int(top == gold), "conf": v[top],
                     "brier": brier})
    return rows


def boot_ci(values: list[float], rng: random.Random) -> list[float]:
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [means[int(0.025 * B)], means[int(0.975 * B) - 1]]


def ece(rows: list[dict], bins: int = 10) -> float:
    tot, n = 0.0, len(rows)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if (lo < r["conf"] <= hi) or (b == 0 and r["conf"] == 0)]
        if sel:
            tot += len(sel) / n * abs(sum(r["correct"] for r in sel) / len(sel) - sum(r["conf"] for r in sel) / len(sel))
    return tot


def run_summary(run: dict) -> dict:
    rows = labeled_rows(run)
    rng = random.Random(f"{SEED}:{run['name']}")
    out = {"records": len(rows)}
    groups = {"overall": rows}
    for s in evalset.load_set()["per_source"]:
        groups[s] = [r for r in rows if r["source"] == s]
    for g, rs in groups.items():
        if not rs:
            continue
        acc = [r["correct"] for r in rs]
        out[g] = {"n": len(rs), "correct": sum(acc), "accuracy": sum(acc) / len(rs), "ci95": boot_ci(acc, rng),
                  "brier": sum(r["brier"] for r in rs) / len(rs), "ece": ece(rs)}
    return out


def margin(v: list[float]) -> float:
    s = sorted(v, reverse=True)
    return s[0] - s[1] if len(s) > 1 else 1.0


def compare(a: dict, b: dict, top1_margin: float = 0.0) -> dict:
    """a vs b over the records both answered. b is the reference: `disagree` counts the questions whose top-1
    differs while b's top-1 margin exceeds `top1_margin` (every question when it is 0); `disagree_rev` the same
    with a as the reference; `disagree_all` every top-1 difference."""
    lq = evalset.load_set()["labeled_qid"]
    aa, bb = answered(a), answered(b)
    ids = [m["id"] for m in evalset.load_set()["manifest"] if m["id"] in aa and m["id"] in bb]
    dp, agree, n_q, agree_l, dis, dis_rev = [], 0, 0, 0, 0, 0
    for rid in ids:
        for qid, va in aa[rid].items():
            vb = bb[rid][qid]
            assert len(va) == len(vb), (rid, qid)
            dp.extend(abs(x - y) for x, y in zip(va, vb))
            same = argmax(va) == argmax(vb)
            agree += same
            if not same:
                dis += top1_margin <= 0 or margin(vb) > top1_margin
                dis_rev += top1_margin <= 0 or margin(va) > top1_margin
            n_q += 1
            if qid == lq:
                agree_l += same
    ra = {r["id"]: r for r in labeled_rows(a)}
    rb = {r["id"]: r for r in labeled_rows(b)}
    diffs = [ra[i]["correct"] - rb[i]["correct"] for i in ids]
    rng = random.Random(f"{SEED}:{a['name']}:{b['name']}")
    rng_b = random.Random(f"{SEED}:brier:{a['name']}:{b['name']}")
    bd = [ra[i]["brier"] - rb[i]["brier"] for i in ids]
    n = len(ids)
    return {
        "records": n, "questions": n_q,
        "top1_margin": top1_margin, "agreement": agree / n_q,
        "disagree": dis / n_q, "disagree_count": dis, "disagree_rev": dis_rev / n_q,
        "disagree_all": (n_q - agree) / n_q, "disagree_all_count": n_q - agree,
        "agreement_labeled": agree_l / n, "disagree_labeled_count": n - agree_l,
        "dp_mean": sum(dp) / len(dp), "dp_p50": q_nearest(dp, 0.5), "dp_p99": q_nearest(dp, 0.99), "dp_max": max(dp),
        "acc_diff": sum(diffs) / n, "acc_diff_ci95": boot_ci(diffs, rng),
        "gains": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
        "brier_diff": sum(bd) / n, "brier_diff_ci95": boot_ci(bd, rng_b),
    }


def envelope(pairs: dict[str, dict]) -> dict:
    """Worst value of each gated metric over the reference pairs, with the pair it came from."""
    env = {}

    def worst(key, val):
        for name, c in pairs.items():
            v = val(c)
            if key not in env or v > env[key][0]:
                env[key] = (v, name)

    worst("disagree", lambda c: max(c["disagree"], c["disagree_rev"]))
    worst("dp_mean", lambda c: c["dp_mean"])
    worst("dp_p99", lambda c: c["dp_p99"])
    # both orders: the CI of b - a is the negation of a - b's (same resamples)
    worst("acc_drop", lambda c: max(0.0, -c["acc_diff_ci95"][0], c["acc_diff_ci95"][1]))
    worst("brier_rise", lambda c: max(0.0, -c["brier_diff_ci95"][0], c["brier_diff_ci95"][1]))
    return env


def _boot_lo_quantile(pmf: dict[int, float]) -> int:
    """The value boot_ci reports as the lower bound (sorted resample means at index int(0.025 B)) in the limit of
    many resamples: the smallest x with P(X <= x) > 0.025."""
    acc = 0.0
    for x in sorted(pmf):
        acc += pmf[x]
        if acc > 0.025:
            return x
    raise AssertionError("pmf does not sum to 1")


def _paired_count_pmf(n: int, gains: int, losses: int) -> dict[int, float]:
    """Exact distribution of sum(diffs) under the paired record bootstrap when `gains` records have diff +1,
    `losses` have -1 and the rest 0: the n resampling counts are multinomial(n; 1/n each)."""
    k = gains + losses
    out: dict[int, float] = {}
    # counts of the k nonzero records: c_1..c_k with sum s <= n; enumerate per record up to a cap
    cap = 20  # P(one record drawn > 20 times) < 1e-18
    lp, lq_ = math.log(1.0 / n), math.log1p(-k / n)
    lf = math.lgamma

    def rec(i, s, val, logw):
        if i == k:
            p = math.exp(lf(n + 1) - lf(n - s + 1) + logw + s * lp + (n - s) * lq_)
            out[val] = out.get(val, 0.0) + p
            return
        for c in range(0, cap + 1):
            if s + c > n:
                break
            rec(i + 1, s + c, val + (c if i < gains else -c), logw - lf(c + 1))
    rec(0, 0, 0, 0.0)
    return out


def acc_floor(n_records: int) -> dict:
    """floor_acc_drop on the CI scale: -(lower CI bound) of a candidate one labeled record worse than the
    reference (one loss) or net zero (one loss + one gain), whichever is lower. Also the tail probabilities
    around the bound, so it can be checked that a 10,000-resample bootstrap lands on it."""
    worst, detail = 0.0, {}
    for g, l in ((0, 1), (1, 1)):
        pmf = _paired_count_pmf(n_records, g, l)
        q = _boot_lo_quantile(pmf)
        cdf = lambda x: sum(p for v, p in pmf.items() if v <= x)  # noqa: E731
        detail[f"{g}gain_{l}loss"] = {"ci_lo": q / n_records, "P(X<=lo)": cdf(q), "P(X<lo)": cdf(q - 1)}
        worst = max(worst, -q / n_records)
    return {"floor": worst, "scenarios": detail}


def ceil_sig(x: float, sig: int = SIG) -> float:
    """x rounded UP to `sig` significant digits (a written limit is never below the rule)."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - sig + 1
    y = math.ceil(round(x / 10.0 ** e, 6)) * 10.0 ** e
    y = float(f"{y:.{sig}g}")
    while y < x:  # guard against the decimal round trip
        y = float(f"{y + 10.0 ** e:.{sig}g}")
    return y


def thresholds(env: dict, n_questions: int, n_records: int, tier: str) -> dict:
    """threshold = max(FACTOR x measured, floor), rounded up to SIG significant digits."""
    floors = {"disagree": 2.0 / n_questions, "acc_drop": acc_floor(n_records)["floor"]}
    out = {}
    for k, (v, src) in env.items():
        fl = floors.get(k, 0.0)
        out[k] = {"limit": ceil_sig(max(FACTOR * v, fl)), "measured": v, "source": src,
                  "floor": fl if k in floors else None}
    return out


def gate(candidate: dict, refs: dict[str, dict], limits: dict) -> dict:
    """Apply the gate: candidate vs every reference path. `limits` maps metric -> limit (and disagree_margin)."""
    checks, ok = [], True
    for name, ref in refs.items():
        c = compare(candidate, ref, limits.get("disagree_margin", 0.0))
        items = [("disagree", c["disagree"], limits["disagree"]),
                 ("dp_mean", c["dp_mean"], limits["dp_mean"]),
                 ("dp_p99", c["dp_p99"], limits["dp_p99"]),
                 ("acc_ci_lo", c["acc_diff_ci95"][0], -limits["acc_drop"])]
        items.append(("brier_ci_hi", c["brier_diff_ci95"][1], limits["brier_rise"]))
        for metric, val, lim in items:
            passed = val >= lim if metric == "acc_ci_lo" else val <= lim
            ok &= passed
            checks.append({"ref": name, "metric": metric, "value": val, "limit": lim, "pass": passed})
        checks.append({"ref": name, "metric": "info", "value": {k: c[k] for k in (
            "records", "agreement_labeled", "disagree_all_count", "dp_max", "acc_diff", "acc_diff_ci95", "gains",
            "losses", "brier_diff")}})
    return {"pass": ok, "checks": checks}


def load_limits(tier: str) -> dict:
    """Thresholds from tests/tolerances.toml [eval.<tier>] (the committed source of truth)."""
    vals = {}
    table = None
    for raw in TOLERANCES.read_text().split("\n"):
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            table = line.strip("[]").strip()
        elif "=" in line and table == f"eval.{tier}":
            k, v = (x.strip() for x in line.split("=", 1))
            if not v.startswith('"') and v not in ("true", "false"):
                vals[k] = float(v)
    keys = ["disagree", "dp_mean", "dp_p99", "acc_drop", "brier_rise", "disagree_margin"]
    missing = [k for k in keys if k not in vals]
    if missing:
        raise SystemExit(f"tolerances.toml [eval.{tier}] lacks {missing}; run stats.py --write-tolerances")
    return {k: vals[k] for k in keys}


def tier_limits(tier: str, th: dict) -> dict:
    return {**{k: d["limit"] for k, d in th.items()}, "disagree_margin": MARGIN[tier]}


def tier_pairs(runs: dict, paths: list[str], tier: str) -> dict:
    return {f"{a}~{b}": compare(runs[a], runs[b], MARGIN[tier]) for i, a in enumerate(paths) for b in paths[i + 1:]}


def fmt_num(x: float) -> str:
    """Full precision (repr), so the Rust consistency test can check limit >= max(factor x measured, floor)
    exactly."""
    return repr(float(x))


def render_tolerances(report: dict) -> str:
    st = evalset.load_set()
    L = [BEGIN,
         "# Statistical end-to-end gates on the labeled eval set (DESIGN §9.2; reference/eval/README.md).",
         f"# Set: tests/golden/dev/eval/set.json ({st['records']} labeled records = {st['records']} labeled + "
         f"{st['support_questions']} support questions; requests sha256 {st['requests_sha256'][:16]}).",
         "# Envelope = worst value over all pairs of the tier's reference paths; limit = max(factor x measured, floor)",
         "# rounded UP to 6 significant digits (measured_ and floor_ are written at full precision).",
         "# A candidate passes iff, against EVERY reference path of the tier: disagree, dp_mean, dp_p99 <= limit;",
         "# the paired-bootstrap (B=10000, records) 95% CI lower bound of acc(candidate) - acc(ref) >= -acc_drop;",
         "# the paired-bootstrap 95% CI upper bound of Brier(candidate) - Brier(ref) <= brier_rise.",
         "# disagree counts top-1 changes over all questions where the reference's top-1 margin > disagree_margin",
         "# (nvfp4: 0 = every question; bf16: [e2e.bf16].top1_margin, DESIGN 9.2's allowance for near-tie flips).",
         "# dp is per option of every question (answer level); acc and Brier are over the labeled questions.",
         "# bf16 paths include fp32_batched: DESIGN 9.2 bounds the error by the bf16 reference's own error vs fp32.",
         "# floor_disagree = 2 questions of the set. floor_acc_drop is on the CI scale: -(CI lower bound) of a",
         "# candidate one labeled loss (or one loss + one gain) away from the reference, computed exactly."]
    for tier in ("nvfp4", "bf16"):
        t = report["tiers"].get(tier)
        if not t:
            continue
        L += ["", f"[eval.{tier}]", f'paths = "{"+".join(t["paths"])}"',
              f"disagree_margin = {fmt_num(MARGIN[tier])}"
              + ("  # = [e2e.bf16].top1_margin" if tier == "bf16" else "  # every question counts")]
        for k, d in t["thresholds"].items():
            L.append(f"{k} = {d['limit']!r}")
            L.append(f"measured_{k} = {fmt_num(d['measured'])}  # {d['source']}")
            if d["floor"] is not None:
                L.append(f"floor_{k} = {fmt_num(d['floor'])}")
    L.append(END)
    return "\n".join(L) + "\n"


def write_tolerances(report: dict) -> None:
    text = TOLERANCES.read_text()
    block = render_tolerances(report)
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END + "\n", 1)
        text = head + block + tail
    else:
        text = text.rstrip("\n") + "\n\n" + block
    TOLERANCES.write_text(text)


def tempered(run: dict, temp: float) -> dict:
    """A copy of `run` with every answer vector re-softmaxed at temperature `temp` (p ** (1/T), renormalized)."""
    def t(v):
        w = [x ** (1.0 / temp) if x > 0 else 0.0 for x in v]
        z = sum(w)
        return [x / z for x in w] if z > 0 else v
    res = [{**r, "answers": {q: t(v) for q, v in r["answers"].items()}} if "answers" in r else r
           for r in run["results"]]
    return {**run, "name": f"{run['name']}@T={temp}", "results": res}


def _flip(run: dict, picks: list[tuple[str, str]]) -> dict:
    """A copy of `run` with the top two probabilities of each (record, question) swapped."""
    want = {}
    for rid, qid in picks:
        want.setdefault(rid, set()).add(qid)
    res = []
    for r in run["results"]:
        if r["id"] in want:
            ans = {}
            for q, v in r["answers"].items():
                if q in want[r["id"]]:
                    v = list(v)
                    i, j = sorted(range(len(v)), key=lambda k: -v[k])[:2]
                    v[i], v[j] = v[j], v[i]
                ans[q] = v
            r = {**r, "answers": ans}
        res.append(r)
    return {**run, "results": res}


def minimal_flips(base: dict, refs: dict[str, dict], top1_margin: float) -> list[tuple[str, dict]]:
    """The deviations the floors and the margin allowance are meant to admit, built on `base`:
    the two smallest-margin support questions flipped (= floor_disagree); the smallest-margin correct labeled
    record flipped to wrong (one labeled loss = floor_acc_drop); that loss plus the smallest-margin wrong labeled
    record whose runner-up is gold flipped to right (net zero); every question at or below the margin allowance
    in every reference path flipped (DESIGN §9.2 allows them)."""
    st = evalset.load_set()
    lq = st["labeled_qid"]
    rows = {r["id"]: r for r in labeled_rows(base)}
    ans = answered(base)
    sup = sorted((margin(v), rid, q) for rid, a in ans.items() for q, v in a.items() if q != lq)
    lab = sorted((margin(a[lq]), rid) for rid, a in ans.items())
    loss = next(rid for _, rid in lab if rows[rid]["correct"])
    gold = {m["id"]: m["gold"][lq] for m in st["manifest"]}

    def runner_up_is_gold(rid):
        names = evalset.option_names(evalset.request_by_id(rid)["request"]["questions"][lq])
        v = ans[rid][lq]
        return sorted(range(len(v)), key=lambda k: -v[k])[1] == names.index(gold[rid])
    gain = next((rid for _, rid in lab if not rows[rid]["correct"] and runner_up_is_gold(rid)), None)
    out = [("2 support flips", _flip(base, [(r, q) for _, r, q in sup[:2]])),
           (f"1 labeled loss ({loss})", _flip(base, [(loss, lq)]))]
    if gain:
        out.append(("1 labeled loss + 1 gain", _flip(base, [(loss, lq), (gain, lq)])))
    if top1_margin > 0:
        ra = {n: answered(r) for n, r in refs.items()}
        near = [(rid, q) for rid, a in ans.items() for q in a
                if all(rid in x and margin(x[rid][q]) <= top1_margin for x in ra.values())]
        out.append((f"all {len(near)} near-tie flips (margin <= {top1_margin} in every reference)",
                    _flip(base, near)))
    return out


def fixture_rows(tier: str, cand: str, g: dict, limits: dict) -> list[dict]:
    """Per reference path: the comparison values and the failed metrics, for src/tolerances.rs to re-apply the
    committed [eval.<tier>] limits with EvalGate::failures and check it agrees with this implementation."""
    by_ref: dict[str, dict] = {}
    for c in g["checks"]:
        if c["metric"] == "info":
            continue
        row = by_ref.setdefault(c["ref"], {"tier": tier, "candidate": cand, "ref": c["ref"], "values": {},
                                           "failed": [], "limits": {k: limits[k] for k in sorted(limits)}})
        row["values"][c["metric"]] = c["value"]
        if not c["pass"]:
            row["failed"].append(c["metric"])
    return list(by_ref.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-tolerances", action="store_true")
    args = ap.parse_args()
    runs = {p.stem: evalset.load_run(p) for p in sorted(evalset.RUNS.glob("*.json"))}
    for name, r in runs.items():
        if r.get("missing"):
            print(f"WARNING: {name} is incomplete ({len(r['missing'])} records missing)")
    st = evalset.load_set()
    report = {"set": {k: st[k] for k in ("records", "per_source", "requests_sha256", "seed")},
              "bootstrap": {"B": B, "seed": SEED}, "factor": FACTOR,
              "runs": {n: run_summary(r) for n, r in runs.items()}, "tiers": {}, "cross": {}}
    print(f"{'run':22s} {'overall':>16s} " + " ".join(f"{s:>16s}" for s in st["per_source"]) + "   brier    ece")
    for n, s in report["runs"].items():
        cells = [f"{s[g]['correct']:3d}/{s[g]['n']:3d} {100 * s[g]['accuracy']:5.1f}%" if g in s else "-"
                 for g in ["overall", *st["per_source"]]]
        print(f"{n:22s} " + " ".join(f"{c:>16s}" for c in cells)
              + f"  {s['overall']['brier']:.4f} {s['overall']['ece']:.4f}")
    for tier, paths in TIERS.items():
        have = [p for p in paths if p in runs]
        if len(have) < 2:
            continue
        if tier in TRUTH and TRUTH[tier] not in have:
            print(f"WARNING: tier {tier} lacks its truth path {TRUTH[tier]}; its envelope omits the error vs truth")
        pairs = tier_pairs(runs, have, tier)
        env = envelope(pairs)
        anyc = next(iter(pairs.values()))
        report["tiers"][tier] = {"paths": have, "disagree_margin": MARGIN[tier], "pairs": pairs,
                                 "acc_floor": acc_floor(anyc["records"]),
                                 "thresholds": thresholds(env, anyc["questions"], anyc["records"], tier)}
        print(f"\n[{tier}] reference paths {have} (disagree counted above reference margin {MARGIN[tier]})")
        for n, c in pairs.items():
            lo, hi = c["acc_diff_ci95"]
            print(f"  {n:42s} agree {100 * c['agreement']:6.2f}% ({c['disagree_all_count']} q;"
                  f" {max(c['disagree'], c['disagree_rev']) * c['questions']:.0f} above margin)"
                  f" lab {c['disagree_labeled_count']}"
                  f"  |dp| mean {c['dp_mean']:.2e} p99 {c['dp_p99']:.2e} max {c['dp_max']:.3f}"
                  f"  dAcc {100 * c['acc_diff']:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}]")
        for k, d in report["tiers"][tier]["thresholds"].items():
            print(f"  limit {k:10s} = {d['limit']:.4g}  (measured {d['measured']:.4g} on {d['source']}"
                  + (f", floor {d['floor']:.4g})" if d["floor"] is not None else ")"))
    for a, b in [("nvfp4_perq", "bf16_perq"), ("nvfp4_perq", "bf16_batched"), ("nvfp4_batched", "bf16_batched"),
                 ("nvfp4_sequential", "bf16_sequential")]:
        if a in runs and b in runs:
            c = compare(runs[a], runs[b])
            report["cross"][f"{a}~{b}"] = c
            lo, hi = c["acc_diff_ci95"]
            print(f"\n[info] {a} vs {b}: agree {100 * c['agreement']:.2f}% (labeled {100 * c['agreement_labeled']:.2f}%),"
                  f" |dp| mean {c['dp_mean']:.2e} p99 {c['dp_p99']:.3f} max {c['dp_max']:.3f},"
                  f" dAcc {100 * c['acc_diff']:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}] gains {c['gains']} losses {c['losses']}")
    # Validation of the gate itself (reported, not gated):
    #  leave-one-out: rebuild a tier's envelope without path p and gate p against the rest: does a new kernel
    #    shape of the same reference pass? (it should; if not, FACTOR is too tight)
    #  cross-tier sensitivity: gate the other precision's run as a candidate: does the gate tell bf16 from NVFP4?
    report["validation"] = {"leave_one_out": {}, "cross_tier": {}}
    fixture: list[dict] = []
    for tier, t in report["tiers"].items():
        paths = t["paths"]
        if len(paths) < 3:
            continue
        for p in paths:
            if p == TRUTH.get(tier):
                continue  # the truth path is the yardstick, not a kernel shape of the reference
            rest = [x for x in paths if x != p]
            pairs = {k: v for k, v in t["pairs"].items() if p not in k.split("~")}
            anyc = next(iter(pairs.values()))
            th = thresholds(envelope(pairs), anyc["questions"], anyc["records"], tier)
            g = gate(runs[p], {r: runs[r] for r in rest}, tier_limits(tier, th))
            fails = [f"{c['ref']}:{c['metric']}" for c in g["checks"] if c["metric"] != "info" and not c["pass"]]
            report["validation"]["leave_one_out"][f"{tier}:{p}"] = {"pass": g["pass"], "failed": fails}
            print(f"[validate] leave-one-out {tier}: {p:18s} vs {rest}: {'PASS' if g['pass'] else 'FAIL ' + str(fails)}")
    for tier, cand in [("nvfp4", "bf16_batched"), ("nvfp4", "bf16_perq"), ("bf16", "nvfp4_batched"),
                       ("bf16", "nvfp4_perq")]:
        t = report["tiers"].get(tier)
        if not t or cand not in runs:
            continue
        g = gate(runs[cand], {r: runs[r] for r in t["paths"]}, tier_limits(tier, t["thresholds"]))
        fails = [f"{c['ref']}:{c['metric']}" for c in g["checks"] if c["metric"] != "info" and not c["pass"]]
        report["validation"]["cross_tier"][f"{tier}:{cand}"] = {"pass": g["pass"], "failed": fails}
        fixture += fixture_rows(tier, cand, g, tier_limits(tier, t["thresholds"]))
        print(f"[validate] {cand} as a {tier} candidate: {'PASS' if g['pass'] else 'FAIL'} {fails}")
    # Synthetic sensitivity: distort a reference run (tempered answer vectors, T != 1) and gate it as a candidate
    # of its own tier: which distortions does the gate catch? (A rough minimum-detectable-degradation probe.)
    report["validation"]["synthetic"] = {}
    for tier, base in [("nvfp4", "nvfp4_batched"), ("bf16", "bf16_batched")]:
        t = report["tiers"].get(tier)
        if not t or base not in runs:
            continue
        for temp in (0.97, 1.03, 1.1, 1.25, 1.5):
            cand = tempered(runs[base], temp)
            g = gate(cand, {r: runs[r] for r in t["paths"]}, tier_limits(tier, t["thresholds"]))
            fails = sorted({c["metric"] for c in g["checks"] if c["metric"] != "info" and not c["pass"]})
            report["validation"]["synthetic"][f"{tier}:{base}@T={temp}"] = {"pass": g["pass"], "failed_metrics": fails}
            fixture += fixture_rows(tier, cand["name"], g, tier_limits(tier, t["thresholds"]))
            print(f"[validate] {base} tempered T={temp} as a {tier} candidate: {'PASS' if g['pass'] else 'FAIL'} {fails}")
    # Minimal allowed deviations (synthetic top-1 flips on a reference run: the top two probabilities of a
    # question swapped). Each must PASS its tier's gate: they are what the floors and the margin allowance promise.
    report["validation"]["minimal_flips"] = {}
    for tier, base in [("nvfp4", "nvfp4_batched"), ("bf16", "bf16_batched")]:
        t = report["tiers"].get(tier)
        if not t or base not in runs:
            continue
        refs = {r: runs[r] for r in t["paths"]}
        for tag, cand in minimal_flips(runs[base], refs, MARGIN[tier]):
            g = gate(cand, refs, tier_limits(tier, t["thresholds"]))
            fails = sorted({c["metric"] for c in g["checks"] if c["metric"] != "info" and not c["pass"]})
            report["validation"]["minimal_flips"][f"{tier}:{base}:{tag}"] = {"pass": g["pass"], "failed_metrics": fails}
            fixture += fixture_rows(tier, f"{base}:{tag}", g, tier_limits(tier, t["thresholds"]))
            print(f"[validate] {base} with {tag} as a {tier} candidate: {'PASS' if g['pass'] else 'FAIL'} {fails}")
    report["rust_fixture"] = fixture
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {REPORT}")
    if args.write_tolerances:
        write_tolerances(report)
        print(f"updated [eval.*] in {TOLERANCES}")


if __name__ == "__main__":
    main()
