"""Dense hidden-state dumps and the hidden-state gate (a) (DESIGN.md §9.2 "Model", Phase 2a workstream A1).

Gate (a) compares sysone's per-layer hidden states with the qev fp32 served path (the truth), row by row, against
the bf16 served path's own error on the SAME rows. Earlier it used 100 sampled rows (8 prefix rows per request plus
the answer rows), so its p99 was the second-largest row; deep-layer errors are heavy-tailed and row-specific
(summation order decides which rows are sensitive), so a p99 threshold from 100 reference rows did not bound the
reference's own outlier rows. This script replaces the row set and adds paired criteria.

Row set (fixed here, before any sysone evaluation): the 10 `dump_hidden` requests of reference/corpus_model; per
request every answer row (last suffix token of each branch) and the prefix rows `dense_prefix_rows(P)`:
stride s = max(1, P // 1024), rows {0, s, 2s, ...} plus P-2 and P-1. That is every prefix row for P < 2048 (all
requests up to the 1k one) and ~1k rows each for the 4k and 16k requests: 3,829 rows per tap in total.

Runs (each one `scripts/gpu.sh` job, one model load):
  fp32_served        qev fp32, served path (prefix with cache, then each suffix): the truth
  bf16_served        qev bf16, served path: the reference whose error sets the threshold
  bf16_perq          qev bf16, per-question path (one prefix+suffix forward per branch, no cache): another bf16
                     path (bit-identical to served on 55-92% of rows, depending on the tap)
  bf16_perm{1,2,3}   bf16 served with the hidden basis permuted (seeds 1-3; permute_hidden): the same function,
                     every hidden-dim reduction in another order -> independent bf16 rounding on every row
  bf16_norm_llama    bf16 served with every Qwen3_5RMSNorm rounding the normalized x to bf16 before the weight
                     multiply (the Llama ordering the HF code warns about): a mildly degraded path
  bf16_norm_bf16     bf16 served with the whole RMSNorm in bf16 (variance, rsqrt, scale): a degraded path

  PY=/home/velvet/Desktop/sysone-ref/venv/bin/python
  for r in fp32_served bf16_served bf16_perq bf16_norm_llama bf16_norm_bf16 bf16_perm1 bf16_perm2 bf16_perm3; do
      scripts/gpu.sh $PY -I reference/hidden_dense.py dump --run $r; done
  $PY -I reference/hidden_dense.py manifest            # tests/golden/dev/model/hidden_dense_manifest.json
  $PY -I reference/hidden_dense.py manifest --check
  $PY -I reference/hidden_dense.py gate --write-tolerances   # CPU: validation report + [model.hidden.*]

Dumps: <sysone-ref>/golden/dev/hidden_dense/<run>/hidden_<id>.safetensors (layout of layer_dump.py), outside git;
the committed manifest holds their sha256.

Gate, per tap (layer0, 3, 4, 15, 31, norm), with e_c(r) = ||h_c - h_fp32|| / ||h_fp32|| (candidate) and e_b(r) the
same for bf16_served, over the same rows r:
  (i)   marginal:  p99(e_c) <= factor x p99(e_b)  (nearest rank; factor from tolerances.toml). max is reported only.
  (ii)  paired:    median_r log(e_c / e_b) <= log(paired_ratio)   (lower median, nearest rank)
  (iii) sign test: n_up = #{e_c > sign_ratio x e_b}, n_down = #{e_b > sign_ratio x e_c}, deflated by the tap's
                   design effect D: k = floor(n / D + 1/2); fail iff P(X >= k_up | Binomial(k_up + k_down, 1/2))
                   < sign_alpha (one-sided, exact). D = max(factor x measured, 1), measured = the largest
                   request-clustered dispersion of the row signs over every ordered pair of the five bf16
                   realizations (served, perq, perm1-3). The plain row binomial (D = 1) fails these equivalent
                   realizations at layer31/norm (p down to 3.6e-4): rows of one request move together.

Validation (`gate`, reference-only, before any sysone run): the null realizations vs the reference (both role
orders), every ordered pair of them, each with the design effect re-measured without it, and leave-one-request-out
must pass; synthetic regressions on top of bf16_perm1 must fail (every row's error x1.75 and x2; 5% of rows, strided
or as one block, x3 and more; gaussian noise of 2x the reference's median error); smaller ones and the two degraded
norm paths are reported (`expect: info`).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qevref  # noqa: E402

ROOT = qevref.GOLDEN_BIN / "hidden_dense"
MODEL_DIR = qevref.REPO / "tests" / "golden" / "dev" / "model"
MANIFEST = MODEL_DIR / "hidden_dense_manifest.json"
REPORT = MODEL_DIR / "hidden_gate_report.json"
TOLERANCES = qevref.REPO / "tests" / "tolerances.toml"
CORPUS = qevref.REPO / "reference" / "corpus_model" / "requests.jsonl"
TAPS = list(qevref.LEVELS)
H = 4096

RUNS = {
    "fp32_served": {"dtype": "float32", "mode": "served", "degrade": None},
    "bf16_served": {"dtype": "bfloat16", "mode": "served", "degrade": None},
    "bf16_perq": {"dtype": "bfloat16", "mode": "per_question", "degrade": None},
    "bf16_norm_llama": {"dtype": "bfloat16", "mode": "served", "degrade": "norm_llama"},
    "bf16_norm_bf16": {"dtype": "bfloat16", "mode": "served", "degrade": "norm_bf16"},
    "bf16_perm1": {"dtype": "bfloat16", "mode": "served", "degrade": None, "permute": 1},
    "bf16_perm2": {"dtype": "bfloat16", "mode": "served", "degrade": None, "permute": 2},
    "bf16_perm3": {"dtype": "bfloat16", "mode": "served", "degrade": None, "permute": 3},
}
NULLS = ["bf16_perq", "bf16_perm1", "bf16_perm2", "bf16_perm3"]
TRUTH, REF = "fp32_served", "bf16_served"
NULL = "bf16_perm1"  # base of the synthetic regressions (an independent bf16 realization on every row)
ROWSET_RULE = ("per dump_hidden request: prefix rows {0, s, 2s, ...} | {P-2, P-1} with s = max(1, P // 1024), "
               "then every branch's answer row (last suffix token), in suffix order; requests sorted by id")

# Gate parameters (committed with the gate; the marginal factor is tolerances.toml's global `factor`).
PAIRED_RATIO = 1.5
SIGN_RATIO = 1.5
SIGN_ALPHA = 0.01
ROWS_MIN = 1000
EPS = 1e-30  # floor for e in log ratios (an exact-zero error only happens for bit-identical rows)


def dense_prefix_rows(P: int) -> list[int]:
    s = max(1, P // 1024)
    return sorted(set(range(0, P, s)) | {r for r in (P - 2, P - 1) if r >= 0})


def dump_requests() -> list[dict]:
    items = [x for x in qevref.load_corpus(CORPUS) if x.get("dump_hidden")]
    return sorted(items, key=lambda x: x["id"])


# ---------------------------------------------------------------------------------------------------- dumping

# Hidden-basis permutation (the bf16 null realizations): new = old[..., perm] on the residual stream. The model
# computes exactly the same function (RMSNorm is permutation-invariant; every projection reading or writing the
# residual stream is permuted to match), but every reduction over the hidden dimension (GEMM K loops, the norms'
# mean of squares) sums in another order, so bf16 rounding differs on every row: an independent bf16 realization
# of the reference, as another correct implementation would be.
PERM_IN = ("embed_tokens.weight", "lm_head.weight", "in_proj_a.weight", "in_proj_b.weight", "in_proj_qkv.weight",
           "in_proj_z.weight", "gate_proj.weight", "up_proj.weight", "q_proj.weight", "k_proj.weight",
           "v_proj.weight")
PERM_OUT = ("linear_attn.out_proj.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight")


def permute_hidden(model, seed: int):
    import torch

    perm = torch.randperm(H, generator=torch.Generator().manual_seed(seed))
    counts = {"in": 0, "out": 0, "vec": 0}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "visual" in name:
                continue
            pd = perm.to(p.device)
            if name.endswith(PERM_IN):
                assert p.shape[-1] == H, name
                p.data = p.data[:, pd].contiguous()
                counts["in"] += 1
            elif name.endswith(PERM_OUT):
                assert p.shape[0] == H, name
                p.data = p.data[pd, :].contiguous()
                counts["out"] += 1
            elif name.endswith("norm.weight") and tuple(p.shape) == (H,):  # layer norms and the final norm
                assert tuple(p.shape) == (H,), name
                p.data = p.data[pd].contiguous()
                counts["vec"] += 1
    print(f"permuted hidden basis (seed {seed}): {counts}", flush=True)
    # 24 GDN layers (4 in, 1 out) + 8 attention layers (3 in, 1 out) + 32 MLPs (2 in, 1 out) + embed (+ head when
    # it is a separate parameter); 2 norms per layer + the final norm
    assert counts["out"] == 64 and counts["vec"] == 65 and counts["in"] in (24 * 4 + 8 * 3 + 64 + 1,
                                                                             24 * 4 + 8 * 3 + 64 + 2), counts
    return torch.argsort(perm)

def degrade_norm(kind: str) -> None:
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    def llama(self, x):
        out = self._norm(x.float()).to(x.dtype)
        return out * (1.0 + self.weight.float()).to(x.dtype)

    def bf16(self, x):
        w = (1.0 + self.weight.float()).to(x.dtype)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    m.Qwen3_5RMSNorm.forward = {"norm_llama": llama, "norm_bf16": bf16}[kind]


def cmd_dump(args) -> None:
    import torch
    import layer_dump

    cfg = RUNS[args.run]
    if cfg["degrade"]:
        degrade_norm(cfg["degrade"])
    engine = qevref.load_engine(qevref.BF16_MODEL, cfg["dtype"], fork_mode="sequential")
    if cfg["degrade"]:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
        n = sum(isinstance(x, m.Qwen3_5RMSNorm) for x in engine.model.modules())
        assert n > 64, f"degrade {cfg['degrade']}: only {n} Qwen3_5RMSNorm modules"
    inv = permute_hidden(engine.model, cfg["permute"]) if cfg.get("permute") else None
    out_dir = ROOT / args.run
    only = set(args.only.split(",")) if args.only else None
    for item in dump_requests():
        if only and item["id"] not in only:
            continue
        path = out_dir / f"hidden_{item['id']}.safetensors"
        if path.exists() and not args.force:
            print(f"{item['id']}: exists, skipped")
            continue
        prefix_ids, suffixes = layer_dump.tree_ids(engine, item["request"], item["debias"])
        rows = dense_prefix_rows(len(prefix_ids))
        torch.cuda.reset_peak_memory_stats()
        tensors = layer_dump.dump_tree(engine, prefix_ids, suffixes, cfg["mode"], qevref.LAYERS, rows)
        if inv is not None:  # back to the model's hidden basis
            for k in [k for k in tensors if k.startswith(("layer", "norm/"))]:
                tensors[k] = tensors[k][:, inv].contiguous()
        meta = {"request": item["id"], "run": args.run, **{k: v for k, v in cfg.items()}, "rowset": ROWSET_RULE,
                "layers": list(qevref.LAYERS), "provenance": qevref.provenance()}
        layer_dump.save(tensors, path, meta)
        print(f"{item['id']}: P={len(prefix_ids)} branches={len(suffixes)} rows={len(rows)} "
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB -> {path}", flush=True)


def cmd_manifest(args) -> None:
    import write_manifest

    if args.check:
        man = json.loads(MANIFEST.read_text())
        root = Path(man["root"])
        bad = [rel for rel, e in man["files"].items()
               if not (root / rel).exists() or qevref.sha256_file(root / rel) != e["sha256"]]
        print(f"{len(man['files']) - len(bad)}/{len(man['files'])} files match" + (f"; bad: {bad}" if bad else ""))
        sys.exit(1 if bad else 0)
    files = {}
    for path in sorted(ROOT.glob("*/*.safetensors")):
        e = write_manifest.describe(path)
        e["metadata"].pop("provenance", None)
        files[str(path.relative_to(ROOT))] = e
    prov = None
    for path in sorted(ROOT.glob("*/*.safetensors")):
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            prov = json.loads((f.metadata() or {}).get("provenance", "null"))
        break
    out = {"root": str(ROOT), "about": "Dense per-layer hidden-state dumps for gate (a); reference/hidden_dense.py "
           "(layout of reference/layer_dump.py, f32 upcasts of the model dtype).", "rowset": ROWSET_RULE,
           "runs": RUNS, "provenance": prov, "files": files}
    MANIFEST.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST}: {len(files)} files, {sum(f['bytes'] for f in files.values()) / 2**20:.1f} MiB")


# ---------------------------------------------------------------------------------------------------- gate

def binom_sf_half(k: int, n: int) -> float:
    """P(X >= k), X ~ Binomial(n, 1/2), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2 ** n))


def nearest_rank(sorted_xs, p: float) -> float:
    n = len(sorted_xs)
    return float(sorted_xs[min(max(math.ceil(p * n), 1), n) - 1])


def sign_counts(ec, eb) -> tuple:
    import numpy as np
    return int(np.sum(ec > SIGN_RATIO * eb)), int(np.sum(eb > SIGN_RATIO * ec))


def effective(count: int, design_effect: float) -> int:
    """A row count deflated by the design effect, rounded half up."""
    return math.floor(count / design_effect + 0.5)


def cluster_dispersion(ec, eb, ids) -> float:
    """Request-clustered dispersion of the sign indicators d_r = [e_c > s e_b] - [e_b > s e_c]:
    sum_u (sum_{r in u} d_r)^2 / sum_r |d_r|. 1 for independent rows; > 1 when rows of a request move together."""
    import numpy as np
    d = (ec > SIGN_RATIO * eb).astype(np.int64) - (eb > SIGN_RATIO * ec).astype(np.int64)
    ids = np.asarray(ids)
    s2 = sum(int(d[ids == u].sum()) ** 2 for u in sorted(set(ids.tolist())))
    n = int(np.abs(d).sum())
    return s2 / n if n else 0.0


def evaluate(ec, eb, factor: float, design_effect: float) -> dict:
    """Gate (a) on one tap: candidate errors ec vs reference errors eb, row-aligned; the sign test runs on the row
    counts deflated by the tap's design effect."""
    import numpy as np

    ec, eb = np.asarray(ec, dtype=np.float64), np.asarray(eb, dtype=np.float64)
    assert ec.shape == eb.shape and ec.ndim == 1
    sc, sb = np.sort(ec), np.sort(eb)
    lr = np.sort(np.log(np.maximum(ec, EPS)) - np.log(np.maximum(eb, EPS)))
    up, down = sign_counts(ec, eb)
    ku, kd = effective(up, design_effect), effective(down, design_effect)
    s = {"n": int(ec.size), "cand_p50": nearest_rank(sc, 0.5), "cand_p99": nearest_rank(sc, 0.99),
         "cand_max": float(sc[-1]), "ref_p50": nearest_rank(sb, 0.5), "ref_p99": nearest_rank(sb, 0.99),
         "ref_max": float(sb[-1]), "median_log_ratio": nearest_rank(lr, 0.5), "n_up": up, "n_down": down,
         "design_effect": design_effect, "eff_up": ku, "eff_down": kd, "sign_p": binom_sf_half(ku, ku + kd),
         "sign_p_rows": binom_sf_half(up, up + down)}
    fails = []
    if not s["cand_p99"] <= factor * s["ref_p99"]:
        fails.append("p99")
    if not s["median_log_ratio"] <= math.log(PAIRED_RATIO):
        fails.append("median_log_ratio")
    if not s["sign_p"] >= SIGN_ALPHA:
        fails.append("sign")
    s["failed"] = fails
    return s


def load_rows(run: str) -> tuple[dict, list[str], list[str]]:
    """{tap: [R, H] float32} over the row set, plus row labels and request ids, for one run."""
    import numpy as np
    from safetensors.numpy import load_file

    per_tap = {t: [] for t in TAPS}
    labels, ids = [], []
    for item in dump_requests():
        d = load_file(str(ROOT / run / f"hidden_{item['id']}.safetensors"))
        P = d["ids/prefix"].shape[0]
        rows = d["prefix_rows"].tolist()
        assert rows == dense_prefix_rows(P), f"{run} {item['id']}: prefix rows are not the row set"
        nb = sum(1 for k in d if k.startswith("ids/suffix"))
        for t in TAPS:
            per_tap[t].append(np.concatenate([d[f"{t}/prefix"], d[f"{t}/answer"]]))
        labels += [f"{item['id']} prefix {r}" for r in rows] + [f"{item['id']} answer {b}" for b in range(nb)]
        ids += [item["id"]] * (len(rows) + nb)
    return {t: np.concatenate(v) for t, v in per_tap.items()}, labels, ids


def rel_rows(a, truth):
    import numpy as np

    a, truth = a.astype(np.float64), truth.astype(np.float64)
    return np.sqrt(((a - truth) ** 2).sum(1) / np.maximum((truth ** 2).sum(1), 1e-300))


def strided5(n: int):
    """5% of rows, spread: row k with k % 20 == 7."""
    import numpy as np
    m = np.zeros(n, dtype=bool)
    m[7::20] = True
    return m


def block5(n: int):
    """5% of rows, contiguous: the last ceil(n/20) rows (the 16k request's deepest positions and its answers)."""
    import numpy as np
    m = np.zeros(n, dtype=bool)
    m[n - math.ceil(n / 20):] = True
    return m


def cmd_gate(args) -> None:
    import numpy as np

    t = {}
    labels = ids = None
    for run in RUNS:
        t[run], lab, rid = load_rows(run)
        if labels is None:
            labels, ids = lab, rid
        assert lab == labels, f"{run}: row set differs from {TRUTH}"
    n = len(labels)
    assert n >= ROWS_MIN, n
    factor = 1.5  # tolerances.toml `factor` (checked below)
    tol_factor = float(next(l.split("=")[1] for l in TOLERANCES.read_text().splitlines() if l.startswith("factor ")))
    assert tol_factor == factor, tol_factor
    err = {run: {tap: rel_rows(t[run][tap], t[TRUTH][tap]) for tap in TAPS} for run in RUNS if run != TRUTH}

    # Design effect of the sign test, per tap: rows of one request move together (a realization that is worse on
    # one region of a long request is worse on many adjacent rows), so the row signs are over-dispersed relative
    # to Binomial(n, 1/2). Measured as the largest request-clustered dispersion over every ordered pair of the five
    # bf16 realizations (served, per-question, three hidden-basis permutations); limit = max(factor x measured, 1),
    # rounded to the 4 significant digits written to tolerances.toml (the value the gate then uses).
    realizations = [REF, *NULLS]
    design, design_src = {}, {}
    for tap in TAPS:
        best, src = 0.0, ""
        for c in realizations:
            for r in realizations:
                if c != r:
                    v = cluster_dispersion(err[c][tap], err[r][tap], ids)
                    if v >= best:
                        best, src = v, f"{c}~{r}"
        design_src[tap] = {"measured": best, "pair": src}
        design[tap] = float(_num(max(factor * best, 1.0)))

    # Bit-identity of the dense bf16 served rerun with the W1b bf16 dumps at the rows both hold (determinism).
    from safetensors.numpy import load_file
    rerun = {"rows_compared": 0, "max_abs_diff": 0.0}
    for item in dump_requests():
        old = load_file(str(qevref.GOLDEN_BIN / "bf16_batched" / f"hidden_{item['id']}.safetensors"))
        new = load_file(str(ROOT / REF / f"hidden_{item['id']}.safetensors"))
        orows, nrows = old["prefix_rows"].tolist(), new["prefix_rows"].tolist()
        for tap in TAPS:
            for i, r in enumerate(orows):
                if r in nrows:
                    d = float(np.abs(old[f"{tap}/prefix"][i] - new[f"{tap}/prefix"][nrows.index(r)]).max())
                    rerun["max_abs_diff"] = max(rerun["max_abs_diff"], d)
                    rerun["rows_compared"] += 1
            d = float(np.abs(old[f"{tap}/answer"] - new[f"{tap}/answer"]).max())
            rerun["max_abs_diff"] = max(rerun["max_abs_diff"], d)

    rng = np.random.default_rng(20261113)
    cands = []  # (name, kind, expect, {tap: (ec, eb)}, rust_check)

    def add(name, kind, expect, pairs, rust=False, info=""):
        cands.append((name, kind, expect, pairs, rust, info))

    # Null candidates: other correct bf16 realizations of the reference, scored against the served reference (as
    # sysone will be), with roles swapped, and against each other (every ordered pair of the bf16 realizations):
    # the gate must pass all of them.
    real = [REF, *NULLS]
    for c in NULLS:
        add(f"null:{c}", "null", "pass", {tap: (err[c][tap], err[REF][tap]) for tap in TAPS}, True,
            f"{c} vs the bf16 served reference")
        add(f"null_swapped:{c}", "null", "pass", {tap: (err[REF][tap], err[c][tap]) for tap in TAPS}, True,
            f"roles swapped: the served reference as candidate, {c} as reference")
    for c in NULLS:
        for r in NULLS:
            if c != r:
                add(f"null_pair:{c}~{r}", "null_pair", "pass", {tap: (err[c][tap], err[r][tap]) for tap in TAPS},
                    False, f"{c} as candidate, {r} as reference")
    # Leave one request out: every null comparison against the served reference, on 9 of the 10 requests.
    uniq = sorted(set(ids))
    ida = np.array(ids)
    for rid in uniq:
        keep = ida != rid
        for c in NULLS:
            add(f"loo:{rid}:{c}", "loo", "pass", {tap: (err[c][tap][keep], err[REF][tap][keep]) for tap in TAPS},
                False, f"{c} vs the reference with request {rid} left out")
            add(f"loo_swapped:{rid}:{c}", "loo", "pass",
                {tap: (err[REF][tap][keep], err[c][tap][keep]) for tap in TAPS}, False,
                f"swapped, with request {rid} left out")
    assert set(real) <= set(err)
    # Synthetic regressions on top of the null candidate (a realistic, independent bf16 summation order).
    # Scaling: h_c = h_fp32 + s (h_null - h_fp32), so e_c = s e_null exactly.
    for s in (1.1, 1.25, 1.4, 1.6, 1.75, 2.0):
        add(f"scale_all:{s}", "scale_all", "fail" if s >= 1.75 else "info",
            {tap: (s * err[NULL][tap], err[REF][tap]) for tap in TAPS}, True,
            f"every row's error x{s}")
    for name, mask in (("rows5_strided", strided5(n)), ("rows5_block", block5(n))):
        for k in (1.5, 2.0, 3.0, 5.0, 10.0):
            add(f"{name}:{k}", name, "fail" if k >= 3.0 else "info",
                {tap: (np.where(mask, k * err[NULL][tap], err[NULL][tap]), err[REF][tap]) for tap in TAPS}, True,
                f"5% of rows ({int(mask.sum())}) with error x{k}")
    # Additive noise: h_c = h_null + g, g ~ N(0, sigma^2 I) per row with ||g|| ~ x * ref_p50(tap) * ||h_fp32||.
    for x in (0.5, 1.0, 2.0):
        pairs = {}
        for tap in TAPS:
            p50 = float(np.median(err[REF][tap]))
            nrm = np.sqrt((t[TRUTH][tap].astype(np.float64) ** 2).sum(1))
            g = rng.standard_normal((n, H)) * (x * p50 * nrm / math.sqrt(H))[:, None]
            hc = t[NULL][tap].astype(np.float64) + g
            pairs[tap] = (rel_rows(hc, t[TRUTH][tap]), err[REF][tap])
        add(f"noise_all:{x}", "noise_all", "fail" if x >= 2.0 else "info", pairs, False,
            f"gaussian noise of {x} x the reference's median relative error on every row")
    for run, info in (("bf16_norm_llama", "RMSNorm rounds x_hat to bf16 before the weight (Llama order)"),
                      ("bf16_norm_bf16", "RMSNorm entirely in bf16")):
        add(f"degraded:{run}", "degraded", "info", {tap: (err[run][tap], err[REF][tap]) for tap in TAPS}, False, info)

    results = []
    for name, kind, expect, pairs, rust, info in cands:
        per_tap = {tap: evaluate(*pairs[tap], factor, design[tap]) for tap in TAPS}
        verdict = "pass" if all(not v["failed"] for v in per_tap.values()) else "fail"
        results.append({"name": name, "kind": kind, "info": info, "expect": expect, "verdict": verdict,
                        "ok": expect == "info" or expect == verdict, "rust_check": rust, "taps": per_tap})
    ref_stats = {tap: evaluate(err[REF][tap], err[REF][tap], factor, design[tap]) for tap in TAPS}
    # Out of sample for the design effect: each null realization vs the reference (both role orders) with the
    # design effect re-measured from the pairs of the OTHER realizations only (it never saw this candidate).
    for c in NULLS:
        others = [x for x in realizations if x != c]
        lro = {tap: float(_num(max(factor * max(cluster_dispersion(err[a][tap], err[b][tap], ids)
                                                 for a in others for b in others if a != b), 1.0))) for tap in TAPS}
        for name, a, b in ((f"null_lro:{c}", c, REF), (f"null_lro_swapped:{c}", REF, c)):
            per_tap = {tap: evaluate(err[a][tap], err[b][tap], factor, lro[tap]) for tap in TAPS}
            verdict = "pass" if all(not v["failed"] for v in per_tap.values()) else "fail"
            results.append({"name": name, "kind": "null_lro", "info": f"design effect without {c}: {lro}",
                            "expect": "pass", "verdict": verdict, "ok": verdict == "pass", "rust_check": False,
                            "taps": per_tap})

    # worst reference rows per tap (where the outliers sit)
    worst = {}
    for tap in TAPS:
        o = np.argsort(-err[REF][tap])[:5]
        worst[tap] = [{"row": labels[i], "ref": float(err[REF][tap][i]),
                       **{c: float(err[c][tap][i]) for c in NULLS}} for i in o]

    report = {
        "about": "Gate (a) validation (reference/hidden_dense.py gate). Candidates are scored against the bf16 served "
                 "reference vs the fp32 served truth on the dense row set; reference-only data.",
        "rowset": ROWSET_RULE, "rows": n, "rows_per_request": {r: ids.count(r) for r in uniq},
        "gate": {"factor": factor, "paired_ratio": PAIRED_RATIO, "sign_ratio": SIGN_RATIO, "sign_alpha": SIGN_ALPHA,
                 "rows_min": ROWS_MIN, "eps": EPS, "design_effect": design,
                 "measured_design_effect": design_src},
        "manifest_sha256": qevref.sha256_file(MANIFEST) if MANIFEST.exists() else None,
        "bf16_rerun_vs_w1b": rerun,
        "reference": {tap: {k: ref_stats[tap][k] for k in ("n", "ref_p50", "ref_p99", "ref_max")} for tap in TAPS},
        "reference_worst_rows": worst,
        "all_ok": all(r["ok"] for r in results),
        "candidates": results,
    }
    REPORT.write_text(json.dumps(report, indent=1) + "\n")
    print(f"rows {n}; bf16 rerun vs W1b dumps: {rerun}")
    for tap in TAPS:
        r = ref_stats[tap]
        print(f"  ref {tap:7s} p50 {r['ref_p50']:.3e} p99 {r['ref_p99']:.3e} max {r['ref_max']:.3e}; design effect "
              f"{design[tap]} (measured {design_src[tap]['measured']:.3f}, {design_src[tap]['pair']})")
    for r in results:
        if r["kind"] in ("loo", "null_pair", "null_lro") and r["verdict"] == "pass":
            continue
        print(f"{r['name']:34s} expect {r['expect']:4s} -> {r['verdict']:4s} {'' if r['ok'] else '  <-- UNEXPECTED'}")
        for tap, v in r["taps"].items():
            print(f"    {tap:7s} p99 {v['cand_p99']:.3e}/{v['ref_p99']:.3e} medlr {v['median_log_ratio']:+.3f} "
                  f"up {v['n_up']:4d} down {v['n_down']:4d} p {v['sign_p']:.2e} (rows {v['sign_p_rows']:.1e}) "
                  f"{','.join(v['failed'])}")
    for kind in ("null", "null_pair", "null_lro", "loo"):
        rs = [r for r in results if r["kind"] == kind]
        print(f"{kind}: {sum(r['verdict'] == 'pass' for r in rs)}/{len(rs)} pass")
    print(f"all expectations met: {report['all_ok']}; wrote {REPORT}")
    if args.write_tolerances:
        write_tolerances(report)
    if not report["all_ok"]:
        sys.exit(1)


BEGIN = "# >>> [model.hidden] generated by reference/hidden_dense.py gate --write-tolerances (do not edit by hand) >>>"
END = "# <<< [model.hidden] end <<<"


def _num(x: float) -> str:
    return f"{x:.4g}" if x != 0 else "0.0"


def render_tolerances(report: dict) -> str:
    g = report["gate"]
    L = [BEGIN,
         "# Per-layer hidden states (DESIGN §9.2 Model, gate (a)): relative error e(r) = ||h - h_fp32|| / ||h_fp32|| per",
         "# row vs the qev fp32 served path, compared with the bf16 served reference's own error on the SAME rows.",
         f"# Row set ({report['rows']} rows per tap): {ROWSET_RULE}.",
         "# Dumps: tests/golden/dev/model/hidden_dense_manifest.json; validation: hidden_gate_report.json.",
         "# Per tap, a candidate passes iff (i) p99(e_c) <= rel_p99 = factor x measured_rel_p99 (the reference's p99",
         "# on these rows; max is reported, not gated); (ii) the lower median over rows of log(e_c / e_b) is at most",
         "# log(paired_ratio); (iii) with n_up = #{e_c > sign_ratio x e_b} and n_down = #{e_b > sign_ratio x e_c},",
         "# deflated by the tap's design effect D (k = floor(n / D + 1/2)), the one-sided exact binomial",
         "# p = P(X >= k_up | Binomial(k_up + k_down, 1/2)) is at least sign_alpha.",
         "# D = max(factor x measured_sign_design_effect, 1): rows of one request move together, so row signs are",
         "# over-dispersed; measured = the largest request-clustered dispersion sum_u (net_u)^2 / sum_u discordant_u",
         "# over every ordered pair of five bf16 realizations of the reference (served, per-question, three",
         "# hidden-basis permutations). Without it the plain row binomial fails equivalent bf16 realizations",
         "# (p down to 3.6e-4 at layer31/norm). The validation report lists the plain row p as sign_p_rows.",
         "[model.hidden]",
         f'rowset = "dense_v1"',
         f"rows_min = {g['rows_min']}",
         f"paired_ratio = {g['paired_ratio']}",
         f"sign_ratio = {g['sign_ratio']}",
         f"sign_alpha = {g['sign_alpha']}"]
    for tap in TAPS:
        r = report["reference"][tap]
        L += [f"[model.hidden.{tap}]",
              f"rel_p99 = {_num(g['factor'] * r['ref_p99'])}",
              f"measured_rel_p99 = {r['ref_p99']!r}",
              f"measured_rel_max = {r['ref_max']!r}",
              f"measured_rel_p50 = {r['ref_p50']!r}",
              f"measured_rows = {r['n']}",
              f"sign_design_effect = {_num(g['design_effect'][tap])}",
              f"measured_sign_design_effect = {g['measured_design_effect'][tap]['measured']!r}"
              f"  # {g['measured_design_effect'][tap]['pair']}",
              "floor_sign_design_effect = 1.0"]
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
    print(f"wrote [model.hidden] to {TOLERANCES}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True, choices=list(RUNS))
    d.add_argument("--only", default=None)
    d.add_argument("--force", action="store_true")
    m = sub.add_parser("manifest")
    m.add_argument("--check", action="store_true")
    g = sub.add_parser("gate")
    g.add_argument("--write-tolerances", action="store_true")
    args = ap.parse_args()
    {"dump": cmd_dump, "manifest": cmd_manifest, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
