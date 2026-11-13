"""Per-layer hidden-state dumps from the reference model, for per-layer comparisons (DESIGN.md §9.1, §9.2 "Model").

Captures the residual stream after decoder layers 0, 3, 4, 15, 31 (any list) and the final-norm output, at each
branch's answer row (last suffix token) and at sampled prefix rows. Two forward modes:

  served        the qev engine's path: one prefix forward with cache, then each suffix forward from a copy of that
                cache (block-bidirectional mask in the full-attention layers). Equals sysone's packed tree forward.
  per_question  one full forward per branch (prefix + suffix), batch 1, use_cache=False: the NVFP4 checkpoint's
                evaluation path.

Library use (W1b's golden runs):  dump_tree(engine, prefix_ids, suffixes, ...) -> {name: tensor}
CLI use (W4 per-layer tests), with the reference venv:

  venv/bin/python -I reference/layer_dump.py --dtype bfloat16 --request req.json [--debias 3] --out h.safetensors
  venv/bin/python -I reference/layer_dump.py --dtype float32 --ids ids.json --out h.safetensors
        ids.json = {"prefix": [int...], "suffixes": [[int...], ...]}    (token ids as the frontend produced them)
  options: --model DIR, --nvfp4, --mode served|per_question, --layers 0,3,4,15,31, --prefix-rows 0,5,100

Output (safetensors, all hidden states f32 = exact upcast of the model dtype's values):
  layer{i}/prefix, norm/prefix   [R, 4096]  rows `prefix_rows` of the prefix
  layer{i}/answer, norm/answer   [B, 4096]  answer row of each branch, in suffix order
  ids/prefix [P] i64, ids/suffix{b} [S_b] i64, prefix_rows [R] i64
  metadata: model, dtype, mode, layers, prefix_rows, provenance (JSON)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qevref  # noqa: E402


def default_prefix_rows(P: int, n: int = 8) -> list[int]:
    """First and last two rows plus evenly spaced interior rows (deterministic)."""
    rows = {0, 1, P - 2, P - 1} | {round(P * k / (n - 3)) for k in range(1, n - 3)}
    return sorted(r for r in rows if 0 <= r < P)


class Taps:
    """Forward hooks on the chosen decoder layers and the final norm; each forward keeps only the rows asked for."""

    def __init__(self, model, layers):
        inner = model.model
        self.rows = None
        self.out: dict[str, list] = {}
        self.handles = [inner.layers[i].register_forward_hook(self._hook(f"layer{i}")) for i in layers]
        self.handles.append(inner.norm.register_forward_hook(self._hook("norm")))

    def _hook(self, name):
        def hook(module, args, output):
            if self.rows is None:
                return
            h = output[0] if isinstance(output, tuple) else output
            self.out.setdefault(name, []).append(h[0, self.rows].detach().float().cpu())
        return hook

    def capture(self, rows):
        self.rows = rows
        self.out = {}

    def take(self):
        out, self.out, self.rows = self.out, {}, None
        return out

    def remove(self):
        for h in self.handles:
            h.remove()


def dump_tree(engine, prefix_ids: list[int], suffixes: list[list[int]], mode: str = "served",
              layers=qevref.LAYERS, prefix_rows: list[int] | None = None) -> dict:
    import torch
    from qev import readout as ro

    model, dev = engine.model, engine.device
    P = len(prefix_ids)
    prefix_rows = prefix_rows if prefix_rows is not None else default_prefix_rows(P)
    taps = Taps(model, layers)
    names = [f"layer{i}" for i in layers] + ["norm"]
    prefix: dict[str, torch.Tensor] = {}
    answers: dict[str, list] = {n: [] for n in names}
    try:
        with torch.inference_mode():
            if mode == "served":
                taps.capture(prefix_rows)
                with ro.blocks(ro.block_ids([P], [P], P, dev)):
                    # logits_to_keep=1 as in the engine; the final norm still runs over every row
                    out = model(input_ids=torch.tensor([prefix_ids], device=dev),
                                attention_mask=torch.ones((1, P), dtype=torch.long, device=dev),
                                use_cache=True, return_dict=True, logits_to_keep=1)
                prefix = {k: v[0] for k, v in taps.take().items()}
                base = out.past_key_values
                del out
                for suf in suffixes:
                    cache = copy.deepcopy(base)
                    n = P + len(suf)
                    taps.capture([len(suf) - 1])
                    with ro.blocks(ro.block_ids([P], [n], n, dev)):
                        o = model(input_ids=torch.tensor([suf], device=dev),
                                  attention_mask=torch.ones((1, n), dtype=torch.long, device=dev),
                                  past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
                    for k, v in taps.take().items():
                        answers[k].append(v[0][0])
                    del o, cache
                del base
            elif mode == "per_question":
                for b, suf in enumerate(suffixes):
                    ids = torch.tensor([prefix_ids + suf], device=dev)
                    n = ids.shape[1]
                    taps.capture(([*prefix_rows] if b == 0 else []) + [n - 1])
                    with ro.blocks(ro.block_ids([P], [n], n, dev)):
                        o = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                  return_dict=True, logits_to_keep=1)
                    for k, v in taps.take().items():
                        rows = v[0]
                        if b == 0:
                            prefix[k] = rows[:-1]
                        answers[k].append(rows[-1])
                    del o
            else:
                raise ValueError(f"unknown mode {mode!r}")
    finally:
        taps.remove()
    tensors = {}
    for k in names:
        tensors[f"{k}/prefix"] = prefix[k].contiguous()
        tensors[f"{k}/answer"] = torch.stack(answers[k]).contiguous()
    tensors["ids/prefix"] = torch.tensor(prefix_ids, dtype=torch.int64)
    for b, suf in enumerate(suffixes):
        tensors[f"ids/suffix{b}"] = torch.tensor(suf, dtype=torch.int64)
    tensors["prefix_rows"] = torch.tensor(prefix_rows, dtype=torch.int64)
    return tensors


class _Stop(Exception):
    pass


def rendered_branches(engine, request: dict, debias: int = 1):
    """The (state, rendered branches) exactly as engine.decide() plans them, without running the model."""
    from qev.schema import SystemOneRequest

    got = {}

    def grab(state, rendered):
        got["v"] = (state, rendered)
        raise _Stop

    saved = engine._run_sequential, engine._run_batched
    engine._run_sequential = engine._run_batched = grab
    try:
        engine.decide(SystemOneRequest.model_validate(request), debias=debias)
    except _Stop:
        pass
    finally:
        engine._run_sequential, engine._run_batched = saved
    return got["v"]


def tree_ids(engine, request: dict, debias: int = 1) -> tuple[list[int], list[list[int]]]:
    state, rendered = rendered_branches(engine, request, debias)
    prefix_ids, suffixes, _ = engine._split(state, rendered)
    return prefix_ids, suffixes


def save(tensors: dict, path: Path, meta: dict) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={k: v if isinstance(v, str) else json.dumps(v) for k, v in meta.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--nvfp4", action="store_true", help="NVFP4 checkpoint as QDQ (run_compressed=False)")
    ap.add_argument("--mode", default="served", choices=["served", "per_question"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--request", type=Path, help="a /v1/systemone request body (JSON)")
    src.add_argument("--ids", type=Path, help='{"prefix": [...], "suffixes": [[...], ...]}')
    ap.add_argument("--debias", type=int, default=1)
    ap.add_argument("--layers", default=",".join(map(str, qevref.LAYERS)))
    ap.add_argument("--prefix-rows", default=None, help="comma-separated prefix rows (default: 8 spread rows)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = args.model or (qevref.NVFP4_MODEL if args.nvfp4 else qevref.BF16_MODEL)
    engine = qevref.load_engine(model, args.dtype, fork_mode="sequential", nvfp4=args.nvfp4)
    if args.request:
        prefix_ids, suffixes = tree_ids(engine, json.loads(args.request.read_text()), args.debias)
    else:
        spec = json.loads(args.ids.read_text())
        prefix_ids, suffixes = spec["prefix"], spec["suffixes"]
    layers = [int(x) for x in args.layers.split(",")]
    rows = [int(x) for x in args.prefix_rows.split(",")] if args.prefix_rows else None
    tensors = dump_tree(engine, prefix_ids, suffixes, args.mode, layers, rows)
    meta = {"model": str(model), "dtype": args.dtype, "nvfp4": args.nvfp4, "mode": args.mode, "layers": layers,
            "prefix_rows": tensors["prefix_rows"].tolist(), "provenance": qevref.provenance()}
    save(tensors, args.out, meta)
    print(f"wrote {args.out}: prefix {len(prefix_ids)} tokens, {len(suffixes)} branches, "
          f"{len(tensors)} tensors")


if __name__ == "__main__":
    main()
