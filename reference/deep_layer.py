"""Deep-layer diagnostics for gate (a) (Phase 2a follow-up A3): where does sysone's layer31/norm excess come from?

Two tools, both on the reference venv, each GPU job through scripts/gpu.sh:

  dump    --run fp32_served|bf16_served|bf16_perm1 [--only id,id]
          The residual stream after EVERY decoder layer (0..31) and the final norm, on gate (a)'s dense row set
          (hidden_dense.py), in the layout of layer_dump.py -> <sysone-ref>/golden/dev/a3_deep/<run>/.
          sysone's counterpart: `cargo test --release --test engine_layers -- --ignored hidden_dump_all_layers`
          with SYSONE_TAP_OUT.

  forced  --dtype float32|bfloat16 --sys <sysone dump with full/ streams> --out <file> [--permute SEED]
          Teacher-forced single layers on a request's prefix: for each layer L the reference model runs the served
          prefix forward with the residual stream entering layer L replaced by sysone's own stream after layer L-1
          (the embedding for L = 0), and records, for every prefix row, the layer's mixer output (`mix{L}`), the
          residual after the mixer add (`mid{L}`), the MLP output (`mlp{L}`) and the layer output (`layer{L}`).
          The same input goes through sysone's layer L in the sysone run, so the three outputs (sysone, bf16
          reference, fp32 reference) of the same layer on the same input can be compared op group by op group.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qevref  # noqa: E402
import hidden_dense  # noqa: E402

ROOT = qevref.GOLDEN_BIN / "a3_deep"
ALL_LAYERS = tuple(range(32))
RUNS = {k: hidden_dense.RUNS[k] for k in ("fp32_served", "bf16_served", "bf16_perm1", "bf16_perm2")}


def cmd_dump(args) -> None:
    import torch
    import layer_dump

    cfg = RUNS[args.run]
    engine = qevref.load_engine(qevref.BF16_MODEL, cfg["dtype"], fork_mode="sequential")
    inv = hidden_dense.permute_hidden(engine.model, cfg["permute"]) if cfg.get("permute") else None
    only = set(args.only.split(",")) if args.only else None
    for item in hidden_dense.dump_requests():
        if only and item["id"] not in only:
            continue
        path = ROOT / args.run / f"hidden_{item['id']}.safetensors"
        if path.exists() and not args.force:
            print(f"{item['id']}: exists, skipped")
            continue
        prefix_ids, suffixes = layer_dump.tree_ids(engine, item["request"], item["debias"])
        rows = hidden_dense.dense_prefix_rows(len(prefix_ids))
        tensors = layer_dump.dump_tree(engine, prefix_ids, suffixes, cfg["mode"], ALL_LAYERS, rows)
        if inv is not None:
            for k in [k for k in tensors if k.startswith(("layer", "norm/"))]:
                tensors[k] = tensors[k][:, inv].contiguous()
        layer_dump.save(tensors, path, {"request": item["id"], "run": args.run, **cfg,
                                        "layers": list(ALL_LAYERS), "provenance": qevref.provenance()})
        print(f"{item['id']}: P={len(prefix_ids)} rows={len(rows)} -> {path}", flush=True)


def cmd_forced(args) -> None:
    import torch
    from safetensors.numpy import load_file
    from qev import readout as ro

    sysd = load_file(str(args.sys))
    segs = sysd["segments"].reshape(-1, 4)
    assert segs[0][2] == -1 and segs[0][0] == 0, "first segment must be the prefix"
    P = int(segs[0][1])
    prefix_ids = sysd["token_ids"][:P].tolist()
    engine = qevref.load_engine(qevref.BF16_MODEL, args.dtype, fork_mode="sequential")
    model, dev = engine.model, engine.device
    dt = next(model.parameters()).dtype
    inner = model.model
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(ALL_LAYERS)
    rows = list(range(0, P, args.stride))
    out: dict[str, torch.Tensor] = {}
    state: dict = {}

    def grab(name):
        def hook(module, a, o):
            if state.get("on") != name.split(":")[0]:
                return
            h = o[0] if isinstance(o, tuple) else o
            out[name.split(":")[1]] = h[0, rows].detach().float().cpu()
        return hook

    def grab_in(name):
        def hook(module, a):
            if state.get("on") != name.split(":")[0]:
                return
            out[name.split(":")[1]] = a[0][0, rows].detach().float().cpu()
        return hook

    class Stop(Exception):
        pass

    def stop(module, a, o):
        if state.get("on") is not None and state.get("stop") is module:
            raise Stop

    handles = []
    for L in layers:
        lay = inner.layers[L]
        mixer = lay.linear_attn if hasattr(lay, "linear_attn") else lay.self_attn
        handles.append(mixer.register_forward_hook(grab(f"{L}:mix{L}")))
        handles.append(lay.post_attention_layernorm.register_forward_pre_hook(grab_in(f"{L}:mid{L}")))
        handles.append(lay.mlp.register_forward_hook(grab(f"{L}:mlp{L}")))
        handles.append(lay.register_forward_hook(grab(f"{L}:layer{L}")))
        handles.append(lay.register_forward_hook(stop))
    try:
        with torch.inference_mode():
            for L in layers:
                src = sysd[f"full/layer{L - 1}"][:P] if L > 0 else None
                hooked = inner.layers[L]
                # the pre-hook replaces the input of layer L; earlier layers run as usual (their output is dropped)
                state.clear()
                state["on"], state["stop"] = str(L), hooked
                if src is not None:
                    x = torch.from_numpy(src).to(dev)
                    assert torch.equal(x.to(torch.bfloat16).float(), x), "sysone stream is not bf16"
                    state["x"] = x[None].to(dt)

                    # install only on layer L: other layers' pre-hooks must not consume it
                    def only_L(module, a, kw, _L=L):
                        if module is inner.layers[_L] and state.get("x") is not None:
                            x2 = state.pop("x")
                            return (x2,) + tuple(a[1:]), kw
                        return None
                    h = hooked.register_forward_pre_hook(only_L, with_kwargs=True, prepend=True)
                try:
                    with ro.blocks(ro.block_ids([P], [P], P, dev)):
                        model(input_ids=torch.tensor([prefix_ids], device=dev),
                              attention_mask=torch.ones((1, P), dtype=torch.long, device=dev),
                              use_cache=True, return_dict=True, logits_to_keep=1)
                    raise RuntimeError("forward did not stop")
                except Stop:
                    pass
                finally:
                    if src is not None:
                        h.remove()
                assert state.get("x") is None, f"layer {L}: input not consumed"
                print(f"layer {L}: done", flush=True)
    finally:
        for h in handles:
            h.remove()
    from safetensors.torch import save_file
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out["prefix_rows"] = torch.tensor(rows, dtype=torch.int64)
    save_file({k: v.contiguous() for k, v in out.items()}, str(args.out),
              metadata={"dtype": args.dtype, "sys": str(args.sys), "P": str(P),
                        "provenance": json.dumps(qevref.provenance())})
    print(f"wrote {args.out}: {len(out)} tensors, P={P}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True, choices=sorted(RUNS))
    d.add_argument("--only", default=None)
    d.add_argument("--force", action="store_true")
    f = sub.add_parser("forced")
    f.add_argument("--dtype", required=True, choices=["float32", "bfloat16"])
    f.add_argument("--sys", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--layers", default=None)
    f.add_argument("--stride", type=int, default=1, help="record prefix rows 0, s, 2s, ...")
    args = ap.parse_args()
    {"dump": cmd_dump, "forced": cmd_forced}[args.cmd](args)


if __name__ == "__main__":
    main()
