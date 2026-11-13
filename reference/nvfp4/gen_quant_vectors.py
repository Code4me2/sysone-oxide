"""Test vectors for `quantize_nvfp4`, produced by compressed-tensors 0.20.1 itself (not by a re-implementation).

For each case: bf16 activations x [T, K] and a static input_global_scale `igs`. The reference runs the exact
activation path of a DECOMPRESSED NVFP4 Linear (`forward_quantize` with `dynamic: local`):
`compute_dynamic_scales_and_zp` → scale [T, K/16]; `quantize` → e2m1 values; `pack_fp4_to_uint8` → bytes;
`fake_quantize` → the bf16 activation the reference GEMM consumes.

Output: reference/nvfp4/vectors/quant.json (index) + quant.bin (raw little-endian arrays).

    CT_ENFORCE_EAGER=1 venv/bin/python -I reference/nvfp4/gen_quant_vectors.py

CT_ENFORCE_EAGER selects compressed-tensors' pure-torch implementations (its Triton backend cannot build on this box:
no Python.h). The two backends differ only in the sign of a zero e2m1 result for an input of exactly -0.0: the eager
`quantize` adds the (zero) zero-point before rounding, which yields +0 as in the forward path.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CT_ENFORCE_EAGER", "1")

import torch  # noqa: E402
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8  # noqa: E402
from compressed_tensors.quantization import QuantizationArgs  # noqa: E402
from compressed_tensors.quantization.lifecycle.forward import fake_quantize, quantize  # noqa: E402
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp  # noqa: E402

# The checkpoint's input_activations args (config.json quantization_config.config_groups.group_0).
ARGS = QuantizationArgs(num_bits=4, type="float", symmetric=True, group_size=16, strategy="tensor_group",
                        dynamic="local", scale_dtype=torch.float8_e4m3fn, observer="static_minmax")


def reference(x: torch.Tensor, igs: float):
    gs = torch.tensor([igs], dtype=torch.float32)
    scale, zp = compute_dynamic_scales_and_zp(value=x, args=ARGS, module=None, global_scale=gs)
    q = quantize(x=x, scale=scale, zero_point=zp, args=ARGS, global_scale=gs)
    packed = pack_fp4_to_uint8(q.to(torch.float32))
    qdq = fake_quantize(x=x, scale=scale, zero_point=zp, args=ARGS, global_scale=gs)
    assert qdq.dtype == torch.bfloat16 and scale.dtype == torch.float32, (qdq.dtype, scale.dtype)
    s8 = scale.to(torch.float8_e4m3fn)
    assert torch.equal(s8.float(), scale), "scale is not e4m3-representable"
    return packed, s8.view(torch.uint8), qdq


def case_random(g, t, k, spread):
    """Normal values with a per-group log-uniform magnitude in 2^[-spread, spread]."""
    x = torch.randn(t, k, generator=g)
    mag = torch.exp2((torch.rand(t, k // 16, 1, generator=g) * 2 - 1) * spread)
    return (x.view(t, k // 16, 16) * mag).view(t, k).to(torch.bfloat16)


def case_ties(igs_log2):
    """Groups whose quotients x / eff land exactly on every e2m1 rounding threshold, both signs.

    With amax = 6 * 2^e and igs = 2^m the scale chain is exact: s = 2^e, s8 = 2^(e+m), eff = 2^e."""
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    rows = []
    for e in range(-6, 7, 2):
        if not (-9 <= e + igs_log2 <= 8):
            continue
        base = 2.0 ** e
        for sign in (1.0, -1.0):
            vals = [6.0] + [sign * t for t in ties] + [sign * (t + 2 ** -6) for t in ties] + [-0.0]
            vals = vals[:16]
            vals += [0.0] * (16 - len(vals))
            rows.append([v * base for v in vals] * 16)  # 256 columns
    return torch.tensor(rows, dtype=torch.float32).to(torch.bfloat16)


def case_special(k):
    """Zeros, -0.0, bf16 extremes, subnormals, inf, and groups whose scale underflows or saturates."""
    bf = torch.finfo(torch.bfloat16)
    rows = []
    rows.append([0.0] * k)
    rows.append([-0.0] * k)
    r = [0.0] * k
    for i in range(0, k, 16):
        r[i] = bf.max if (i // 16) % 2 == 0 else -bf.max
        r[i + 1] = 1.0
    rows.append(r)
    r = [0.0] * k
    for i in range(k):
        r[i] = (bf.smallest_normal * 2 ** -(i % 7)) * (-1 if i % 3 == 0 else 1)  # incl. subnormals
    rows.append(r)
    r = [(i % 13 - 6) * 1e-3 for i in range(k)]
    r[5] = float("inf")
    r[40] = float("-inf")
    rows.append(r)
    r = [(i % 16 - 8) * 2.0 ** -(20 + (i // 16) % 30) for i in range(k)]  # tiny groups: s8 underflow -> 0.125
    rows.append(r)
    r = [(i % 16 - 8) * 2.0 ** ((i // 16) % 40) for i in range(k)]  # big groups: igs*s clamps at 448
    rows.append(r)
    return torch.tensor(rows, dtype=torch.float32).to(torch.bfloat16)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "vectors")
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(1234)
    cases = [
        ("random_gateup_igs628", case_random(g, 32, 4096, 3.0), 628.0),
        ("random_down_igs692", case_random(g, 8, 12288, 3.0), 692.0),
        ("random_wide_range_igs628", case_random(g, 32, 1024, 40.0), 628.0),
        ("random_igs1", case_random(g, 16, 512, 6.0), 1.0),
        ("random_igs_huge", case_random(g, 16, 512, 6.0), 2.0 ** 30),
        ("random_igs_tiny", case_random(g, 16, 512, 6.0), 1e-3),
        ("ties_igs1", case_ties(0), 1.0),
        ("ties_igs256", case_ties(8), 256.0),
        ("special_igs628", case_special(256), 628.0),
        ("special_igs1", case_special(256), 1.0),
    ]
    index, blob = [], bytearray()

    def put(t: torch.Tensor):
        off = len(blob)
        blob.extend(t.contiguous().numpy().tobytes())
        return off

    for name, x, igs in cases:
        packed, s8, qdq = reference(x, igs)
        t, k = x.shape
        index.append({
            "name": name, "t": t, "k": k, "igs": igs,
            "x_bf16": put(x.view(torch.int16)), "packed_u8": put(packed), "scales_u8": put(s8),
            "qdq_bf16": put(qdq.view(torch.int16)),
        })
        print(f"{name:<28} T={t:<4} K={k:<6} igs={igs:<12g} 0.125-scale groups={(s8 == 0x20).sum().item()}")
    (out / "quant.bin").write_bytes(bytes(blob))
    (out / "quant.json").write_text(json.dumps({"generator": "compressed-tensors 0.20.1 (eager)",
                                                "torch": torch.__version__, "cases": index}, indent=1))
    print(f"wrote {out}/quant.bin ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
