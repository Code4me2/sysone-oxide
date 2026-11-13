"""Error envelope of transformers' own f32 GatedDeltaNet against an f64 recurrence, per prefix.

The GDN tests (tests/kernels_gdn.rs) hold sysone's prefix final states to "twice transformers' own f32 error". This
script measures that error on the exact inputs of the test: run the test with a dump directory, then this script.

    GDN_BACKENDS=tile GDN_DUMP_DIR=/tmp/gdn_dump scripts/gpu.sh cargo test --release --test kernels_gdn \\
        gdn_kernels_match_f64_references_on_a_tree_batch -- --test-threads=1
    scripts/gpu.sh <python with torch> reference/gdn_envelope.py /tmp/gdn_dump [path/to/transformers-5.19.0]

For each prefix segment (zero initial state) it runs `torch_chunk_gated_delta_rule` from transformers 5.19.0
`modeling_qwen3_5.py` in f32 on the GPU (TF32 off, as the reference model runs it), with the GPU conv output as q, k, v
(q, k repeated to the 32 value heads, `use_qk_l2norm_in_kernel=True`) and the dumped g, beta; and
`torch_recurrent_gated_delta_rule` with float32 replaced by float64 as the truth. It prints, per prefix, the maximum
absolute final-state error relative to the f64 state's rms, the same measure the test reports. The functions are taken
from the transformers source file itself (no transformers install needed), so the measurement uses the reference code
verbatim. It also prints a variant that applies each chunk's system through its explicit inverse (as sysone's kernels
do) instead of `solve_triangular`, to separate that design choice from the rest of the arithmetic.

The per-prefix results are hard-coded in the test (`TRANSFORMERS_STATE_ENVELOPE`); re-measure them whenever the test
batch, its weights or the dumped stages change. Measured 2026-11-13 (torch 2.10, GB10): docs/G1.md, "Accuracy".
"""

import ast
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

CONV_DIM, KEY_DIM, NUM_V_HEADS, HEAD_DIM = 8192, 2048, 32, 128


def load_functions(transformers_root: pathlib.Path) -> dict:
    src = (transformers_root / "transformers/models/qwen3_5/modeling_qwen3_5.py").read_text()
    tree = ast.parse(src)
    wanted = {"l2norm", "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"}
    ns: dict = {"torch": torch, "F": F, "is_torchdynamo_exporting": lambda: False}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            code = ast.get_source_segment(src, node)
            code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith("@"))
            if node.name == "torch_recurrent_gated_delta_rule":
                exec(code.replace("torch.float32", "torch.float64").replace(node.name, node.name + "_f64"), ns)
            if node.name == "torch_chunk_gated_delta_rule":
                # Variant: apply the chunk system through its explicit inverse (as sysone's kernels do) instead of
                # a triangular solve, to separate that design choice from the rest of the arithmetic.
                exec(code.replace("torch.linalg.solve_triangular", "_inverse_then_matmul").replace(node.name, node.name + "_explicit_inverse"), ns)
            exec(code, ns)
    def _inverse_then_matmul(a, b, upper, unitriangular):
        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand(a.shape)
        return torch.linalg.solve_triangular(a, eye, upper=upper, unitriangular=unitriangular) @ b

    ns["_inverse_then_matmul"] = _inverse_then_matmul
    missing = wanted - set(ns)
    assert not missing, f"not found in the transformers source: {missing}"
    return ns


def main() -> None:
    dump = pathlib.Path(sys.argv[1])
    root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).resolve().parents[1] / "../sysone-ref/transformers-5.19.0"
    fns = load_functions(root.resolve())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"

    def load(name: str) -> np.ndarray:
        return np.fromfile(dump / f"{name}.f64", dtype="<f8")

    conv = load("conv").reshape(-1, CONV_DIM)
    t = conv.shape[0]
    g = load("g").reshape(t, NUM_V_HEADS)
    beta = load("beta").reshape(t, NUM_V_HEADS)
    segs = [tuple(int(x) for x in line.split()) for line in (dump / "segments.txt").read_text().splitlines() if line.strip()]
    prefix_no = 0
    worst = 0.0
    for start, length, parent in segs:
        if parent >= 0:
            continue
        rows = slice(start, start + length)
        q = conv[rows, :KEY_DIM].reshape(1, length, 16, HEAD_DIM)
        k = conv[rows, KEY_DIM : 2 * KEY_DIM].reshape(1, length, 16, HEAD_DIM)
        v = conv[rows, 2 * KEY_DIM :].reshape(1, length, NUM_V_HEADS, HEAD_DIM)
        q, k = (np.repeat(x, NUM_V_HEADS // 16, axis=2) for x in (q, k))
        gg = g[rows].reshape(1, length, NUM_V_HEADS)
        bb = beta[rows].reshape(1, length, NUM_V_HEADS)

        def run(fn, dtype):
            args = [torch.tensor(x, dtype=dtype, device=dev) for x in (q, k, v, gg, bb)]
            return fn(*args, output_final_state=True, use_qk_l2norm_in_kernel=True)

        _, s32 = run(fns["torch_chunk_gated_delta_rule"], torch.float32)
        _, s32i = run(fns["torch_chunk_gated_delta_rule_explicit_inverse"], torch.float32)
        _, s64 = run(fns["torch_recurrent_gated_delta_rule_f64"], torch.float64)
        rms = s64.pow(2).mean().sqrt().item()
        err = (s32.double() - s64).abs().max().item()
        err_i = (s32i.double() - s64).abs().max().item()
        worst = max(worst, err / rms)
        print(
            f"prefix {prefix_no} (len {length}): transformers f32 final-state max abs {err:.3e}, rms {rms:.3e}, "
            f"{err / rms:.3e} x rms; with an explicit chunk inverse {err_i / rms:.3e} x rms"
        )
        prefix_no += 1
    print(f"worst: {worst:.3e} x rms")


if __name__ == "__main__":
    main()
