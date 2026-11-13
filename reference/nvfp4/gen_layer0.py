"""Layer-0 MLP reference for `tests/kernels_nvfp4.rs` (gemm_layer0_vs_compressed_tensors_qdq).

Loads elephant-1-nvfp4-mlp exactly as its evaluation did (transformers 5.19 Qwen3_5ForCausalLM, bf16,
CompressedTensorsConfig(run_compressed=False); only layer 0 is instantiated) and runs its MLP projections, which are
compressed-tensors DECOMPRESSED modules: bf16 weights + activation fake-quant (QDQ), bf16 GEMM on GPU, TF32 off.

Saved (reference/nvfp4/data/layer0.{json,bin}, not committed):
- x              bf16 [T, 4096]   random MLP input
- gateup_ref     bf16 [T, 24576]  cat(gate_proj(x), up_proj(x))       (the compressed-tensors QDQ path)
- m              bf16 [T, 12288]  silu(gate) * up, as Qwen3_5MLP computes it
- down_ref       bf16 [T, 4096]   down_proj(m)
- *_exact_f32    f32              f64 matmul of the exactly dequantized operands (e2m1 · e4m3 / global scale, no bf16
                                  rounding of operands), i.e. the value the block-scaled kernel approximates

    CT_ENFORCE_EAGER=1 venv/bin/python -I reference/nvfp4/gen_layer0.py [T]
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CT_ENFORCE_EAGER", "1")

import torch  # noqa: E402
import transformers  # noqa: E402
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8  # noqa: E402
from compressed_tensors.quantization.lifecycle.forward import quantize  # noqa: E402
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp  # noqa: E402
from safetensors import safe_open  # noqa: E402

CKPT = "/home/velvet/Desktop/hf_models/elephant-1-nvfp4-mlp"
T = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = Path(__file__).parent / "data"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

cfg = transformers.AutoConfig.from_pretrained(CKPT).get_text_config()
cfg.num_hidden_layers = 1
cfg.layer_types = cfg.layer_types[:1]
model = transformers.Qwen3_5ForCausalLM.from_pretrained(
    CKPT, config=cfg, dtype=torch.bfloat16, device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
    quantization_config=transformers.CompressedTensorsConfig(run_compressed=False))
mlp = model.model.layers[0].mlp
f = safe_open(f"{CKPT}/model_backbone.safetensors", "pt")
pre = "model.language_model.layers.0.mlp."


def exact(mod, name, x):
    """f64 (dequantized activation) @ (dequantized weight)^T using compressed-tensors' own quantize / unpack."""
    args = mod.quantization_scheme.input_activations
    igs = mod.input_global_scale.detach().float().cpu()
    scale, zp = compute_dynamic_scales_and_zp(value=x.cpu(), args=args, module=mod, global_scale=igs)
    q = quantize(x=x.cpu(), scale=scale, zero_point=zp, args=args, global_scale=igs).double()
    xd = (q.unflatten(-1, (-1, 16)) * scale.double().unsqueeze(-1)).flatten(-2) / igs.double()
    packed = f.get_tensor(pre + name + ".weight_packed")
    n, kh = packed.shape
    wq = unpack_fp4_from_uint8(packed, n, 2 * kh, dtype=torch.float32).double()
    ws = f.get_tensor(pre + name + ".weight_scale").double().repeat_interleave(16, dim=1)
    wgs = f.get_tensor(pre + name + ".weight_global_scale").double()
    wd = wq * ws / wgs
    return (xd.cuda() @ wd.cuda().T).float().cpu()


g = torch.Generator().manual_seed(7)
x = torch.randn(T, 4096, generator=g).to(torch.bfloat16)
with torch.no_grad():
    xc = x.cuda()
    gate = mlp.gate_proj(xc)
    up = mlp.up_proj(xc)
    m = mlp.act_fn(gate) * up
    down = mlp.down_proj(m)
gateup_exact = torch.cat([exact(mlp.gate_proj, "gate_proj", x), exact(mlp.up_proj, "up_proj", x)], dim=1)
down_exact = exact(mlp.down_proj, "down_proj", m.cpu())

arrays = {
    "x": x.view(torch.int16),
    "gateup_ref": torch.cat([gate, up], dim=1).cpu().view(torch.int16),
    "gateup_exact_f32": gateup_exact,
    "m": m.cpu().view(torch.int16),
    "down_ref": down.cpu().view(torch.int16),
    "down_exact_f32": down_exact,
}
OUT.mkdir(parents=True, exist_ok=True)
index, blob = {}, bytearray()
for k, v in arrays.items():
    index[k] = (len(blob), v.numel())
    blob.extend(v.contiguous().numpy().tobytes())
(OUT / "layer0.bin").write_bytes(bytes(blob))
(OUT / "layer0.json").write_text(json.dumps({"checkpoint": CKPT, "t": T, "torch": torch.__version__,
                                             "arrays": index}, indent=1))
for k, v in arrays.items():
    print(k, tuple(v.shape))
print("ref vs exact (gate/up) max |d| / rms:",
      ((torch.cat([gate, up], 1).float().cpu() - gateup_exact).abs().max() / gateup_exact.pow(2).mean().sqrt()).item())
