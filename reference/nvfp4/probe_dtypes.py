"""Print the runtime dtypes of the NVFP4 MLP parameters exactly as the checkpoint's evaluation loaded them.

transformers 5.19 + compressed-tensors 0.20.1, `dtype=torch.bfloat16`, default CompressedTensorsConfig
(dequantize at load, i.e. `run_compressed=False`). Only layer 0 is instantiated, so this needs ~1 GB.

    venv/bin/python -I reference/nvfp4/probe_dtypes.py [checkpoint]
"""
import sys

import torch
import transformers

path = sys.argv[1] if len(sys.argv) > 1 else "/home/velvet/Desktop/hf_models/elephant-1-nvfp4-mlp"
# As OneJev's qev.engine / train.common.load_bundle: Qwen3_5ForCausalLM on the text config.
cfg = transformers.AutoConfig.from_pretrained(path).get_text_config()
print("text config has quantization_config:", getattr(cfg, "quantization_config", None) is not None)
cfg.num_hidden_layers = 1
cfg.layer_types = cfg.layer_types[:1]
# The checkpoint's evaluation: run_compressed=False (= dequantize=True in transformers 5.19).
qc = transformers.CompressedTensorsConfig(run_compressed=False)
model = transformers.Qwen3_5ForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16,
                                                        device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
                                                        quantization_config=qc)
mlp = model.model.layers[0].mlp
for name in ["gate_proj", "up_proj", "down_proj"]:
    mod = getattr(mlp, name)
    print(name, "status:", getattr(mod, "quantization_status", None))
    print("   scheme:", getattr(mod, "quantization_scheme", None))
    for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
        print("  ", n, t.dtype, tuple(t.shape), t.flatten()[:2].tolist() if t.numel() < 3 else "")

# The first forward decompresses (dequantize-on-first-forward hook); show the state the evaluation ran in.
x = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda:0")
with torch.no_grad():
    y = mlp(x)
print("after first forward: out", y.dtype)
for name in ["gate_proj", "up_proj", "down_proj"]:
    mod = getattr(mlp, name)
    print(name, "status:", getattr(mod, "quantization_status", None), "forward:", mod.forward)
    for n, t in list(mod.named_parameters()) + list(mod.named_buffers()):
        print("  ", n, t.dtype, tuple(t.shape), t.flatten()[:2].tolist() if t.numel() < 3 else "")

# Check the formulas written in src/kernels/nvfp4/mod.rs against the live modules, bit for bit.
from safetensors import safe_open  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_weight(packed, scale_e4m3, wgs):
    n, kh = packed.shape
    nib = torch.stack(((packed & 0xF), (packed >> 4)), dim=-1).reshape(n, 2 * kh).long()
    q = E2M1[nib & 7] * torch.where((nib & 8) != 0, -1.0, 1.0)            # exact
    s = scale_e4m3.float().repeat_interleave(16, dim=1) / wgs              # f32 division
    return (q * s).to(torch.bfloat16)                                      # f32 product, one bf16 rounding


def qdq_act(x, igs):
    xg = x.unflatten(-1, (-1, 16))                                         # bf16
    amax = xg.float().abs().amax(-1, keepdim=True)
    s = (amax.to(torch.bfloat16) / torch.tensor(6.0, dtype=torch.bfloat16)) # bf16 division (f32 then RNE to bf16)
    s = (igs * s.float()).clamp(-448, 448).to(torch.float8_e4m3fn).float() # f32 product, RNE to e4m3
    s = torch.where(s == 0, torch.tensor(0.125), s)
    eff = s / igs                                                          # f32
    y = (xg.float() / eff).clamp(-6, 6)                                    # f32 division
    a = y.abs()
    q = torch.zeros_like(a)
    for lo, hi_incl, v in [(0.25, False, 0.5), (0.75, True, 1.0), (1.25, False, 1.5), (1.75, True, 2.0),
                           (2.5, False, 3.0), (3.5, True, 4.0), (5.0, False, 6.0)]:
        q = torch.where(a >= lo if hi_incl else a > lo, torch.tensor(v), q)
    q = q * torch.sign(y)
    return (q * eff).to(torch.bfloat16).flatten(-2)


f = safe_open(f"{path}/model_backbone.safetensors", "pt")
pre = "model.language_model.layers.0.mlp."
xt = (torch.randn(64, 4096) * 3).to(torch.bfloat16)
for name in ["gate_proj", "down_proj"]:
    mod = getattr(mlp, name)
    g = lambda k: f.get_tensor(pre + name + "." + k)  # noqa: E731
    w = dequant_weight(g("weight_packed"), g("weight_scale"), g("weight_global_scale"))
    print(name, "dequant weight == module weight:", torch.equal(w, mod.weight.cpu()))
    xin = xt if name == "gate_proj" else (torch.randn(64, 12288) * 0.3).to(torch.bfloat16)
    xq = qdq_act(xin, g("input_global_scale"))
    with torch.no_grad():
        y_mod = mod(xin.cuda()).cpu()
        y_lin = torch.nn.functional.linear(xq.cuda(), mod.weight).cpu()
        y_raw = torch.nn.functional.linear(xin.cuda(), mod.weight).cpu()
    print(name, "module == linear(qdq_act(x), W):", torch.equal(y_mod, y_lin),
          "| module == linear(x, W) (no act quant):", torch.equal(y_mod, y_raw))
