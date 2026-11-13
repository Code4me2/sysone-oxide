"""Shared helpers for the sysone model-side reference harness (DESIGN.md §9.1).

Loads the patched OneJev `qev` DecisionEngine (OneJev @81ce62f + elephant-1's serve.patch) in bf16 or fp32, or the
NVFP4 W4A4-MLP checkpoint as QDQ simulation (compressed-tensors `run_compressed=False`), with TF32 off. Every run
records its provenance next to its outputs.

Run with the reference venv, isolated from the working directory:
    /home/velvet/Desktop/sysone-ref/venv/bin/python -I reference/<script>.py ...
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

REF = Path(os.environ.get("SYSONE_REF", "/home/velvet/Desktop/sysone-ref"))
ONEJEV = REF / "OneJev-patched"
MODELS = Path(os.environ.get("SYSONE_MODELS", "/home/velvet/Desktop/hf_models"))
BF16_MODEL = MODELS / "elephant-1"
NVFP4_MODEL = MODELS / "elephant-1-nvfp4-mlp"
GOLDEN_BIN = REF / "golden" / "dev"
REPO = Path(__file__).resolve().parent.parent
LAYERS = (0, 3, 4, 15, 31)
LEVELS = (*(f"layer{i}" for i in LAYERS), "norm")

if str(ONEJEV) not in sys.path:
    sys.path.insert(0, str(ONEJEV))

# Triton (compressed-tensors' NVFP4 QDQ kernels) compiles a small C launcher against Python.h. DGX OS has no
# python3.12-dev and we have no root, so its headers are extracted (dpkg-deb -x of the matching 3.12.3 debs) here.
PYDEV = REF / "pydev" / "usr" / "include"
if not Path("/usr/include/python3.12/Python.h").exists() and (PYDEV / "python3.12" / "Python.h").exists():
    os.environ["C_INCLUDE_PATH"] = os.pathsep.join(
        p for p in (str(PYDEV / "python3.12"), str(PYDEV), os.environ.get("C_INCLUDE_PATH")) if p)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def numerics_off() -> dict:
    """TF32 off everywhere, highest fp32 matmul precision. Returns the flags as recorded provenance."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda.matmul.allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def gdn_backend() -> str:
    import importlib.util

    fla = importlib.util.find_spec("fla") is not None
    conv = importlib.util.find_spec("causal_conv1d") is not None
    return f"fla={'yes' if fla else 'no (torch_chunk_gated_delta_rule)'}, " \
           f"causal_conv1d={'yes' if conv else 'no (torch conv1d)'}"


def provenance(extra: dict | None = None) -> dict:
    import compressed_tensors
    import torch
    import transformers

    patch = BF16_MODEL / "serve.patch"
    return {
        "tier": "dev (GB10)",
        "arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "compressed_tensors": compressed_tensors.__version__,
        "gdn_backend": gdn_backend(),
        "attn_implementation": "sdpa (transformers default)",
        "onejev": "81ce62f1597c91e46767d4d02d6ac2e18534fe94 + serve.patch",
        "serve_patch_sha256": sha256_file(patch) if patch.exists() else None,
        "numerics": numerics_off(),
        **(extra or {}),
    }


def _nvfp4_from_pretrained():
    """Make every `from_pretrained` inside the qev engine load compressed-tensors weights decompressed to bf16 with
    activation QDQ kept (`run_compressed=False`), the way the NVFP4 checkpoint was evaluated (its README)."""
    import transformers
    from transformers import CompressedTensorsConfig

    cls = transformers.Qwen3_5ForCausalLM
    original = cls.from_pretrained.__func__

    def from_pretrained(klass, *args, **kwargs):
        from compressed_tensors.offload.cache.base import OffloadCache
        from compressed_tensors.offload.module import remove_module_offload

        kwargs.setdefault("quantization_config", CompressedTensorsConfig(run_compressed=False))
        model = original(klass, *args, **kwargs)
        # As the checkpoint's evaluation did: materialize the (unquantized) LM head out of compressed-tensors'
        # offload dispatch, so qev's fp32 head upcast replaces a plain parameter.
        head = model.get_output_embeddings()
        if isinstance(head._parameters, OffloadCache):
            remove_module_offload(head, onload_tensors=True)
        return model

    cls.from_pretrained = classmethod(from_pretrained)
    return lambda: setattr(cls, "from_pretrained", classmethod(original))


def load_engine(model: Path, dtype: str, fork_mode: str, nvfp4: bool = False, device: str = "cuda:0"):
    """The patched qev DecisionEngine as the README instantiates it (fp32 LM head by default)."""
    numerics_off()
    from qev.engine import DecisionEngine

    restore = _nvfp4_from_pretrained() if nvfp4 else None
    try:
        engine = DecisionEngine(str(model), device=device, dtype=dtype, fork_mode=fork_mode)
    finally:
        if restore:
            restore()
    if nvfp4:
        check_nvfp4_qdq(engine.model)
    return engine


def check_nvfp4_qdq(model) -> dict:
    """Assert the 96 MLP projections carry activation and weight quantization (QDQ) and nothing else does."""
    quantized, other = [], []
    for name, mod in model.named_modules():
        scheme = getattr(mod, "quantization_scheme", None)
        if scheme is None:
            continue
        (quantized if ".mlp." in name else other).append(name)
        assert scheme.input_activations is not None and scheme.input_activations.dynamic == "local", name
        assert getattr(mod, "quantization_enabled", True), name
    assert len(quantized) == 96 and not other, (len(quantized), other[:4])
    return {"quantized_modules": len(quantized)}


def per_question_mode(engine) -> None:
    """Replace the engine's branch runner with the NVFP4 checkpoint's evaluation path: one independent full forward
    per branch (prefix + suffix), batch 1, use_cache=False, block-bidirectional mask (train.common.batch_scores).
    decide() keeps its rendering, limits and answer assembly unchanged."""
    import torch
    from qev import readout as ro
    from qev.engine import BranchResult

    def run(state, rendered):
        prefix_ids, suffixes, _ = engine._split(state, rendered)
        P = len(prefix_ids)
        out = []
        with torch.inference_mode():
            for r, suf in zip(rendered, suffixes):
                ids = torch.tensor([prefix_ids + suf], device=engine.device)
                n = ids.shape[1]
                with ro.blocks(ro.block_ids([P], [n], n, engine.device)):
                    o = engine.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                     return_dict=True, logits_to_keep=1)
                logits, mass = engine._readout(o.logits[0, -1], r.n_slots)
                out.append(BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, mass, len(suf)))
                del o
        return out

    engine.info.fork_mode = "per_question"
    engine._run_sequential = run
    engine._run_batched = run


def model_files_sha(model: Path) -> dict:
    """sha256 of the weight files, cached next to the golden binaries (hashing 18 GB takes a while)."""
    cache = GOLDEN_BIN / "model_sha256.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    out = {}
    for f in sorted(model.glob("*.safetensors")):
        key = str(f)
        st = f.stat()
        tag = f"{st.st_size}:{int(st.st_mtime)}"
        if known.get(key, {}).get("tag") != tag:
            known[key] = {"tag": tag, "sha256": sha256_file(f)}
        out[f.name] = known[key]["sha256"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known, indent=1))
    return out


def load_corpus(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
