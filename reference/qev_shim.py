"""A weightless qev DecisionEngine: the reference's own frontend code paths, with no model behind them.

`ShimEngine` is a `qev.engine.DecisionEngine` built without `__init__`, so nothing is loaded but the tokenizer,
the model config and the readout config. It sets exactly the attributes `__init__` sets (minus the model), so every
frontend method of the reference runs unchanged:

- `decide()`: branch planning (debias shifts, `_plan_chunks` + the "none of the options" slot), owners, limits;
- `_prefix_and_suffix_ids` (marker render + longest common token prefix) and serve.patch's `_split`
  (the noncausal prefix-block check and option token spans);
- `_render_prompt`, `_verify_slots`.

The only replaced methods are the two forwards (`_run_sequential`, `_run_batched`). The replacement reads the
plan out of the calling `decide()` frame, runs serve.patch's `_split` (which the real forwards call first), and
returns zero logits so `decide()` can finish and the server can build its response. Answers built from those
logits are meaningless and are never recorded.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import transformers
from qev import readout as ro
from qev.calibrate import Calibration
from qev.engine import BranchResult, DecisionEngine, EngineInfo
from qev.prompt import DEFAULT_STYLE

_DECIDE_CODE = DecisionEngine.decide.__code__
# Locals of `decide()` that make up the plan; they are all bound before the forward is called.
PLAN_LOCALS = ("rendered", "owners", "prefix_ids", "suffixes", "longest", "total")


class ShimEngine(DecisionEngine):
    """DecisionEngine without weights. After each `decide()` (successful or not) `self.last` holds the plan."""

    def __init__(self, model_path: str, max_branch_tokens: int = 32768, max_request_tokens: int = 65536) -> None:
        # Mirrors DecisionEngine.__init__ (OneJev@81ce62f + serve.patch) except the model load, the head upcast and
        # the fork-mode probe, which need weights.
        self.model_path = model_path
        self.device = "cpu"
        self.torch_dtype = None
        self.calibration = Calibration()
        self.max_branch_tokens = max_branch_tokens
        self.max_request_tokens = max_request_tokens
        self.fork_token_budget = 200_000
        self.fork_tolerance = 5e-2
        self.style = DEFAULT_STYLE
        local = Path(model_path).exists()
        common = {"local_files_only": local, "trust_remote_code": False}
        config = transformers.AutoConfig.from_pretrained(model_path, **common)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, **common)
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            config = config.get_text_config()
        self.model = None
        self.head_dtype = "float32"
        self.model_type = config.model_type
        self.template_kwargs = {"enable_thinking": False} if "qwen" in self.model_type else {}
        self.slot_ids = self._verify_slots()
        rcfg = ro.ReadoutConfig.load(model_path) if local else ro.ReadoutConfig()
        if rcfg.head:
            raise ValueError("the shim covers slot_lm checkpoints only (a head readout needs its weights)")
        # Readout.attach would install the noncausal mask on the model; there is none, and _split only reads config.
        self.readout = ro.Readout(rcfg)
        if (Path(model_path) / "calibration.json").exists():
            self.calibration = Calibration.load(Path(model_path) / "calibration.json")
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.info = EngineInfo(model_path=model_path, model_type=self.model_type, dtype="none", device="cpu",
                               fork_mode="sequential", fork_max_abs_diff=None,
                               transformers_version=transformers.__version__, torch_version="",
                               extra={"head_dtype": self.head_dtype, "attention": rcfg.attention,
                                      "readout": rcfg.readout, "calibration": self.calibration.source})
        self.last: dict[str, Any] | None = None

    def decide(self, request, debias: int = 1, debug: bool = False):
        self.last = None
        try:
            return DecisionEngine.decide(self, request, debias=debias, debug=debug)
        except Exception as exc:
            if self.last is None:  # raised inside decide() before the forward: recover its plan from the traceback
                frame = _decide_frame_from_tb(exc.__traceback__)
                if frame is not None:
                    self.last = _plan(frame.f_locals)
            if self.last is not None:
                self.last["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def _forward(self, state: Any, rendered) -> list[BranchResult]:
        frame = sys._getframe(2)
        if frame.f_code is not _DECIDE_CODE:
            raise RuntimeError("the shim forward must be called from DecisionEngine.decide")
        self.last = _plan(frame.f_locals)
        prefix_ids, suffixes, spans = self._split(state, rendered)  # what the real forwards do first
        self.last["split"] = {"prefix_ids": prefix_ids, "suffixes": suffixes, "spans": spans}
        return [BranchResult(r.qid, r.kind, r.labels, r.descriptions, [0.0] * r.n_slots, 0.0, len(s))
                for r, s in zip(rendered, suffixes)]

    def _run_sequential(self, state: Any, rendered) -> list[BranchResult]:
        return self._forward(state, rendered)

    def _run_batched(self, state: Any, rendered) -> list[BranchResult]:
        return self._forward(state, rendered)


def _plan(f_locals: dict[str, Any]) -> dict[str, Any]:
    return {k: f_locals.get(k) for k in PLAN_LOCALS}


def _decide_frame_from_tb(tb: types.TracebackType | None):
    found = None
    while tb is not None:
        if tb.tb_frame.f_code is _DECIDE_CODE:
            found = tb.tb_frame
        tb = tb.tb_next
    return found
