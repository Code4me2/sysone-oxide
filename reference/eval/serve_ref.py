"""Serve the patched qev reference over HTTP (qev's own FastAPI app, `qev.server.create_app`), for score.py.

Why not `qev serve`: for elephant-1 (a config with `vision_config`) `qev serve` builds `MMDecisionEngine`, which
serve.patch does not patch and which lacks the block-bidirectional mask (DESIGN §1, 2026-11-13 row). This serves
the text `DecisionEngine` the way the reference harness loads it (qevref.load_engine: TF32 off, fp32 head; NVFP4 as
compressed-tensors QDQ), optionally on the per-question path. GPU: run it through scripts/gpu.sh.

    scripts/gpu.sh venv/bin/python -I reference/eval/serve_ref.py --port 8765 [--nvfp4] [--path batched]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import qevref  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--nvfp4", action="store_true")
    ap.add_argument("--path", default="batched",
                    choices=["batched", "sequential", "per_question", "per_question_batched"])
    args = ap.parse_args()

    import uvicorn
    from qev.server import create_app

    model = qevref.NVFP4_MODEL if args.nvfp4 else qevref.BF16_MODEL
    fork = args.path if args.path in ("batched", "sequential") else "sequential"
    engine = qevref.load_engine(model, "bfloat16", fork_mode=fork, nvfp4=args.nvfp4)
    if args.path == "per_question":
        qevref.per_question_mode(engine)
    elif args.path == "per_question_batched":
        import run_eval
        run_eval.per_question_batched_mode(engine)
    uvicorn.run(create_app(engine, model_name=model.name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
