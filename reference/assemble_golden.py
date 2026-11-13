"""Answer-assembly and schema goldens for the W2 frontend tests, from qev's own code (no weights).

    python -I reference/assemble_golden.py [--out tests/golden/dev/assemble] [--model DIR] [--check]

Writes (deterministic; a rerun is byte-identical):
    answers.jsonl   qev's `decide()` answer assembly on synthetic slot logits. Each record names a corpus request
                    (reference/corpus/requests.jsonl) or an engine_direct one, the debias, an optional calibration
                    table, the logits and allowed-token masses fed to each branch, and what `decide()` returned:
                    the answers and the debug diagnostics, as Starlette's JSONResponse bytes (`json.dumps(...,
                    ensure_ascii=False, allow_nan=False, separators=(",", ":"))`).
    schema.jsonl    extra validation cases POSTed to qev's server: the exact 422/404 response bodies.
    pyfloat.json    Python float built-ins the frontend reimplements: repr, round(x, 6) and sum().
    provenance.json versions, seed, checkpoint file hashes and per-file sha256 (no timestamps).

The engine is reference/qev_shim.py's ShimEngine with its forward replaced by one that returns the synthetic logits
(after running serve.patch's `_split`, as the real forward does). Everything after the forward is qev's code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

DEFAULT_MODEL = "/home/velvet/Desktop/hf_models/elephant-1"
SERVED_NAME = "elephant-1"
SEED = 20261113


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def make_logits(rng: random.Random, mode: str, k: int) -> list[float]:
    if mode == "normal":
        return [f32(rng.gauss(0.0, 3.0)) for _ in range(k)]
    if mode == "large":
        return [f32(rng.uniform(15.0, 45.0)) for _ in range(k)]
    if mode == "ties":
        return [f32(2.5)] * k
    if mode == "peaked":
        out = [f32(rng.gauss(0.0, 1.0)) for _ in range(k)]
        out[rng.randrange(k)] = f32(60.0)
        return out
    if mode == "wide":
        return [f32(rng.uniform(-400.0, 400.0)) for _ in range(k)]
    if mode == "quantized":  # bf16-like steps of 0.125, many exact ties
        return [f32(round(rng.uniform(20.0, 24.0) * 8) / 8) for _ in range(k)]
    raise ValueError(mode)


MODES = ("normal", "large", "ties", "peaked", "wide", "quantized")

CALIBRATIONS = {
    "none": {},
    "kinds": {"noul": 0.8, "choice": 1.7, "score": 2.5},
    "per_k": {"choice": 1.3, "choice:3": 0.6, "choice:151": 2.0, "choice:256": 0.9, "noul:2": 1.9, "score:4": 0.7},
}


def build_engine(model: str):
    from qev.engine import BranchResult
    from qev_shim import ShimEngine

    class SyntheticEngine(ShimEngine):
        """The shim with a forward that returns `self.feed(rendered)` as slot logits and masses."""

        feed = None

        def _forward(self, state, rendered):
            prefix_ids, suffixes, spans = self._split(state, rendered)
            logits, masses = self.feed(rendered)
            self.fed = (logits, masses)
            return [BranchResult(r.qid, r.kind, r.labels, r.descriptions, list(l), m, len(s))
                    for r, s, l, m in zip(rendered, suffixes, logits, masses)]

    return SyntheticEngine(model)


def answer_cases(engine, corpus: list[dict]) -> list[str]:
    from qev.calibrate import Calibration
    from qev.schema import ChoiceQuestion, SystemOneRequest

    rng = random.Random(SEED)
    picked = []
    for case in corpus:
        if case["group"] in ("invalid", "limits", "long"):
            continue
        if case.get("engine_direct"):
            picked.append(case)
            continue
        size = len(json.dumps(case["body"]))
        if size < 20000 or case["group"] == "choice_many":
            picked.append(case)
    # Chunking is unreachable through the schema with 256 slots, so also replay choices with more options than a
    # smaller slot count (as a tokenizer that splits the two-letter labels would have, or smaller still).
    def n_options(case):
        return max((len(q.get("criteria") or {}) for q in case["body"]["questions"].values()
                    if isinstance(q, dict) and q.get("type") == "choice"), default=0)

    runs = [(case, 256) for case in picked]
    runs += [(case, 26) for case in picked if n_options(case) > 26]
    runs += [(case, 4) for case in picked if 4 < n_options(case) <= 40]
    out = []
    for n, (case, max_slots) in enumerate(runs):
        engine.max_slots = max_slots
        body = case["body"]
        if case.get("engine_direct"):
            qs = {qid: ChoiceQuestion.model_construct(**q) for qid, q in body["questions"].items()}
            req = SystemOneRequest.model_construct(state=body.get("state"), questions=qs)
        else:
            try:
                req = SystemOneRequest.model_validate(body)
            except Exception:
                continue
            if req.model not in (None, "", "jev-latest", "jev-preview", "qev-latest", SERVED_NAME):
                continue
        mode = MODES[n % len(MODES)]
        calib = list(CALIBRATIONS)[(n // len(MODES)) % len(CALIBRATIONS)]
        debias = case["debias"] if n % 4 else 2 + n % 3
        with_mass = n % 2 == 0

        def feed(rendered, mode=mode, with_mass=with_mass):
            logits = [make_logits(rng, mode, r.n_slots) for r in rendered]
            masses = [rng.random() if with_mass else 0.0 for _ in rendered]
            return logits, masses

        engine.feed = feed
        engine.calibration = Calibration(temperatures=dict(CALIBRATIONS[calib]), source=calib)
        try:
            resp, meta = engine.decide(req, debias=debias, debug=True)
        except ValueError as exc:  # hostile prefix-split case
            out.append(dumps({"id": f"answers-{n:03d}", "case": case["id"], "engine_direct": bool(case.get("engine_direct")),
                              "max_slots": max_slots, "debias": debias, "calibration": CALIBRATIONS[calib],
                              "error": str(exc)}))
            continue
        except KeyError:  # an option literally named like the chunk none-slot: qev fails with a 500
            continue
        logits, masses = engine.fed
        payload = resp.model_dump()
        payload["model"] = SERVED_NAME
        diag = meta["diagnostics"]
        if not with_mass:
            for d in diag.values():
                d.pop("allowed_token_mass")
        out.append(dumps({
            "id": f"answers-{n:03d}", "case": case["id"], "engine_direct": bool(case.get("engine_direct")),
            "max_slots": max_slots, "debias": debias, "calibration": CALIBRATIONS[calib], "mode": mode,
            "logits": logits, "masses": masses if with_mass else None,
            "answers_json": dumps(payload["answers"]), "usage": payload["usage"],
            "diagnostics_json": dumps(diag),
        }))
    engine.max_slots = 256
    return out


SCHEMA_BODIES = [
    "null", "[]", "\"x\"", "{}", "{\"questions\": null}", "{\"questions\": []}", "{\"questions\": {\"q\": null}}",
    "{\"questions\": {\"q\": \"noul\"}}", "{\"questions\": {\"q\": []}}", "{\"questions\": {\"q\": {}}}",
    "{\"questions\": {\"q\": {\"type\": 1}}}", "{\"questions\": {\"q\": {\"type\": \"NOUL\"}}}",
    "{\"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": true}}}",
    "{\"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": 1.5}}}",
    "{\"questions\": {\"q\": {\"type\": \"noul\", \"criteria\": {\"true\": 1, \"false\": [1]}}}}",
    "{\"questions\": {\"q\": {\"type\": \"noul\", \"criteria\": \"yes\"}}}",
    "{\"questions\": {\"q\": {\"type\": \"choice\", \"criteria\": {\"a\": 1, \"b\": false, \"c\": null}}}}",
    "{\"questions\": {\"q\": {\"type\": \"choice\", \"criteria\": []}}}",
    "{\"questions\": {\"q\": {\"type\": \"choice\"}}}",
    "{\"questions\": {\"q\": {\"type\": \"score\", \"criteria\": {}}}}",
    "{\"questions\": {\"q\": {\"type\": \"score\", \"criteria\": [null, 2, \"x\", {\"a\": 1}, 3.5]}}}",
    "{\"questions\": {\"q\": {\"type\": \"score\", \"criteria\": \"abc\"}}}",
    "{\"questions\": {\"a\": {\"type\": \"noul\"}, \"b\": {\"type\": \"bad\"}, \"c\": 5}}",
    "{\"model\": null, \"questions\": {\"q\": {\"type\": \"noul\"}}}",
    "{\"model\": 5, \"questions\": {}}",
    "{\"model\": [], \"questions\": 3}",
    "{\"model\": \"\", \"questions\": {\"q\": {\"type\": \"noul\"}}}",
    "{\"model\": \"x'y\", \"questions\": {\"q\": {\"type\": \"noul\"}}}",
    "{\"model\": \"tab\\there\\u0001\\u00e9\\u200b\\ud83d\\ude00\", \"questions\": {\"q\": {\"type\": \"noul\"}}}",
    "{\"model\": \"a\\\"b'c\", \"questions\": {\"q\": {\"type\": \"noul\"}}}",
    "{\"questions\": {\"q\": {\"type\": \"noul\"}}, \"extra\": NaN}",
    "{\"questions\": {\"q\": {\"type\": \"noul\"}}",
    "{\"questions\": {\"q\": {\"type\": \"noul\",}}}",
    "",
    "{\"questions\": {\"q\": {\"type\": \"noul\"}}} x",
    "{\"questions\": {\"q\": {\"type\": \"noul\", \"criteria\": {\"true\": \"a\"}, \"instructions\": [1, {}]}}, \"state\": 1e400}",
]


def schema_cases(engine) -> list[str]:
    from fastapi.testclient import TestClient
    from qev.server import create_app

    def feed(rendered):
        return [[0.0] * r.n_slots for r in rendered], [0.0] * len(rendered)

    engine.feed = feed
    client = TestClient(create_app(engine, SERVED_NAME))
    out = []
    for i, body in enumerate(SCHEMA_BODIES):
        resp = client.post("/v1/systemone", content=body.encode("utf-8"), headers={"content-type": "application/json"})
        rec = {"id": f"schema-{i:03d}", "body": body, "status": resp.status_code}
        rec["response"] = resp.json() if resp.status_code != 200 else None
        out.append(dumps(rec))
    return out


def pyfloat_cases() -> dict:
    rng = random.Random(SEED + 1)
    xs = [j / 128 for j in range(-130, 260)] + [j / 1024 for j in range(0, 1100, 7)]
    xs += [rng.random() for _ in range(300)] + [rng.uniform(0, 10) for _ in range(100)]
    xs += [rng.random() * 10 ** rng.randint(-12, 20) for _ in range(200)]
    xs += [0.0, -0.0, 5e-324, 1e16, 1e15, 0.0001, 0.00001, 1.7976931348623157e308, 0.1 + 0.2]
    reprs = [[x.hex(), repr(x)] for x in xs]
    rounds = [[x.hex(), repr(round(x, 6))] for x in xs if abs(x) < 1e9]
    sums = []
    for _ in range(200):
        k = rng.randint(1, 40)
        v = [rng.choice([rng.random(), rng.gauss(0, 1e-3), rng.uniform(-1e8, 1e8), 1e-17]) for _ in range(k)]
        sums.append([[x.hex() for x in v], sum(v).hex()])
    return {"repr": reprs, "round6": rounds, "sum": sums}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "tests" / "golden" / "dev" / "assemble"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default=str(HERE / "corpus" / "requests.jsonl"))
    ap.add_argument("--check", action="store_true", help="fail unless a regeneration is byte-identical to --out")
    args = ap.parse_args(argv)

    corpus = [json.loads(l) for l in Path(args.corpus).read_text(encoding="utf-8").split("\n") if l]
    engine = build_engine(args.model)
    files = {
        "answers.jsonl": "".join(l + "\n" for l in answer_cases(engine, corpus)),
        "schema.jsonl": "".join(l + "\n" for l in schema_cases(engine)),
        "pyfloat.json": json.dumps(pyfloat_cases(), indent=1) + "\n",
    }
    import importlib.metadata as md
    import platform

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    model = Path(args.model)
    prov = {
        "generator": "reference/assemble_golden.py", "seed": SEED, "served_model_name": SERVED_NAME,
        "script_sha256": sha(Path(__file__).resolve()), "shim_sha256": sha(HERE / "qev_shim.py"),
        "corpus_sha256": sha(Path(args.corpus)),
        "checkpoint": {f: sha(model / f) for f in ("tokenizer.json", "chat_template.jinja", "config.json", "readout.json")},
        "versions": {"python": platform.python_version(), "machine": platform.machine(),
                     **{p: md.version(p) for p in ("pydantic", "pydantic_core", "fastapi", "starlette", "transformers",
                                                   "tokenizers")}},
        "files": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()},
    }
    files["provenance.json"] = json.dumps(prov, indent=2) + "\n"
    out = Path(args.out)
    if args.check:
        same = all((out / n).read_text(encoding="utf-8") == t for n, t in files.items())
        print(("OK" if same else "MISMATCH") + f": regenerated vs {out}", file=sys.stderr)
        return 0 if same else 1
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"{name}: {text.count(chr(10))} lines, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
