"""Manifest of the binary dev goldens kept outside the repo (no git-lfs): sha256, size, tensor names, shapes,
dtypes and provenance of every safetensors file under the golden binary directory. The manifest is committed at
tests/golden/dev/model/manifest.json; tests that need a binary golden look it up there and verify its sha256.

    venv/bin/python -I reference/write_manifest.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qevref  # noqa: E402

MANIFEST = qevref.REPO / "tests" / "golden" / "dev" / "model" / "manifest.json"


def describe(path: Path) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        tensors = {}
        for k in sorted(f.keys()):
            sl = f.get_slice(k)
            tensors[k] = {"dtype": sl.get_dtype(), "shape": list(sl.get_shape())}
    return {"sha256": qevref.sha256_file(path), "bytes": path.stat().st_size, "metadata": meta, "tensors": tensors}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the files against the committed manifest")
    args = ap.parse_args()
    root = qevref.GOLDEN_BIN
    if args.check:
        man = json.loads(MANIFEST.read_text())
        bad = [rel for rel, e in man["files"].items() if qevref.sha256_file(root / rel) != e["sha256"]]
        print(f"{len(man['files']) - len(bad)}/{len(man['files'])} files match" + (f"; mismatched: {bad}" if bad else ""))
        sys.exit(1 if bad else 0)
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((MANIFEST.parent / "runs").glob("*.json"))}
    files = {}
    for path in sorted(root.glob("*/*.safetensors")):
        rel = str(path.relative_to(root))
        entry = describe(path)
        run = runs.get(path.parent.name)
        if run:
            entry["run"] = {"name": run["name"], **run["config"]}
        files[rel] = entry
    out = {
        "root": str(root),
        "about": "Binary dev goldens (per-layer hidden states) of the W1b runs; see reference/layer_dump.py for the "
                 "layout. Hidden states are f32 upcasts of the model dtype's values.",
        "provenance": next(iter(runs.values()))["provenance"] if runs else {},
        "files": files,
    }
    MANIFEST.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST}: {len(files)} files, {sum(f['bytes'] for f in files.values()) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
