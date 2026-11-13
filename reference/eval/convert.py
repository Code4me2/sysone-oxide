"""Convert the downloaded BANKING77 / LEDGAR test parquet files to JSONL with their label names (run by fetch.sh in
the pyarrow tool venv). Output: <data>/banking77_test.jsonl, <data>/ledgar_test.jsonl and <data>/labels.json.

Each JSONL line is {"index": i, "text": ..., "label": int}; `index` is the row number in the parquet test split.
Label names come from the parquet's embedded `huggingface` features metadata (ClassLabel names, in id order).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SOURCES = {
    "banking77": "banking77/data/test-00000-of-00001.parquet",
    "ledgar": "lex_glue/ledgar/test-00000-of-00001.parquet",
}


def label_names(path: Path) -> list[str]:
    meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    return list(names)


def main() -> None:
    data = Path(sys.argv[1])
    labels = {}
    for name, rel in SOURCES.items():
        path = data / rel
        table = pq.read_table(path)
        labels[name] = label_names(path)
        with open(data / f"{name}_test.jsonl", "w", encoding="utf-8") as f:
            for i, row in enumerate(table.to_pylist()):
                assert set(row) == {"text", "label"}, row.keys()
                assert 0 <= row["label"] < len(labels[name])
                f.write(json.dumps({"index": i, "text": row["text"], "label": row["label"]}, ensure_ascii=False) + "\n")
        print(f"{name}: {table.num_rows} rows, {len(labels[name])} labels")
    (data / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
