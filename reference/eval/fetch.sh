#!/usr/bin/env bash
# Download the three public sources of the labeled eval set (DESIGN §9.2, NVFP4 statistical gate) at pinned
# revisions, check their sha256, and convert the two parquet files to JSONL (Python stdlib only downstream).
#
# Everything lands in $EVAL_DATA (default /home/velvet/Desktop/sysone-ref/eval-data). Downloaded files are
# untrusted data: they are only ever read as data (json / parquet), never executed or imported.
#
#   bash reference/eval/fetch.sh
#
# Needs the `hf` CLI and, for the parquet conversion, a separate tool venv with pyarrow
# ($EVAL_VENV, default sysone-ref/eval-venv, created on first use: python3 -m venv + pip install pyarrow==21.0.0).
# The reference venv is not modified.
set -euo pipefail
REF=${SYSONE_REF:-/home/velvet/Desktop/sysone-ref}
DATA=${EVAL_DATA:-$REF/eval-data}
VENV=${EVAL_VENV:-$REF/eval-venv}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DATA"

hf download --repo-type dataset PolyAI/banking77 data/test-00000-of-00001.parquet README.md \
  --revision 1846984189f2bc0ac51c6fc3a600fa9e1714e0c6 --local-dir "$DATA/banking77" >/dev/null
hf download --repo-type dataset coastalcph/lex_glue ledgar/test-00000-of-00001.parquet README.md \
  --revision c23fdff1a6bf74e0e1a71cb86f1e781d37da888c --local-dir "$DATA/lex_glue" >/dev/null
# THUDM/LongBench-v2 now redirects to zai-org/LongBench-v2 (same repository).
hf download --repo-type dataset zai-org/LongBench-v2 data.json README.md \
  --revision 2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9 --local-dir "$DATA/longbench_v2" >/dev/null

(cd "$DATA" && sha256sum -c --quiet - <<'EOF'
1f40f777c33f0edfa6c530ff84fe2878db6490838c12d22d036f626948905341  banking77/data/test-00000-of-00001.parquet
af9ed0ec09545d39449ffddca3996a1a7774e2bd0a9d6fc09d4e9dd067892a7a  lex_glue/ledgar/test-00000-of-00001.parquet
15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2  longbench_v2/data.json
EOF
)

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q pyarrow==21.0.0
fi
"$VENV/bin/python" -I "$HERE/convert.py" "$DATA"
echo "eval data ready in $DATA"
