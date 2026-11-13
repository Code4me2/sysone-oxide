#!/usr/bin/env bash
# Make the patched OneJev reference (OneJev@81ce62f + elephant-1's serve.patch) and install it editable into the
# shared reference venv. Idempotent: an existing patched copy is verified against a fresh application of the
# patch instead of being recreated. The unpatched copy is never modified.
#
#   bash reference/setup.sh
#
# Overrides (env): SYSONE_REF (default /home/velvet/Desktop/sysone-ref), ELEPHANT (default
# /home/velvet/Desktop/hf_models/elephant-1).
set -euo pipefail

REF=${SYSONE_REF:-/home/velvet/Desktop/sysone-ref}
ELEPHANT=${ELEPHANT:-/home/velvet/Desktop/hf_models/elephant-1}
SRC=$REF/OneJev
DST=$REF/OneJev-patched
PY=$REF/venv/bin/python
PATCH=$ELEPHANT/serve.patch
ONEJEV=81ce62f1597c91e46767d4d02d6ac2e18534fe94

die() { echo "setup.sh: $*" >&2; exit 1; }
[ -f "$PATCH" ] || die "missing $PATCH"
[ -x "$PY" ] || die "missing venv python $PY"
head=$(cat "$SRC/.git/HEAD")
[ "$head" = "$ONEJEV" ] || die "$SRC is at $head, expected $ONEJEV"

# Apply serve.patch to a scratch copy of the unpatched tree. The copy has no .git, so `git apply` acts as a plain
# patcher there (GIT_CEILING_DIRECTORIES stops it from discovering an enclosing repository).
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
(cd "$SRC" && tar --exclude=.git --exclude=__pycache__ --exclude='*.egg-info' -cf - .) | (cd "$scratch" && tar -xf -)
(cd "$scratch" && GIT_CEILING_DIRECTORIES="$(dirname "$scratch")" git apply "$PATCH")

if [ ! -d "$DST" ]; then
    cp -a "$SRC" "$DST"
    (cd "$DST" && git apply "$PATCH")
    echo "created $DST"
fi
# The patched copy must match the scratch application byte for byte (sources only).
diff -r -q -x .git -x __pycache__ -x '*.egg-info' "$scratch" "$DST" || die "$DST differs from OneJev@$ONEJEV + serve.patch"
echo "verified $DST == OneJev@${ONEJEV:0:7} + serve.patch (sha256 $(sha256sum "$PATCH" | cut -c1-16)...)"

loc=$("$PY" -I -c 'import qev, os; print(os.path.dirname(os.path.dirname(os.path.abspath(qev.__file__))))' 2>/dev/null || true)
if [ "$loc" != "$DST" ]; then
    "$PY" -m pip install -q --no-deps -e "$DST"
fi
"$PY" -I -c 'import qev.readout, qev.engine, qev.server; print("qev importable from", qev.__file__)'
