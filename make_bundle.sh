#!/usr/bin/env bash
# Make a self-contained copy of this project to carry into the closed network:
#   dist/coverage-gbm-<git sha>.tar.gz   = this folder + the generator (srd) from the repo root in vendor/
# Optional: WHEELS=1 also downloads the locked wheels for Linux x86_64 into vendor/wheels
# (only needed if the closed network has no package mirror; then install with WHEELS=vendor/wheels ./install.sh).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
SHA=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)
DIRTY=$(git -C "$ROOT" status --porcelain -- srd offline/coverage-gbm 2>/dev/null | head -1)
[ -n "$DIRTY" ] && SHA="$SHA-dirty"
rm -rf vendor && mkdir -p vendor dist
cp -R "$ROOT/srd" vendor/srd
find vendor -name __pycache__ -prune -exec rm -rf {} +
echo "srd from $ROOT at $SHA" > vendor/SRD_VERSION.txt
if [ "${WHEELS:-}" = 1 ]; then
  uv export --frozen --no-hashes --format requirements-txt > vendor/req.txt
  uv run --no-project --with pip python -m pip download -r vendor/req.txt -d vendor/wheels \
     --only-binary=:all: --python-version 3.13 --implementation cp \
     --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64
fi
NAME=coverage-gbm-$SHA
rm -rf "dist/$NAME" && mkdir -p "dist/$NAME"
tar cf - --exclude=./dist --exclude=./work --exclude=./results --exclude=./smoke --exclude=./data \
    --exclude=./.venv --exclude=./_to_delete --exclude=__pycache__ --exclude=./logs . | (cd "dist/$NAME" && tar xf -)
(cd dist && tar czf "$NAME.tar.gz" "$NAME" && rm -rf "$NAME")
echo "bundle -> dist/$NAME.tar.gz  ($(du -h "dist/$NAME.tar.gz" | cut -f1))"
