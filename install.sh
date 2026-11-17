#!/usr/bin/env bash
# Create .venv with the locked package versions.
#   uv (preferred):  uv sync --frozen  (uses uv.lock; set UV_INDEX_URL for an internal mirror)
#   no uv:           pip install -r requirements.txt  (exported from uv.lock; PIP_INDEX_URL for a mirror)
#   no network:      WHEELS=/path/to/wheels ./install.sh
set -euo pipefail
cd "$(dirname "$0")"
if command -v uv >/dev/null && [ -z "${WHEELS:-}" ]; then
  uv sync --frozen
else
  PYTHON=${PYTHON:-python3.13}
  command -v "$PYTHON" >/dev/null || PYTHON=python3
  "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 13), sys.version' || { echo "need Python >= 3.13"; exit 1; }
  "$PYTHON" -m venv .venv
  if [ -n "${WHEELS:-}" ]; then
    .venv/bin/pip install --no-index --find-links "$WHEELS" -r requirements.txt
  else
    .venv/bin/pip install -r requirements.txt
  fi
fi
.venv/bin/python -c "import numpy, pandas, pyarrow, lightgbm, h3, sklearn, scipy, matplotlib, yaml, pydantic, ulid; print('environment ok')"
echo "next: ./smoke_test.sh   (a few minutes, checks everything end to end)"
