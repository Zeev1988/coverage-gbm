#!/usr/bin/env bash
# Ask the trained model a question. Examples:  ./ask.sh radars   |   ./ask.sh --help
#   ./ask.sh point --radar 101 --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1
# Uses results/model by default (--model DIR for another one).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m covgbm.query "$@"
