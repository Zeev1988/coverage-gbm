#!/usr/bin/env bash
# Full pipeline: prepare -> train -> evaluate -> report.  Resumable: finished days are skipped.
#   ./run.sh                          uses config.yaml
#   ./run.sh --config other.yaml      another config
#   ./run.sh data.body_frac=0.3       any override (see config.yaml)
#   STEPS="train report" ./run.sh     only some steps
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || { echo "no environment: run ./install.sh first"; exit 1; }
export PYTHONPATH="$PWD:$PWD/vendor:$PWD/../..${PYTHONPATH:+:$PYTHONPATH}"   # srd: vendor/ in a bundle, repo root in the repo
ARGS=("$@")
if [[ ! " ${ARGS[*]-} " == *" --config "* ]]; then ARGS=(--config config.yaml "${ARGS[@]+"${ARGS[@]}"}"); fi
STEPS=${STEPS:-"prepare train evaluate report"}
mkdir -p logs
for s in $STEPS; do
  echo "=== $s  $(date '+%F %T')"
  "$PY" -m covgbm.$s "${ARGS[@]}" 2>&1 | tee -a "logs/$s.log"
done
echo "=== done. open results/report.html ; ask questions with ./ask.sh (see README)"
