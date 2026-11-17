#!/usr/bin/env bash
# End-to-end check on a tiny synthetic corpus (generated here, offline, by the generator srd:
# repo root, or vendor/srd in a bundle). Runs the REAL-DATA path (fused tracks + witness labels), which
# on synthetic data also reports how often witness labels agree with the oracle truth.
# Takes a few minutes. Everything goes to smoke/ (safe to delete).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONPATH="$PWD:$PWD/vendor:$PWD/../..${PYTHONPATH:+:$PYTHONPATH}"   # srd: vendor/ in a bundle, repo root in the repo
mkdir -p smoke
if [ ! -f smoke/corpus/systems.parquet ]; then
  echo "=== generating a tiny synthetic corpus (6 days x 2 h, 8 radars)"
  "$PY" -m srd.cli generate generator/smoke.yaml smoke/corpus --workers 2
fi
O="--config config.yaml paths.corpus=$PWD/smoke/corpus paths.work=$PWD/smoke/work paths.results=$PWD/smoke/results
   data.workers=2 split.val_start=3 split.test_start=5 evaluate.hidden_radars.k=2 evaluate.spatial.cell_km=20
   model.max_fit_rows=2000000 report.map_radars=2 report.map_step_km=4"
for s in prepare train evaluate report; do
  echo "=== $s"
  "$PY" -m covgbm.$s $O
done
echo "=== query"
"$PY" -m covgbm.query radars --model smoke/results/model
SID=$("$PY" -c "import json;print(json.load(open('smoke/results/model/radars.json'))['radars'][0]['sid'])")
LL=$("$PY" -c "import json,math;r=json.load(open('smoke/results/model/radars.json'))['radars'][0];print(f\"{math.degrees(r['lat'])+0.15:.4f} {math.degrees(r['lon'])+0.1:.4f}\")")
set -- $LL
"$PY" -m covgbm.query point --model smoke/results/model --radar "$SID" --lat "$1" --lon "$2" --agl 500 --rcs 1
"$PY" -m covgbm.query map --model smoke/results/model --radar "$SID" --agl 500 --rcs 1 --out smoke/map.png
echo
echo "SMOKE TEST PASSED - report: smoke/results/report.html"
