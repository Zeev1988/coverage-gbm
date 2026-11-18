# coverage-gbm: radar coverage model, ready for a closed network

**The question:** will radar R detect a body of size X (`RCS`) passing a point / line / polygon at height H?
**The answer:** the probability that, within a 30-second window, the radar produces at least one `plot` of that body.

Everything needed to prepare data, train, evaluate, build a report, ask questions and export insights is here. No internet and no LLM are needed.
After a run: `results/report.html` (report with explanations and a glossary), `./ask.sh` (questions) and `results/insights/` (aggregate bundle that can leave the closed network, see section 8).

---

## 0. Installation (once)

```bash
./install.sh            # uv sync --frozen  (or pip -r requirements.txt if there is no uv)
```
- Internal mirror: `export UV_INDEX_URL=...` (or `PIP_INDEX_URL`) before `./install.sh`.
- No mirror at all: prepare outside with `WHEELS=1 ./make_bundle.sh`, and inside run `WHEELS=vendor/wheels ./install.sh`.
- Python >= 3.13. Versions are pinned in `uv.lock`; `requirements.txt` is exported from it.

## 1. Where the data goes

In `config.yaml`: `paths.corpus` (default `data/corpus`). The layout is the product contract:
```
data/corpus/date=YYYY-MM-DD/batch=BB/*.parquet  (+ manifest.json)   plots
data/corpus/tracks/datetime_partition=YYYY-MM-DD/*.parquet          tracks (needs the fused tracks TRACK-50__*)
data/corpus/systems.parquet                                           radars
data/corpus/terrain_h3.parquet                                        h3_cell -> terrain_ft
```
Columns in use:
- `plots`: `entity_m_systemid`, `timestamp`, `entity_m_range/azimuth/elevation`, `entity_sensorposition_latitude/longitude`.
- `tracks`: `collectionname`, `entity_m_header_sendtimemillis`, `entity_statevector_position_*`, `entity_statevector_velocity_vx/vy`, `entity_radar_crosssection` (or `entity_radarcrosssection`), `entity_trackid`.
- `systems`: `system_id`, `position_*`, `max_range_m`, and optionally `family`, `coverage_azimuth`, `coverage_elevation`.

If `terrain_h3.parquet` is missing everything still runs with terrain at 0 m (a warning is logged), but `los_margin_deg` and `agl_m` lose their meaning.

## 2. Running

```bash
./run.sh                                   # prepare -> train -> evaluate -> report -> insights
./run.sh data.body_frac=0.3                # any setting in config.yaml can be overridden this way
STEPS="train report" ./run.sh              # only some of the steps
```
| Step | What it does | Output |
|---|---|---|
| `prepare` | training table: a row = (radar, body, 30 s window) for every body within <= 1.3x the radar's range | `work/examples/day=NNN.parquet` |
| `train` | split by days (train / validation / test), LightGBM, metrics | `results/model/`, `results/metrics/main.json` |
| `evaluate` | hidden radars, hidden areas, training density | `results/metrics/*.json` |
| `report` | static HTML report with charts, coverage maps, glossary | `results/report.html` |
| `insights` | aggregate bundle for export and later retraining (section 8) | `results/insights/` |

- **Resumes where it stopped:** days that are already built are skipped. If you change a `data.*` setting, `prepare` refuses to mix and asks you to delete `work/examples` (or to point `paths.work` at a new folder).
- **Memory:** roughly 10 GB per worker for a day of about 40M plots. Set `data.workers` accordingly. If training runs out of memory: `model.max_fit_rows=5000000`.
- Every output is saved with the full config and its hash (`*.config.json`), so any result can be reproduced.

## 3. Questions (offline, from `results/model` only)

```bash
./ask.sh radars                                                       # radars: type, range, sector, load
./ask.sh point   --radar 101 --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1
./ask.sh pass    --radar 101 --from 31.5,34.8 --to 31.7,35.1 --agl 300 --rcs 0.1 --speed 30 --out pass.csv
./ask.sh polygon --radar 101 --polygon "31.5,34.8;31.6,34.8;31.6,34.9;31.5,34.9" --agl 300 --rcs 0.1
./ask.sh map     --radar 101 --agl 300 --rcs 0.1 --out map.png       # map + csv
./ask.sh best    --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1           # which radars see it best
```
Options: `--time 14:30` (UTC), `--dow 0..6`, `--load quiet|typical|busy` (radar load), `--msl 900` (height above sea level instead of `--agl`).
Decimal degrees, metres, m², m/s.

**How to read an answer:**
- `point`: probability for a 30 s window. If the point is outside the envelope (beyond range, outside the sector, hidden by terrain), a `NOTE` is printed: the model was not measured there.
- `pass` / `polygon`: two bounds are printed. "Best window" is a lower bound, and `P(at least one)` (windows treated as independent) is an upper bound. The truth lies between them, because neighbouring windows are correlated.

## 4. Bodies and labels

- **Bodies:** the fused system tracks (`collectionname` matching `data.fused_collection_regex`, default `^TRACK-50__`).
- **Label (`witness`):** a plot of R within `max(1500 m, 1% of range)` of an update of a fused track counts as a detection of that body, where every update is extended by +-6 s according to its velocity. The plot is credited to all tracks inside the gate, because duplicate tracks of the same body are common.
- **Envelope:** the sector is estimated from the previous day's plots; range and elevation come from `systems.parquet`.

**Important limitation of `witness`:** the table only contains bodies that somebody tracked. Bodies that no sensor saw are missing, so the model is optimistic for particularly hard bodies.

## 5. The metrics: inside the envelope only

Outside the envelope "not detected" is trivial and inflates `AUC`, so every metric in the report is computed only on rows with `range_frac<=1`, `az_off_frac<=1`, `0<=el_frac<=1`, an active radar and an open line of sight (`los_margin_deg>0`). The report states what percentage was excluded.
The baseline is the rule "open line of sight => detected". A full glossary is inside `report.html`.

### 5a. Reading the report in order

1. The headline number: `AUC` against the `AUC` of the LOS rule, and the `Brier skill`.
2. Calibration: are the points on the diagonal?
3. Breakdowns: by radar type, `RCS`, range, height, radar.
4. Generalisation: `hidden radars` and `spatial holdout`. A small gap = it learns physics and does not memorise places.
5. Coverage maps.

## 6. Files

```
config.yaml          all settings (complete defaults: covgbm/config.py)
covgbm/prepare.py    data -> training table (bodies, labels, features)
covgbm/features.py   the features, shared by training and questions; inside_envelope
covgbm/train.py      training + metrics;  evaluate.py  generalisation tests;  metrics.py
covgbm/report.py     HTML report;  query.py  question tool
covgbm/insights.py   aggregate export bundle + retraining from it
covgbm/terrain.py    terrain from h3 + horizon per radar;  geo.py  coordinates (4/3 earth)
make_bundle.sh       packaging for transfer to the closed network
```

## 7. Transfer to the closed network

```bash
./make_bundle.sh          # -> dist/coverage-gbm-<sha>.tar.gz  (+ WHEELS=1 if there is no mirror there)
```
On the other side: extract, `./install.sh`, put the data in `data/corpus`, `./run.sh`.

## 8. Insights bundle: what can leave the confidential environment

The pipeline runs on confidential data. `insights` turns the run into **aggregates only**, so you can study the data and the model outside and retrain later without the raw data.

```bash
STEPS=insights ./run.sh                 # after prepare + train (also part of the default ./run.sh)
python -m covgbm.insights --config config.yaml insights.min_count=50     # stricter suppression
```

It writes two folders:
- `results/insights/`: **the exportable folder.** No plot, track or body ids, no timestamps below the day.
- `results/insights_private/`: stays inside. The radar-id mapping and a privacy audit (smallest published cell).

What is in `results/insights/`:

| File | What you learn |
|---|---|
| `INSIGHTS.md` | headline numbers + an automatic "what to look at / what to fix" list (day-to-day jumps, drift, weak radars, worst-calibrated slices) |
| `manifest.json` | versions, config hash, model parameters, data fingerprint, sha256 of every file |
| `summary.json` | the same numbers in machine-readable form |
| `tables/per_day.csv` | per day: detection rate, feature means, AUC/Brier (honest only for test days): outages, drift, label problems |
| `tables/per_radar.csv` | per radar: envelope range and sector, terrain blockage, load, detection rate, AUC |
| `tables/slice_*.csv`, `slice2_*.csv` | detection rate **and** model prediction vs every key variable (RCS, range, elevation, height, LOS margin, speed, hour, load, ...), singly and in pairs |
| `tables/test_slice_*.csv`, `worst_slices.csv`, `calibration_test.csv` | on the test days only: where the model is wrong, ranked by statistical significance |
| `tables/feature_importance.csv`, `partial_dependence.csv` | gain, mean abs SHAP, and the learned response curve per feature |
| `tables/feature_profile.csv`, `feature_histograms.csv`, `feature_correlation.csv` | the reference distributions: lets you check later whether new data looks like the old |
| `tables/drift_train_vs_test.csv` | population stability index per feature |
| `tables/corpus_days.csv`, `radars_declared.csv` | data availability from parquet metadata: missing batches, plots and tracks per day, declared radar parameters |
| `cube.parquet` + `cube.meta.json` | **binned sufficient statistics** for retraining (below) |

**Retraining without the raw data.** The cube holds, for every occupied cell of the binned key features (family, RCS, slant range, elevation, height, LOS margin, range fraction, azimuth fraction, hour) and split, the number of rows `n` and detections `k`. This is enough to rebuild a model:
```bash
python -m covgbm.insights refit --cube results/insights/cube.parquet --out refit/
```
It trains a weighted LightGBM on the cell midpoints (train cells), early-stops on the validation cells and reports AUC on the test cells. Cubes from several runs or sites can be concatenated when the bins are the same (they are fixed in `covgbm/insights.py`). The cube model is coarser than the main model (it has no radar id, no load, no per-radar envelope), so use it to tune, compare and sanity-check, not as a drop-in replacement.

**Privacy controls (`insights:` in `config.yaml`):**
- `min_count` (default 20): every table row and cube cell with fewer rows is dropped. The share of rows lost in the cube is reported.
- `anonymize_radars` (default true): radar ids become stable salted pseudonyms (`R526e7e`). Set `insights.salt` to your own secret to make them unguessable. The mapping is only in `insights_private/`.
- `include_geo` (default false): off means no radar altitude and no spatial table. On adds detection rates per cell in a local frame (origin = mean radar position, never absolute coordinates).

Aggregates can still be sensitive (a radar's range and sector, the detection rate in a region). Review `results/insights/` against your release rules before moving it out. Nothing in the bundle is sent anywhere by the code.
