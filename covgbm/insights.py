"""Step 5 - insights bundle: as much aggregate knowledge as possible about the data and the model,
packaged so it can leave a confidential environment and be used later to understand, diagnose
and RETRAIN the model without the raw data.

  python -m covgbm.insights --config config.yaml [key=value ...]
  python -m covgbm.insights refit --cube results/insights/cube.parquet [--out DIR]

Run it after prepare + train. It writes <results>/insights/ (the exportable folder) and
<results>/insights_private/ (stays inside: the radar-id mapping and the privacy audit).

What is in insights/ (nothing row-level: no plot, track or body ids, no timestamps below the day):
  INSIGHTS.md          human summary + automatic "what to look at / what to fix" list
  manifest.json        versions, config hash, data fingerprint, privacy settings, sha256 of every file
  summary.json         headline numbers, split, privacy settings, recommendations
  tables/*.csv         every breakdown (listed in INSIGHTS.md)
  cube.parquet (+meta) BINNED SUFFICIENT STATISTICS: counts n and detections k per cell of
                       (family, rcs, slant, elevation, height, LOS margin, range, azimuth, hour).
                       `refit` retrains a GBM from it with weights, so the model can be rebuilt or
                       tuned outside with no raw data. Cells with fewer than min_count rows are dropped.
Privacy: every table cell and cube cell with fewer than insights.min_count rows is suppressed; radar ids
are replaced by stable salted pseudonyms (insights.anonymize_radars); no coordinates unless
insights.include_geo. Aggregates can still be sensitive: review the folder before moving it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
from . import io
from .data import day_files, load, split_days
from .features import DESCRIPTIONS, inside_envelope
from .metrics import basic

# fixed bin edges: the same in every run, so bundles from different runs/sites can be compared and pooled
BINS = {
    "log10_rcs": [-9, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, 9],
    "slant_km": [0, 5, 10, 20, 40, 60, 80, 120, 160, 250, 1e4],
    "el_deg": [-90, -1, 0, 1, 2, 4, 8, 15, 30, 60, 90],
    "agl_m": [-1e9, 0, 50, 100, 150, 300, 600, 1000, 2000, 4000, 1e9],
    "los_margin_deg": [-90, -2, 0, 0.25, 0.5, 1, 2, 4, 8, 90],
    "range_frac": [0, .1, .25, .4, .55, .7, .8, .9, 1.0001],
    "az_off_frac": [-1e9, -1.0001, -.75, -.5, -.25, 0, .25, .5, .75, 1.0001, 1e9],
    "el_frac": [-1e9, 0, .1, .25, .5, .75, 1.0001, 1e9],
    "speed_mps": [-1, 1, 5, 15, 30, 60, 120, 250, 1e4],
    "snr_geom_db": [-1e9, -60, -40, -30, -20, -10, 0, 10, 20, 1e9],
    "hour": [0, 3, 6, 9, 12, 15, 18, 21, 24],
    "hist_rate_all": [-1, 0, 5, 20, 50, 100, 200, 500, 1e9],
    "env_max_range_km": [0, 10, 20, 40, 80, 120, 200, 1e4],
    "env_az_width_deg": [0, 30, 60, 120, 240, 359, 1e4],
}
SLICES_1D = ["family", "log10_rcs", "slant_km", "el_deg", "agl_m", "los_margin_deg", "range_frac", "az_off_frac",
             "el_frac", "speed_mps", "snr_geom_db", "hour", "day_of_week", "hist_rate_all", "env_max_range_km",
             "env_az_width_deg"]
SLICES_2D = [("family", "log10_rcs"), ("family", "range_frac"), ("family", "agl_m"), ("family", "hour"),
             ("log10_rcs", "range_frac"), ("agl_m", "range_frac"), ("agl_m", "log10_rcs"), ("el_deg", "range_frac"),
             ("los_margin_deg", "range_frac"), ("speed_mps", "log10_rcs"), ("az_off_frac", "range_frac"),
             ("hist_rate_all", "family")]
CUBE_KEYS = ["family", "log10_rcs", "slant_km", "el_deg", "agl_m", "los_margin_deg", "range_frac", "az_off_frac", "hour"]
CATEGORICAL_SLICES = ("family", "day_of_week")


# ───────────────────────────── helpers ─────────────────────────────
def _fmt(v):
    return "-inf" if v <= -1e8 else ("inf" if v >= 1e8 else f"{v:g}")


def _values(df, col):
    return (df["minute_of_day"] / 60.0) if col == "hour" else df[col]


def binned(df, col):
    """String interval labels for a column using the fixed BINS (family / day_of_week stay categorical)."""
    v = _values(df, col)
    if col in CATEGORICAL_SLICES:
        return v.astype(str)
    edges = BINS[col]
    idx = np.clip(np.digitize(v.to_numpy(float), edges[1:-1]), 0, len(edges) - 2)
    labels = np.array([f"[{_fmt(edges[i])},{_fmt(edges[i + 1])})" for i in range(len(edges) - 1)])
    out = pd.Series(labels[idx], index=df.index)
    out[v.isna()] = "nan"
    return out


def bin_index(df, col):
    edges = BINS[col]
    return np.clip(np.digitize(_values(df, col).to_numpy(float), edges[1:-1]), 0, len(edges) - 2).astype(np.int16)


class Anon:
    """Stable pseudonyms for radar ids. The mapping stays in insights_private/."""
    def __init__(self, on, salt):
        self.on, self.salt = on, str(salt)

    def __call__(self, sid):
        if not self.on:
            return str(int(sid))
        return "R" + hashlib.sha256(f"{self.salt}:{int(sid)}".encode()).hexdigest()[:6]


def agg(g, with_p=True):
    y = g.label.to_numpy()
    out = dict(n=len(g), detected=int(y.sum()), rate=float(y.mean()))
    if with_p and "p" in g:
        pm = g.p.to_numpy(float)
        out |= dict(pred_mean=float(pm.mean()), gap=float(y.mean() - pm.mean()), brier=float(np.mean((pm - y) ** 2)))
        se = np.sqrt(max(pm.mean() * (1 - pm.mean()), 1e-9) / len(g))
        out["gap_z"] = float((y.mean() - pm.mean()) / se)
    return out


def table(df, by, min_count, with_p=True):
    """Group-by on binned columns; groups with n < min_count are dropped (suppression)."""
    keys = [binned(df, b) for b in by]
    rows = []
    for k, g in df.groupby(keys, observed=True, sort=True):
        if len(g) < min_count:
            continue
        k = k if isinstance(k, tuple) else (k,)
        rows.append(dict(zip(by, k)) | agg(g, with_p))
    return pd.DataFrame(rows)


def psi(a, b, bins=10):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100:
        return None
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        return 0.0
    e[0], e[-1] = -np.inf, np.inf
    pa = np.histogram(a, e)[0] / len(a) + 1e-6
    pb = np.histogram(b, e)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=5).stdout.strip() or None
    except Exception:
        return None


# ───────────────────────────── sections ─────────────────────────────
def corpus_profile(cfg, anon, include_geo=False):
    """Facts from parquet metadata, manifests and systems/terrain only: no plot is loaded."""
    corpus = Path(cfg.paths.corpus)
    rows = []
    for d in (io.list_days(corpus) if corpus.exists() else []):
        bs = io.batches(corpus, d)
        tr = sorted(Path(corpus, "tracks").glob(f"datetime_partition={d}/*.parquet"))
        rosters = [r for _, _, r, _ in bs if r]
        rows.append(dict(date=d, weekday=pd.Timestamp(d).day_of_week, batches=len(bs),
                         batches_complete=sum(1 for b in bs if b[1]), batches_missing_of_48=48 - len(bs),
                         plots=int(sum(pq.ParquetFile(f).metadata.num_rows for b in bs for f in b[3])),
                         track_rows=int(sum(pq.ParquetFile(f).metadata.num_rows for f in tr)),
                         radars_in_rosters=len(set().union(*rosters)) if rosters else None,
                         min_roster=min((len(r) for r in rosters), default=None)))
    daily = pd.DataFrame(rows)
    systems = terrain = None
    if (corpus / "systems.parquet").exists():
        s = io.read_systems(corpus)
        systems = pd.DataFrame(dict(radar=[anon(x) for x in s.sid], family=s.family, max_range_km=(s.max_range_m / 1000).round(1),
                                    az_width_deg=np.degrees(s.az_width).round(0), el_max_deg=np.degrees(s.el_max).round(1),
                                    **({"alt_m": s.alt.round(0)} if include_geo else {})))
    if (corpus / "terrain_h3.parquet").exists():
        t = pq.read_table(corpus / "terrain_h3.parquet").to_pandas()
        c = [x for x in t.columns if "terrain" in x.lower()]
        if c:
            terrain = dict(cells=int(len(t)), terrain_ft_quantiles={str(q): float(t[c[0]].quantile(q)) for q in (0, .05, .25, .5, .75, .95, 1)})
    return daily, systems, terrain


def feature_profile(df, feats, min_count):
    q = [0.0, .01, .05, .25, .5, .75, .95, .99, 1.0]
    rows, hists = [], []
    for f in feats:
        if f in ("family", "sid"):
            continue
        v = df[f].astype(float)
        qs = v.quantile(q)
        rows.append(dict(feature=f, description=DESCRIPTIONS.get(f, ""), nan_share=float(v.isna().mean()), mean=float(v.mean()),
                         std=float(v.std()), **{f"q{int(x * 100):02d}": float(qs[x]) for x in q}))
        e = np.unique(np.nanquantile(v, np.linspace(0, 1, 21)))
        if len(e) > 2:
            h, e2 = np.histogram(v.dropna(), e)
            hists += [dict(feature=f, lo=float(e2[i]), hi=float(e2[i + 1]), n=int(h[i])) for i in range(len(h)) if h[i] >= min_count]
    return pd.DataFrame(rows), pd.DataFrame(hists)


def _quality(g):
    return {k: v for k, v in basic(g.label, g.p).items() if k in ("auc", "brier", "brier_skill")}


def per_day(df, vs, ts):
    rows = []
    for d, g in df.groupby("day"):
        ins = inside_envelope(g, True)
        gi = g[ins]
        r = dict(day=int(d), split="train" if d < vs else ("val" if d < ts else "test"), rows=len(g), share_inside_envelope=float(ins.mean()),
                 detected_rate_all=float(g.label.mean()), detected_rate_inside=float(gi.label.mean()) if len(gi) else None,
                 mean_slant_km=float(g.slant_km.mean()), mean_log10_rcs=float(g.log10_rcs.mean()), mean_agl_m=float(g.agl_m.mean()),
                 mean_speed_mps=float(g.speed_mps.mean()), mean_hist_rate_all=float(g.hist_rate_all.mean()),
                 bodies=int(g.body.nunique()), radars=int(g.sid.nunique()))
        if len(gi) and gi.label.nunique() == 2:
            r |= _quality(gi) | dict(pred_mean_inside=float(gi.p.mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def per_radar(df, anon, min_count):
    rows = []
    for s, g in df.groupby("sid", observed=True):
        if len(g) < min_count:
            continue
        ins = inside_envelope(g, True)
        gi = g[ins]
        r = dict(radar=anon(s), family=str(g.family.iloc[0]), rows=len(g), days=int(g.day.nunique()), bodies=int(g.body.nunique()),
                 share_inside_envelope=float(ins.mean()), share_los_blocked=float((g.los_margin_deg <= 0).mean()),
                 detected_rate_inside=float(gi.label.mean()) if len(gi) else None,
                 env_max_range_km_median=float(g.env_max_range_km.median()), env_max_range_km_min=float(g.env_max_range_km.min()),
                 env_max_range_km_max=float(g.env_max_range_km.max()), env_az_width_deg_median=float(g.env_az_width_deg.median()),
                 share_active=float((g.era_active == 1).mean()), load_p25=float(g.hist_rate_all.quantile(.25)),
                 load_p50=float(g.hist_rate_all.median()), load_p75=float(g.hist_rate_all.quantile(.75)),
                 mean_agl_m=float(g.agl_m.mean()), mean_log10_rcs=float(g.log10_rcs.mean()))
        if len(gi) >= min_count and gi.label.nunique() == 2:
            r |= _quality(gi) | dict(pred_mean_inside=float(gi.p.mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def calibration_tables(te, min_count):
    g = te[inside_envelope(te, True)]
    rows = []
    for fam in ["__all__"] + sorted(g.family.astype(str).unique()):
        x = g if fam == "__all__" else g[g.family.astype(str) == fam]
        b = np.clip((x.p.to_numpy() * 20).astype(int), 0, 19)
        for k, y in x.groupby(b):
            if len(y) >= min_count:
                rows.append(dict(family=fam, p_bin=f"{k / 20:.2f}-{(k + 1) / 20:.2f}", n=len(y), predicted=float(y.p.mean()), actual=float(y.label.mean())))
    return pd.DataFrame(rows)


def shap_tables(m, X, feats, min_count, n_sample, seed=0):
    """Mean |SHAP| per feature and the mean SHAP contribution (log-odds) per feature bin = the learned response curve."""
    if len(X) > n_sample:
        X = X.sample(n=n_sample, random_state=seed)
    contrib = m.predict(X[feats], num_iteration=m.best_iteration or None, pred_contrib=True)
    Cn = pd.DataFrame(contrib[:, :-1], columns=feats, index=X.index)
    gain = m.feature_importance("gain")
    imp = pd.DataFrame(dict(feature=feats, mean_abs_contrib=Cn.abs().mean().to_numpy(), mean_contrib=Cn.mean().to_numpy(),
                            gain_pct=(100 * gain / max(1.0, gain.sum())).round(2), splits=m.feature_importance("split"))
                       ).sort_values("mean_abs_contrib", ascending=False)
    pdp = []
    for f in feats:
        if f in ("family", "sid"):
            kv = X[f].astype(str)
            for k, ix in kv.groupby(kv).groups.items():
                if len(ix) >= min_count:
                    pdp.append(dict(feature=f, bin=k, n=int(len(ix)), feature_mean=None, mean_contrib=float(Cn.loc[ix, f].mean())))
            continue
        v = X[f].astype(float).to_numpy()
        e = np.unique(np.nanquantile(v, np.linspace(0, 1, 16)))
        if len(e) < 3:
            continue
        idx = np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2)
        cf = Cn[f].to_numpy()
        for i in range(len(e) - 1):
            sel = idx == i
            if sel.sum() >= min_count:
                pdp.append(dict(feature=f, bin=f"[{e[i]:.4g},{e[i + 1]:.4g}]", n=int(sel.sum()), feature_mean=float(np.nanmean(v[sel])),
                                mean_contrib=float(cf[sel].mean())))
    return imp, pd.DataFrame(pdp)


def worst_slices(te, min_count, top=40):
    """Where is the model most wrong? All 1-D and 2-D slices of the inside-envelope test rows, ranked by |z|."""
    g = te[inside_envelope(te, True)]
    out = []
    for c in SLICES_1D:
        t = table(g, [c], min_count)
        if len(t):
            out.append(t.assign(slice=c, bin_b="").rename(columns={c: "bin_a"}))
    for a, b in SLICES_2D:
        t = table(g, [a, b], min_count)
        if len(t):
            out.append(t.assign(slice=f"{a} x {b}").rename(columns={a: "bin_a", b: "bin_b"}))
    if not out:
        return pd.DataFrame()
    W = pd.concat(out, ignore_index=True)
    W["abs_z"] = W.gap_z.abs()
    return W.sort_values("abs_z", ascending=False).head(top)[["slice", "bin_a", "bin_b", "n", "rate", "pred_mean", "gap", "gap_z"]].reset_index(drop=True)


def build_cube(df, min_count):
    """Binned sufficient statistics over inside-envelope rows: one row per occupied (cell, split)."""
    g = df[inside_envelope(df, True)]
    cd = pd.DataFrame({k: (g.family.astype(str).to_numpy() if k == "family" else bin_index(g, k)) for k in CUBE_KEYS})
    cd["split"] = g["split"].to_numpy()
    cd["label"] = g.label.to_numpy()
    c = cd.groupby(CUBE_KEYS + ["split"], observed=True).label.agg(n="size", k="sum").reset_index()
    tot = int(c.n.sum())
    kept = c[c.n >= min_count].reset_index(drop=True)
    meta = dict(keys=CUBE_KEYS, edges={k: BINS[k] for k in CUBE_KEYS if k != "family"}, rows_total=tot, rows_kept=int(kept.n.sum()),
                share_suppressed=float(1 - kept.n.sum() / max(1, tot)), min_count=min_count,
                note="n = rows in the cell, k = detections; a cell value is an index i into edges[key]: [edges[i], edges[i+1])")
    return kept, meta


def refit(cube_path, out_dir=None, params=None, rounds=400):
    """Retrain a GBM from the cube alone (no raw data): each cell becomes two weighted rows (detected / not detected)
    at the cell midpoint. Train on split=train, early-stop on val, report on test."""
    from sklearn.metrics import roc_auc_score
    cube = pd.read_parquet(cube_path)
    meta = json.loads(Path(str(cube_path).replace(".parquet", ".meta.json")).read_text())
    keys, edges = meta["keys"], meta["edges"]

    def mid(k, idx):
        e = np.asarray(edges[k], float)
        lo, hi = e[idx], e[idx + 1]
        step = np.median(np.diff(e[(e > -1e8) & (e < 1e8)])) if ((e > -1e8) & (e < 1e8)).sum() > 1 else 1.0
        lo = np.where(lo < -1e8, hi - step, lo)       # open-ended bins: a finite stand-in one step wide
        hi = np.where(hi > 1e8, lo + step, hi)
        return (lo + hi) / 2

    X = pd.DataFrame({"family": pd.Categorical(cube.family.astype(str))})
    for k in keys:
        if k != "family":
            X[k] = mid(k, cube[k].to_numpy())
    feats = list(X.columns)

    def rows(s):
        c = cube[cube.split == s]
        x = X.loc[c.index]
        r = pd.concat([x.assign(_y=1, _w=c.k.to_numpy(float)), x.assign(_y=0, _w=(c.n - c.k).to_numpy(float))])
        return r[r._w > 0]

    tr, va, te = rows("train"), rows("val"), rows("test")
    if not len(tr) or not len(va):
        raise SystemExit("cube has no train/val cells")
    p = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=1, min_sum_hessian_in_leaf=5.0,
             feature_fraction=0.9, verbose=-1, seed=0) | (params or {})
    mk = lambda r, ref=None: lgb.Dataset(r[feats], r._y, weight=r._w, categorical_feature=["family"], reference=ref)
    dtr = mk(tr)
    m = lgb.train(p, dtr, rounds, valid_sets=[mk(va, dtr)], callbacks=[lgb.early_stopping(50, verbose=False)])
    rep = dict(trees=int(m.best_iteration), cells={s: int((cube.split == s).sum()) for s in ("train", "val", "test")})
    if len(te) and te._y.nunique() == 2:
        pt = m.predict(te[feats], num_iteration=m.best_iteration)
        y, w = te._y.to_numpy(), te._w.to_numpy()
        rep["test"] = dict(rows=float(w.sum()), actual_rate=float(np.average(y, weights=w)), predicted_rate=float(np.average(pt, weights=w)),
                           auc=float(roc_auc_score(y, pt, sample_weight=w)))
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        m.save_model(out / "model_from_cube.txt", num_iteration=m.best_iteration)
        (out / "refit.json").write_text(json.dumps(rep, indent=1))
    return m, rep


# ───────────────────────────── recommendations ─────────────────────────────
def recommendations(S, d, drift, pr, worst, cube_suppressed):
    r = []
    x = d.detected_rate_inside.dropna() if "detected_rate_inside" in d else pd.Series(dtype=float)
    if len(x) and x.max() - x.min() > 0.15:
        r.append(f"Detection rate inside the envelope varies a lot between days ({x.min():.2f}..{x.max():.2f}): check per_day.csv for "
                 "outages, a changed radar configuration or label problems before retraining.")
    t = d[d.split == "test"]
    if "auc" in t and t.auc.notna().any() and t.auc.max() - t.auc.min() > 0.1:
        r.append("Test AUC differs by more than 0.1 between test days: the model is not stable in time; add more recent days to training.")
    if len(drift):
        hi = drift[drift.psi > 0.25].feature.tolist()
        if hi:
            r.append(f"Strong train-vs-test drift (PSI > 0.25) in: {', '.join(hi)}. Retrain on newer data and check whether the data source changed.")
    if "auc" in pr and pr.auc.notna().any():
        lo = pr[pr.auc < pr.auc.median() - 0.1].radar.tolist()
        if lo:
            r.append(f"Radars with clearly lower AUC than the median: {', '.join(lo[:10])}. Look at their envelope (sector width, range) and terrain in "
                     "per_radar.csv; the sector or horizon estimate for them may be wrong.")
    if len(worst):
        w = worst.iloc[0]
        r.append(f"Largest miscalibration: slice '{w['slice']}' bin {w.bin_a} {w.bin_b}: actual {w.rate:.2f} vs predicted {w.pred_mean:.2f} (n={int(w.n)}). "
                 "See worst_slices.csv; more data or a feature for these cases is the first retraining lever.")
    if S["share_outside_envelope"] > 0.6:
        r.append(f"{100 * S['share_outside_envelope']:.0f}% of rows are outside the radar envelope and excluded from every metric: most of the table is trivial negatives. "
                 "Consider lowering data.range_factor to save space and time.")
    if S["share_los_blocked"] > 0.3:
        r.append("Many rows are behind terrain: if terrain_h3.parquet is missing or coarse, los_margin_deg is meaningless; verify the terrain input.")
    if S["detected_rate_inside"] is not None and S["detected_rate_inside"] > 0.97:
        r.append("Almost everything inside the envelope is detected: the labels carry little information; check that the witness gate is not too loose.")
    if cube_suppressed > 0.2:
        r.append(f"{100 * cube_suppressed:.0f}% of rows fall into cube cells below min_count and were dropped; lower insights.min_count (if policy allows) or use fewer bins.")
    return r or ["Nothing alarming found automatically. Read the calibration and per-radar tables before trusting a small difference."]


# ───────────────────────────── main ─────────────────────────────
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "refit":
        import argparse
        ap = argparse.ArgumentParser(prog="covgbm.insights refit")
        ap.add_argument("--cube", required=True)
        ap.add_argument("--out", default=None)
        a = ap.parse_args(argv[1:])
        print(json.dumps(refit(a.cube, a.out)[1], indent=1))
        return
    cfg, _ = C.from_argv(argv)
    ic = cfg.insights
    min_count, include_geo = int(ic["min_count"]), bool(ic["include_geo"])
    res, work = Path(cfg.paths.results), Path(cfg.paths.work)
    out, priv = res / "insights", res / "insights_private"
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)
    for old in tdir.glob("*.csv"):   # a rerun must not leave tables from an older run in the export folder
        old.unlink()
    saved = {}

    def put(name, tab, desc):
        if tab is not None and len(tab):
            tab.to_csv(tdir / f"{name}.csv", index=False)
            saved[name] = desc

    meta = json.loads((res / "model" / "meta.json").read_text())
    salt = ic["salt"] or hashlib.sha256(json.dumps(meta["sids"]).encode()).hexdigest()[:8]
    anon = Anon(ic["anonymize_radars"], salt)

    files = day_files(work)
    vs, ts = split_days(cfg, list(files))
    df, frac = load(cfg, files, fit_cap=int(cfg.model.max_fit_rows * 1.25), test_start=ts,
                    test_pass_frac=cfg.evaluate.test_pass_frac, families=meta["families"], sids=meta["sids"])
    feats = meta["features"]
    m = lgb.Booster(model_file=str(res / "model" / "model.txt"))
    df["p"] = m.predict(df[feats]).astype("float32")
    df["split"] = np.where(df.day < vs, "train", np.where(df.day < ts, "val", "test"))
    ins = inside_envelope(df, True)
    print(f"{len(df):,} rows, {100 * ins.mean():.0f}% inside envelope; train<{vs}<=val<{ts}<=test", flush=True)

    daily, systems, terrain = corpus_profile(cfg, anon, include_geo)
    put("corpus_days", daily, "per corpus day: batches present/complete, plots, track rows, radars in rosters (parquet metadata only)")
    put("radars_declared", systems, "declared radar parameters (pseudonymised id)")

    fp, fh = feature_profile(df, feats, min_count)
    put("feature_profile", fp, "distribution of every feature (quantiles, mean, std, NaN share): the reference for drift checks")
    put("feature_histograms", fh, "20-quantile-bin histograms per feature (bins below min_count dropped)")
    num = [f for f in feats if f not in ("family", "sid")]
    gin = df[ins]
    smp = gin.sample(n=min(200_000, len(gin)), random_state=0) if len(gin) else gin
    if len(smp):
        put("feature_correlation", smp[num].corr(method="spearman").round(3).reset_index(names="feature"),
            "Spearman correlation between features (inside-envelope sample)")
    d = per_day(df, vs, ts)
    put("per_day", d, "per example day: split, detection rate, drift indicators, AUC/Brier (honest only for test days)")
    pr = per_radar(df, anon, min_count)
    put("per_radar", pr, "per radar: envelope, load, terrain blockage, detection rate, AUC (in-sample for train days)")

    te = df[df.split == "test"]
    gte = te[inside_envelope(te, True)]
    put("calibration_test", calibration_tables(te, min_count), "reliability table on test days, overall and per family (20 bins)")
    tr_ = df[df.split == "train"]
    drift = pd.DataFrame([dict(feature=f, psi=psi(tr_[f].to_numpy(float), te[f].to_numpy(float))) for f in num]).dropna() if len(te) else pd.DataFrame(columns=["feature", "psi"])
    put("drift_train_vs_test", drift.sort_values("psi", ascending=False), "population stability index train -> test per feature (>0.1 moderate, >0.25 strong)")
    for c in SLICES_1D:
        put(f"slice_{c}", table(gin, [c], min_count), f"detection rate vs {c}, inside envelope, all days; pred_mean = model on the same rows")
    for a, b in SLICES_2D:
        put(f"slice2_{a}__{b}", table(gin, [a, b], min_count), f"detection rate vs {a} x {b}, inside envelope, all days")
    for c in ("family", "log10_rcs", "range_frac", "agl_m"):
        put(f"test_slice_{c}", table(gte, [c], min_count), f"TEST days only: actual vs predicted by {c}")
    worst = worst_slices(te, min_count)
    put("worst_slices", worst, "test slices where the model is most wrong (ranked by |z| of actual - predicted)")
    if include_geo and "tgt_x_km" in df and len(gin):
        cell = float(cfg.evaluate.spatial.cell_km) / 2
        g2 = gin.assign(cx=np.floor(gin.tgt_x_km / cell).astype(int), cy=np.floor(gin.tgt_y_km / cell).astype(int))
        put("geo_cells", pd.DataFrame([dict(cell_x=int(k[0]), cell_y=int(k[1]), cell_km=cell, **agg(g, True))
                                       for k, g in g2.groupby(["cx", "cy"]) if len(g) >= min_count]),
            "detection rate per local-frame cell (origin = mean radar position; no absolute coordinates)")

    if int(ic["shap_rows"]) > 0 and len(gte):
        imp, pdp = shap_tables(m, gte, feats, min_count, int(ic["shap_rows"]))
        if anon.on and len(pdp):
            pdp.loc[pdp.feature == "sid", "bin"] = pdp.loc[pdp.feature == "sid", "bin"].map(lambda s: anon(int(float(s))))
        put("feature_importance", imp, "gain %, split count, mean |SHAP| (log-odds) per feature, on a test inside-envelope sample")
        put("partial_dependence", pdp, "mean SHAP contribution (log-odds) per feature bin: the learned response curve")

    base = basic(gte.label, gte.p) if len(gte) else {}
    summary = dict(rows=len(df), days=int(df.day.nunique()), radars=int(df.sid.nunique()), families=sorted(df.family.astype(str).unique()),
                   split=dict(val_start=vs, test_start=ts), share_outside_envelope=float(1 - ins.mean()),
                   share_los_blocked=float((df.los_margin_deg <= 0).mean()), detected_rate_inside=float(gin.label.mean()) if len(gin) else None,
                   test_inside_envelope=base, trees=int(m.best_iteration), pre_test_sampling=frac, label_mode=cfg.data.label,
                   bodies=cfg.data.bodies, envelope=cfg.data.envelope, terrain=terrain,
                   privacy=dict(min_count=min_count, anonymize_radars=bool(ic["anonymize_radars"]), include_geo=include_geo))
    if "label_truth" in df and len(gin) and (gin.label_truth >= 0).any():
        summary["witness_vs_truth_agreement"] = float((gin.label == gin.label_truth).mean())

    cube, cmeta = build_cube(df, min_count)
    if len(cube):
        cube.to_parquet(out / "cube.parquet", index=False)
        (out / "cube.meta.json").write_text(json.dumps(cmeta, indent=1))
    summary["cube"] = dict(cells=len(cube), share_suppressed=cmeta["share_suppressed"])

    recs = recommendations(summary, d, drift, pr, worst, cmeta["share_suppressed"])
    summary["recommendations"] = recs
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    if anon.on:
        (priv / "radar_id_map.json").write_text(json.dumps({anon(s): int(s) for s in meta["sids"]}, indent=1))

    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
    md = ["# Insights bundle", "",
          f"Config hash `{C.cfg_hash(cfg)}`; model {summary['trees']} trees; label `{cfg.data.label}`, bodies `{cfg.data.bodies}`.", "",
          f"- {summary['rows']:,} example rows, {summary['days']} days, {summary['radars']} radars; "
          f"{100 * summary['share_outside_envelope']:.0f}% of rows outside the envelope (excluded from every rate and metric).",
          f"- Detection rate inside the envelope: {f3(summary['detected_rate_inside'])}",
          f"- Test days, inside envelope: AUC {f3(base.get('auc'))}, Brier skill {f3(base.get('brier_skill'))}, "
          f"actual rate {f3(base.get('detected_rate'))} vs predicted {f3(base.get('predicted_rate'))}",
          "", "## What to look at / fix", ""] + [f"{i + 1}. {x}" for i, x in enumerate(recs)] + \
         ["", "## Retraining without the raw data", "",
          "`cube.parquet` holds counts (rows, detections) per cell of the binned key features. Rebuild a model from it with",
          "`python -m covgbm.insights refit --cube cube.parquet --out refit/`. Pool cubes from several runs by concatenating them (same bins).", "",
          "## Files", ""] + [f"- `tables/{k}.csv`: {v}" for k, v in sorted(saved.items())] + \
         ["", f"Privacy: cells with fewer than {min_count} rows suppressed; radar ids "
              f"{'pseudonymised' if anon.on else 'REAL'}; coordinates {'included (local frame)' if include_geo else 'omitted'}. Review before export."]
    (out / "INSIGHTS.md").write_text("\n".join(md))

    manifest = dict(created=pd.Timestamp.now("UTC").isoformat(), config_hash=C.cfg_hash(cfg), git=git_sha(), python=platform.python_version(),
                    lightgbm=lgb.__version__, pandas=pd.__version__, numpy=np.__version__, model_params=dict(cfg.model.params),
                    data_fingerprint=dict(example_files=len(files), corpus_days=len(daily),
                                          corpus_plot_rows=int(daily.plots.sum()) if len(daily) else 0),
                    privacy=summary["privacy"], files={})
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            manifest["files"][str(p.relative_to(out))] = dict(sha256=sha256(p), bytes=p.stat().st_size)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    mins = []
    for f in tdir.glob("*.csv"):
        t = pd.read_csv(f)
        if "n" in t and len(t):
            mins.append(int(t["n"].min()))
    (priv / "privacy_audit.json").write_text(json.dumps(dict(min_count=min_count, smallest_published_table_cell=min(mins) if mins else None,
                                                             smallest_published_cube_cell=int(cube.n.min()) if len(cube) else None), indent=1))
    print(f"insights written to {out}: {len(saved)} tables, {len(cube)} cube cells")
    for x in recs:
        print(" -", x)


if __name__ == "__main__":
    main()
