"""Step 3 - generalisation tests (do the numbers hold where the model has not been?).

  python -m covgbm.evaluate --config config.yaml [hidden] [spatial] [density]   (default: all)

hidden  : radars split into k folds (round-robin within each type, every radar hidden once);
          a model WITHOUT the radar id is trained on the other radars, scored on the hidden
          ones on the test days. Compared with the main model on the same rows.
spatial : the area is cut into cell_km x cell_km cells; holdout_frac of the cells are removed
          from training entirely; scored on the test days in those cells.
density : main-model AUC on test rows grouped by how many training rows fell in the same cell.
          A model that memorised places would be clearly better in dense cells.
All AUCs: inside-envelope rows only.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import day_files, load, split_days
from .features import FEATURES, inside_envelope
from .metrics import basic, by
from .train import fit, importance


def cells(df, km):
    return (np.floor(df.tgt_x_km / km).astype(np.int64) * 100_003 + np.floor(df.tgt_y_km / km).astype(np.int64))


def with_main(te, main_pred):
    """attach the main model's prediction p for the same (day, radar, body, window)"""
    te = te.assign(sid=te.sid.astype(np.int64))
    if main_pred is None:
        return te.assign(p=np.nan)
    return te.merge(main_pred[["day", "sid", "body", "win", "p"]], on=["day", "sid", "body", "win"], how="left")


def hidden(cfg, df, vs, ts, main_pred, out):
    k = int(cfg.evaluate.hidden_radars.k)
    fam = df.groupby("sid", observed=True).family.first().astype(str)
    folds = {}
    for fi, (f, sids) in enumerate(fam.groupby(fam)):
        for i, s in enumerate(sorted(sids.index)):
            folds.setdefault((i + fi) % k, []).append(int(s))
    feats = FEATURES
    preds, per_fold = [], []
    for fi, hid in sorted(folds.items()):
        h = df.sid.astype(np.int64).isin(hid)
        m = fit(cfg, df[~h & (df.day < vs)], df[~h & (df.day >= vs) & (df.day < ts)], feats,
                cfg.evaluate.hidden_radars.max_fit_rows, seed=fi)
        te = df[h & (df.day >= ts)].copy()
        te["p_hidden"] = m.predict(te[feats], num_iteration=m.best_iteration)
        preds.append(te)
        per_fold.append(dict(fold=fi, hidden=hid, trees=int(m.best_iteration), top_features=dict(list(importance(m, feats).items())[:8])))
        print(f"hidden fold {fi}: {len(hid)} radars, {m.best_iteration} trees", flush=True)
        del m; gc.collect()
    P = pd.concat(preds, ignore_index=True)
    P = with_main(P, main_pred)
    ins = inside_envelope(P, cfg.evaluate.use_los_in_hard)
    g = P[ins]
    rep = dict(folds=per_fold, n_radars=int(P.sid.nunique()), share_outside_excluded=float(1 - ins.mean()),
               hidden_model=basic(g.label, g.p_hidden),
               main_model_same_rows=basic(g.label[g.p.notna()], g.p[g.p.notna()]) if g.p.notna().any() else None,
               by_family=by(g.assign(p=g.p_hidden), "family"))
    per = []
    for s, x in g.groupby("sid", observed=True):
        if x.label.nunique() == 2 and len(x) >= 50:
            a = basic(x.label, x.p_hidden)["auc"]
            b = basic(x.label[x.p.notna()], x.p[x.p.notna()])["auc"] if x.p.notna().any() else None
            per.append(dict(sid=int(s), family=str(x.family.iloc[0]), n=len(x), auc_hidden=a, auc_main=b))
    rep["by_radar"] = per
    P[["day", "sid", "body", "win", "family", "label", "p_hidden", "p"]].to_parquet(out / "hidden_predictions.parquet", index=False)
    return rep


def spatial(cfg, df, vs, ts, main_pred):
    km, frac = float(cfg.evaluate.spatial.cell_km), float(cfg.evaluate.spatial.holdout_frac)
    c = cells(df, km)
    held = ((c * 2654435761) % 1000) < frac * 1000
    feats = FEATURES + (["sid"] if cfg.model.use_radar_id else [])
    m = fit(cfg, df[~held & (df.day < vs)], df[~held & (df.day >= vs) & (df.day < ts)], feats,
            cfg.evaluate.spatial.max_fit_rows)
    te = df[held & (df.day >= ts)].copy()
    te["p_holdout"] = m.predict(te[feats], num_iteration=m.best_iteration)
    te = with_main(te, main_pred)
    ins = inside_envelope(te, cfg.evaluate.use_los_in_hard)
    g = te[ins]
    return dict(cell_km=km, holdout_frac=frac, n_cells_held=int(np.unique(c[held]).size),
                rows_test=int(len(g)), share_outside_excluded=float(1 - ins.mean()),
                model_never_saw_these_cells=basic(g.label, g.p_holdout),
                main_model_same_rows=basic(g.label[g.p.notna()], g.p[g.p.notna()]))


def density(cfg, df, vs, ts, main_pred):
    km = float(cfg.evaluate.spatial.cell_km) / 2
    tr = df[df.day < vs]
    counts = cells(tr, km).value_counts()
    te = main_pred.copy()
    te["n_train_in_cell"] = cells(te, km).map(counts).fillna(0).to_numpy()
    te = te[inside_envelope(te, cfg.evaluate.use_los_in_hard)]
    te["density_decile"] = pd.qcut(te.n_train_in_cell.rank(method="first"), 10, labels=False)
    out = []
    for d, g in te.groupby("density_decile"):
        out.append(dict(decile=int(d), median_train_rows_in_cell=float(g.n_train_in_cell.median()), **basic(g.label, g.p)))
    return dict(cell_km=km, deciles=out)


def main(argv=None):
    cfg, which = C.from_argv(argv)
    which = set(which) or {"hidden", "spatial", "density"}
    res, work = Path(cfg.paths.results), Path(cfg.paths.work)
    out = res / "metrics"; out.mkdir(parents=True, exist_ok=True)
    C.save(cfg, out, "evaluate")
    files = day_files(work)
    vs, ts = split_days(cfg, list(files))
    df, _ = load(cfg, files, fit_cap=int(cfg.model.max_fit_rows * 1.25), test_start=ts,
                 test_pass_frac=cfg.evaluate.test_pass_frac)
    mp_path = work / "test_predictions.parquet"
    main_pred = pd.read_parquet(mp_path) if mp_path.exists() else None
    if main_pred is not None:
        main_pred["sid"] = main_pred.sid.astype(np.int64)
    if "hidden" in which and cfg.evaluate.hidden_radars.enabled:
        r = hidden(cfg, df, vs, ts, main_pred, work)
        (out / "hidden_radars.json").write_text(json.dumps(r, indent=1, default=str))
    if "spatial" in which and cfg.evaluate.spatial.enabled and main_pred is not None:
        r = spatial(cfg, df, vs, ts, main_pred)
        (out / "spatial_holdout.json").write_text(json.dumps(r, indent=1, default=str))
        print(f"spatial holdout: AUC never-seen cells {r['model_never_saw_these_cells']['auc']}, "
              f"main model same rows {r['main_model_same_rows']['auc']}", flush=True)
    if "density" in which and main_pred is not None:
        r = density(cfg, df, vs, ts, main_pred)
        (out / "density.json").write_text(json.dumps(r, indent=1, default=str))
    print("evaluate done", flush=True)


if __name__ == "__main__":
    main()
