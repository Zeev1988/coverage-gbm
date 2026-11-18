"""Step 2 - train the main model (time split) and evaluate it on the later test days.

  python -m covgbm.train --config config.yaml [key=value ...]

Split by day: train < val_start <= early-stopping validation < test_start <= test.
Writes <results>/model/ (everything the query tool needs) and <results>/metrics/main.json.
"""
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .data import day_files, load, split_days
from .features import FEATURES
from .metrics import basic, hard_report, pass_level


def fit(cfg, tr, va, feats, max_rows=None, seed=0):
    if max_rows and len(tr) > max_rows:
        tr = tr.sample(n=max_rows, random_state=seed)
    if max_rows and len(va) > max_rows // 4:
        va = va.sample(n=max_rows // 4, random_state=seed)
    cats = [c for c in ("family", "sid") if c in feats]
    dtr = lgb.Dataset(tr[feats], tr.label, categorical_feature=cats, free_raw_data=True)
    dva = lgb.Dataset(va[feats], va.label, categorical_feature=cats, reference=dtr)
    return lgb.train(dict(cfg.model.params), dtr, cfg.model.num_boost_round, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(cfg.model.early_stopping, verbose=False)])


def importance(m, feats):
    imp = pd.Series(m.feature_importance("gain"), index=feats)
    return (100 * imp / imp.sum()).sort_values(ascending=False).round(2).to_dict()


def load_levels(df):
    """Radar load levels (plots/min, own body removed) per radar over the last 7 days of data:
    the query tool uses them for 'quiet' (25%), 'typical' (median) and 'busy' (75%)."""
    last = df[df.day >= df.day.max() - 6]
    q = last.groupby("sid", observed=True).hist_rate_all.quantile([0.25, 0.5, 0.75]).unstack()
    return {int(s): dict(quiet=float(r[0.25]), typical=float(r[0.5]), busy=float(r[0.75]))
            for s, r in q.iterrows() if np.isfinite(r).all()}


def main(argv=None):
    cfg, _ = C.from_argv(argv)
    res, work = Path(cfg.paths.results), Path(cfg.paths.work)
    (res / "metrics").mkdir(parents=True, exist_ok=True)
    C.save(cfg, res / "metrics", "train")
    files = day_files(work)
    vs, ts = split_days(cfg, list(files))
    t0 = time.time()
    df, frac = load(cfg, files, fit_cap=int(cfg.model.max_fit_rows * 1.25), test_start=ts,
                    test_pass_frac=cfg.evaluate.test_pass_frac)
    print(f"{len(df):,} rows loaded ({time.time() - t0:.0f}s), pre-test sampling {frac:.3f}; "
          f"train days < {vs} <= val < {ts} <= test", flush=True)
    feats = FEATURES + (["sid"] if cfg.model.use_radar_id else [])
    tr, va, te = df[df.day < vs], df[(df.day >= vs) & (df.day < ts)], df[df.day >= ts].copy()
    m = fit(cfg, tr, va, feats, cfg.model.max_fit_rows)
    print(f"trained: {m.best_iteration} trees ({time.time() - t0:.0f}s)", flush=True)
    te["p"] = m.predict(te[feats], num_iteration=m.best_iteration).astype("float32")

    use_los = cfg.evaluate.use_los_in_hard
    hard, mask = hard_report(te, use_los=use_los)
    hard_noslos, _ = hard_report(te, use_los=False)
    rep = dict(split=dict(train_days=f"< {vs}", val_days=f"{vs}..{ts - 1}", test_days=f">= {ts}",
                          rows_train=len(tr), rows_val=len(va), rows_test=len(te), pre_test_sampling=frac,
                          detected_rate_train=float(tr.label.mean())),
               trees=int(m.best_iteration), features=feats, importance_gain_pct=importance(m, feats),
               inside_envelope=hard if use_los else hard_noslos,
               inside_envelope_any_los=hard_noslos,
               pass_level=pass_level(te, mask))
    (res / "metrics" / "main.json").write_text(json.dumps(rep, indent=1, default=str))
    keep = [c for c in ("day", "sid", "body", "win", "t_mid", "family", "label", "p", "range_frac",
                        "az_off_frac", "el_frac", "era_active", "los_margin_deg", "agl_m", "log10_rcs", "slant_km",
                        "tgt_x_km", "tgt_y_km") if c in te]
    te[keep].to_parquet(work / "test_predictions.parquet", index=False)

    # the model bundle: everything the query tool needs, no corpus required
    mb = res / "model"
    mb.mkdir(parents=True, exist_ok=True)
    m.save_model(mb / "model.txt", num_iteration=m.best_iteration)
    meta = dict(features=feats, families=list(df.family.cat.categories), sids=[int(s) for s in df.sid.cat.categories],
                config_hash=C.cfg_hash(cfg), trained_on_days=f"< {vs}", load_levels=load_levels(df))
    (mb / "meta.json").write_text(json.dumps(meta, indent=1))
    for f in (work / "radar_static").glob("*"):
        shutil.copy(f, mb / f.name)
    a = rep["inside_envelope"]["overall"]
    print(f"inside envelope: AUC {a['auc']:.3f}  Brier skill {a['brier_skill']:.3f}  "
          f"LOS-rule AUC {rep['inside_envelope'].get('baseline_los_rule_auc', float('nan')):.3f}  "
          f"({100 * rep['inside_envelope']['share_outside_excluded']:.0f}% of test rows outside, excluded)", flush=True)


if __name__ == "__main__":
    main()
