"""Load the examples table and split it by day."""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .features import FEATURES

KEEP = list(dict.fromkeys(FEATURES + ["sid", "label", "day", "body", "win", "t_mid", "tgt_x_km", "tgt_y_km"]))


def day_files(work):
    files = sorted(glob.glob(str(Path(work) / "examples" / "day=*.parquet")))
    if not files:
        raise SystemExit(f"no examples in {work}/examples - run prepare first")
    return {int(Path(f).stem.split("=")[1]): f for f in files}


def split_days(cfg, days):
    days = sorted(days)
    n = len(days)
    vs = cfg.split.val_start if cfg.split.val_start is not None else days[min(n - 2, int(n * cfg.split.val_frac))]
    ts = cfg.split.test_start if cfg.split.test_start is not None else days[min(n - 1, int(n * cfg.split.test_frac))]
    if not (days[0] < vs < ts <= days[-1]):
        raise SystemExit(f"bad split: days {days[0]}..{days[-1]}, val_start {vs}, test_start {ts} (need >= 3 days)")
    return int(vs), int(ts)


def load(cfg, files, fit_cap=None, test_start=None, test_pass_frac=1.0, families=None, sids=None, seed=0):
    """Read day files; days < test_start are sampled down to fit_cap rows in total;
    test days keep whole passes (radar, body) when test_pass_frac < 1."""
    pre = [d for d in files if test_start is None or d < test_start]
    n_pre = sum(pq.ParquetFile(files[d]).metadata.num_rows for d in pre)
    frac = min(1.0, fit_cap / n_pre) if fit_cap and n_pre else 1.0
    parts = []
    for d, f in sorted(files.items()):
        cols = [c for c in KEEP if c in pq.ParquetFile(f).schema_arrow.names]
        x = pd.read_parquet(f, columns=cols)
        if test_start is not None and d >= test_start:
            if test_pass_frac < 1:
                h = (x.sid.astype(np.int64) * 7919 + x.body.astype(np.int64) * 104729) % 1000
                x = x[h < test_pass_frac * 1000]
        elif frac < 1:
            x = x.sample(frac=frac, random_state=seed + d)
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    df["family"] = pd.Categorical(df.family.astype(str), categories=families or sorted(df.family.astype(str).unique()))
    df["sid"] = pd.Categorical(df.sid.astype(np.int64), categories=sids or sorted(df.sid.astype(np.int64).unique()))
    return df, frac
