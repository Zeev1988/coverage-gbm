"""Configuration: config.yaml + defaults + command-line overrides.

Every script takes `--config <file>` and any number of `key.sub=value` overrides,
e.g.  python -m covgbm.prepare --config config.yaml data.body_frac=0.2
The fully resolved configuration is written next to every output together with its
hash, so any result can be traced back to the exact settings that produced it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

DEFAULTS = {
    "paths": {
        "corpus": "data/corpus",        # product-format tables (plots / tracks / systems / terrain)
        "work": "work",                 # examples table, models, predictions
        "results": "results",           # report.html, metrics json, query bundle
    },
    "data": {
        "label": "witness",             # witness (real data) | truth (synthetic, needs src_target_id + plan.json)
        "bodies": "fused_tracks",       # fused_tracks (real data) | truth (synthetic plan.json)
        "fused_collection_regex": r"^TRACK-50__",
        "witness_gate_m": 1500.0,       # plot within this distance of a fused-track update = detection of that body
        "witness_gate_range_frac": 0.01, # ... or this fraction of the plot's range, whichever is larger
        "witness_time_scale_mps": 50.0, # 1 s of time difference counts as this many metres in the match distance
        "witness_k": 32,                # candidate track points checked per plot
        "witness_extrapolate_s": 6.0,   # dead-reckon each fused update +-this many seconds (0 = off)
        "witness_extrapolate_step_s": 2.0,
        "window_s": 30,
        "range_factor": 1.3,            # radar considered for a body if ground distance <= factor x max range
        "history_min": 10,
        "n_sectors": 12,
        "envelope": "estimated",        # estimated (from systems + previous day's plots) | truth (synthetic envelopes.json)
        "min_agl_m": 0.0,               # drop rows with the body below this height above terrain (null = keep all)
        "body_frac": 1.0,               # keep this fraction of bodies (hash of the id), to shrink very large data
        "day_start": 0,                 # day index range to process (day_start-1 is read for warm-up only)
        "day_end": None,
        "workers": 1,                   # parallel day processes
        "horizon_az_step_deg": 0.5,
        "horizon_range_step_m": 500.0,
        "terrain_cell_match_m": 2500.0, # terrain lookup: nearest h3 centre within this distance, else 0 m
    },
    "split": {                          # by day index (examples files day=NNN)
        "val_start": None,              # None -> fractions below
        "test_start": None,
        "val_frac": 0.65,
        "test_frac": 0.8,
    },
    "model": {
        "use_radar_id": True,           # main model may learn per-radar quirks; hidden-radar test never uses it
        "max_fit_rows": 20_000_000,
        "num_boost_round": 3000,
        "early_stopping": 100,
        "params": {
            "objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
            "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0,
        },
    },
    "evaluate": {
        "use_los_in_hard": True,
        "hidden_radars": {"enabled": True, "k": 5, "max_fit_rows": 10_000_000},
        "spatial": {"enabled": True, "cell_km": 40.0, "holdout_frac": 0.2, "max_fit_rows": 10_000_000},
        "test_pass_frac": 1.0,          # fraction of test passes (radar, body, day) kept, whole passes only
    },
    "report": {
        "map_radars": 4,                # coverage maps for this many radars (one per family first)
        "map_alt_agl_m": [100, 1000],
        "map_rcs_m2": [0.1, 1.0],
        "map_step_km": 2.0,
    },
    "query": {
        "history": "typical",           # typical | quiet | busy : which radar-load level to assume
        "default_speed_mps": 30.0,
    },
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Cfg(dict):
    """dict with attribute access: cfg.data.label"""
    def __getattr__(self, k):
        v = self[k]
        return Cfg(v) if isinstance(v, dict) else v


def load(path=None, overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
        base = Path(path).resolve().parent
        for k in ("corpus", "work", "results"):   # relative paths are relative to the config file
            p = Path(cfg["paths"][k])
            cfg["paths"][k] = str(p if p.is_absolute() else base / p)
    for o in overrides:
        key, _, val = o.partition("=")
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return Cfg(cfg)


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def save(cfg, out_dir, name):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.config.json").write_text(json.dumps(dict(cfg, _hash=cfg_hash(cfg)), indent=1, default=str))


def from_argv(argv=None):
    """--config FILE plus key=value overrides; returns (cfg, remaining positional args)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path, ov, rest = None, [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config":
            path = argv[i + 1]; i += 2; continue
        (ov if "=" in a and not a.startswith("-") else rest).append(a); i += 1
    return load(path, ov), rest
