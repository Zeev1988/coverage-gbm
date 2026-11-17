"""Terrain from the product table  h3_cell -> terrain_ft  (resolution 7, ~1.2 km cells).

Lookup = nearest cell centre (KD-tree), so no per-point h3 calls; points farther than
`match_m` from any cell get 0 m (sea level) and are counted in `missing_share`.
Horizon profile per radar: for each azimuth bin and range bin, the lowest elevation
angle a target must have to clear all terrain between the radar and that range.
"""
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import EARTH_RADIUS_M, K_FACTOR

FT_TO_M = 0.3048


class Terrain:
    def __init__(self, frame, lat=None, lon=None, elev_m=None, match_m=2500.0):
        self.frame, self.match_m = frame, match_m
        if lat is None or len(lat) == 0:
            self.tree, self.elev = None, np.zeros(0)
        else:
            x, y = frame.xy(lat, lon)
            self.tree, self.elev = cKDTree(np.c_[x, y]), np.asarray(elev_m, float)

    @classmethod
    def from_h3_parquet(cls, path, frame, match_m=2500.0):
        if path is None or not Path(path).exists():
            print(f"WARNING: no terrain table at {path}: terrain = 0 m everywhere", flush=True)
            return cls(frame, match_m=match_m)
        t = pd.read_parquet(path)
        ll = np.array([h3.cell_to_latlng(c) for c in t["h3_cell"]])  # degrees
        return cls(frame, np.radians(ll[:, 0]), np.radians(ll[:, 1]), t["terrain_ft"].to_numpy() * FT_TO_M, match_m)

    def at_xy(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.tree is None:
            return np.zeros(x.shape)
        d, i = self.tree.query(np.c_[x.ravel(), y.ravel()], distance_upper_bound=self.match_m)
        out = np.zeros(len(i))
        ok = np.isfinite(d)
        out[ok] = self.elev[i[ok]]
        return out.reshape(x.shape)


class Horizon:
    def __init__(self, az_step, r_step, min_el):
        self.az_step, self.r_step, self.min_el = float(az_step), float(r_step), min_el  # (n_az, n_r) rad

    @classmethod
    def build(cls, terrain, sx, sy, salt, max_range_m, az_step_deg=0.5, r_step_m=500.0):
        az = np.radians(np.arange(0, 360, az_step_deg))
        r = np.arange(1, int(np.ceil(max_range_m / r_step_m)) + 1) * r_step_m
        X = sx + r[None, :] * np.sin(az)[:, None]
        Y = sy + r[None, :] * np.cos(az)[:, None]
        h = terrain.at_xy(X, Y)
        el_t = np.arctan2(h - salt - r ** 2 / (2 * K_FACTOR * EARTH_RADIUS_M), r)
        # target at range bin i must clear terrain at bins < i
        cm = np.maximum.accumulate(el_t, axis=1)
        min_el = np.concatenate([np.full((len(az), 1), -np.pi / 2), cm[:, :-1]], axis=1)
        return cls(np.radians(az_step_deg), r_step_m, min_el.astype(np.float32))

    def threshold(self, az, rng):
        ai = np.round(np.mod(az, 2 * np.pi) / self.az_step).astype(np.int64) % self.min_el.shape[0]
        ri = np.clip(np.round(np.asarray(rng) / self.r_step).astype(np.int64) - 1, 0, self.min_el.shape[1] - 1)
        return self.min_el[ai, ri]
