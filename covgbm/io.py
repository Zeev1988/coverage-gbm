"""Readers for the product tables.

  <corpus>/date=YYYY-MM-DD/batch=BB/*.parquet (+ manifest.json)   plots, 30-min batches
  <corpus>/tracks/datetime_partition=YYYY-MM-DD/*.parquet         tracks
  <corpus>/systems.parquet                                          one row per radar
  <corpus>/terrain_h3.parquet                                       h3_cell -> terrain_ft
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BATCH_MS = 30 * 60_000
DAY_MS = 24 * 3600_000


def day0_ms(date):
    return int(pd.Timestamp(str(date), tz="UTC").value // 10**6)


def list_days(corpus):
    return sorted(p.name[5:] for p in Path(corpus).glob("date=*") if p.is_dir())


def batches(corpus, date):
    """[(batch_no, good, roster or None, [files])] for the batches present on disk.
    good = manifest says complete (no manifest -> assumed complete)."""
    out = []
    for b in sorted(Path(corpus, f"date={date}").glob("batch=*")):
        files = sorted(b.glob("*.parquet"))
        good, roster = True, None
        m = b / "manifest.json"
        if m.exists():
            j = json.loads(m.read_text())
            good = bool(j.get("complete", True))
            roster = {int(s) for s in j.get("sensors", [])} or None
        out.append((int(b.name.split("=")[1]), good and bool(files), roster, files))
    return out


def good_spans(corpus, date):
    """Time spans [t0, t1) of complete batches and the per-batch rosters."""
    d0 = day0_ms(date)
    return [(d0 + n * BATCH_MS, d0 + (n + 1) * BATCH_MS, roster) for n, good, roster, _ in batches(corpus, date) if good]


PLOT_COLS = ["entity_m_systemid", "timestamp", "entity_m_range", "entity_m_azimuth", "entity_m_elevation",
             "entity_sensorposition_latitude", "entity_sensorposition_longitude"]


def read_plots(corpus, date, extra=()):
    files = [f for _, _, _, fs in batches(corpus, date) for f in fs]
    if not files:
        return pd.DataFrame(columns=PLOT_COLS + list(extra))
    parts = []
    for f in files:
        names = pq.ParquetFile(f).schema_arrow.names
        parts.append(pq.read_table(f, columns=[c for c in PLOT_COLS + list(extra) if c in names]).to_pandas())
    p = pd.concat(parts, ignore_index=True)
    p = p.rename(columns={"entity_m_systemid": "sid", "timestamp": "t"})
    p["sid"] = p["sid"].astype(np.int64)
    return p


def read_systems(corpus):
    s = pq.read_table(Path(corpus) / "systems.parquet").to_pandas()
    ren = {"entity_systemid": "system_id", "entity_location_latitude": "position_latitude",
           "entity_location_longitude": "position_longitude", "entity_location_altitude": "position_altitude",
           "entity_coverage_maxrange": "max_range_m"}
    s = s.rename(columns={k: v for k, v in ren.items() if k in s.columns})
    out = pd.DataFrame({
        "sid": s["system_id"].astype(str).str.extract(r"(\d+)")[0].astype(np.int64),
        "name": s.get("name", s["system_id"]).astype(str),
        "lat": s["position_latitude"].astype(float), "lon": s["position_longitude"].astype(float),
        "alt": s["position_altitude"].astype(float), "max_range_m": s["max_range_m"].astype(float),
        "family": s["family"].astype(str) if "family" in s else "unknown",
        "az_width": s["coverage_azimuth"].astype(float) if "coverage_azimuth" in s else np.nan,
        "el_max": s["coverage_elevation"].astype(float) if "coverage_elevation" in s else np.nan,
    })
    return out.drop_duplicates("sid").sort_values("sid").reset_index(drop=True)


TRACK_ALIASES = {"entity_radarcrosssection": "entity_radar_crosssection"}


def read_fused_tracks(corpus, date, regex):
    files = sorted(Path(corpus, "tracks").glob(f"datetime_partition={date}/*.parquet"))
    if not files:
        return pd.DataFrame(columns=["body", "t", "lat", "lon", "alt", "rcs", "vx", "vy"])
    t = pd.concat([pq.read_table(f).to_pandas() for f in files], ignore_index=True)
    t = t.rename(columns={k: v for k, v in TRACK_ALIASES.items() if k in t.columns})
    t = t[t["collectionname"].astype(str).str.match(regex)]
    t = t.dropna(subset=["entity_statevector_position_latitude", "entity_statevector_position_longitude",
                         "entity_statevector_position_altitude"])
    return pd.DataFrame({
        "body": t["entity_trackid"].astype(np.int64).to_numpy(),
        "t": t["entity_m_header_sendtimemillis"].astype(np.int64).to_numpy(),
        "lat": t["entity_statevector_position_latitude"].to_numpy(float),
        "lon": t["entity_statevector_position_longitude"].to_numpy(float),
        "alt": t["entity_statevector_position_altitude"].to_numpy(float),
        "rcs": t["entity_radar_crosssection"].to_numpy(float),
        "vx": t["entity_statevector_velocity_vx"].fillna(0).to_numpy(float),
        "vy": t["entity_statevector_velocity_vy"].fillna(0).to_numpy(float),
    })
