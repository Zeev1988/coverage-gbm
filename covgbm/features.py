"""Model features. The SAME functions build the training table (prepare.py) and answer
questions (query.py), so the model never sees features computed two different ways.

A row is (radar, body, 30 s window). Everything here is known at question time:
where the body is, how high, how big, how fast; the radar's installation and current
envelope; terrain; time of day; how busy the radar has been recently.
"""
import numpy as np
import pandas as pd

from .geo import angle_diff, polar

FEATURES = [
    "slant_km", "el_deg", "los_margin_deg", "alt_m", "agl_m",            # geometry + terrain
    "log10_rcs", "speed_mps", "snr_geom_db",                             # body
    "range_frac", "az_off_frac", "el_frac", "env_max_range_km",          # envelope
    "env_az_width_deg", "era_active", "family",
    "hist_rate_all", "hist_rate_sector", "prevday_rate_all", "prevday_rate_sector",  # radar load
    "minute_of_day", "day_of_week",                                      # calendar
]
CATEGORICAL = ["family", "sid"]

DESCRIPTIONS = {
    "slant_km": "straight-line distance radar to body (km)",
    "el_deg": "elevation angle of the body seen from the radar, 4/3-earth (deg)",
    "los_margin_deg": "elevation above the terrain horizon in that direction; < 0 = hidden behind terrain",
    "alt_m": "body altitude above sea level (m)",
    "agl_m": "body height above the terrain under it (m)",
    "log10_rcs": "log10 of body radar cross-section (m^2)",
    "speed_mps": "body ground speed (m/s)",
    "snr_geom_db": "radar-equation term 10log10(RCS) - 40log10(range/1 km)",
    "range_frac": "slant range / the radar's current max range (>1 = beyond range)",
    "az_off_frac": "angle from the sector centre / half sector width (>1 = outside sector)",
    "el_frac": "elevation / max elevation of the beam (>1 above, <0 below)",
    "env_max_range_km": "radar's current max range (km)",
    "env_az_width_deg": "radar's sector width (deg, 360 = all around)",
    "era_active": "1 if the radar is on",
    "family": "radar type",
    "sid": "radar id (main model only, never in the hidden-radar test)",
    "hist_rate_all": "radar's plots per minute in the last 10 min (this body's own plots removed)",
    "hist_rate_sector": "same, only in the 30-deg sector of the body",
    "prevday_rate_all": "radar's plots per minute on the previous day",
    "prevday_rate_sector": "same, in the body's sector",
    "minute_of_day": "UTC minute of day",
    "day_of_week": "0 = Monday",
}


def sector_of(az, n_sectors):
    return (np.mod(az, 2 * np.pi) / (2 * np.pi) * n_sectors).astype(np.int64) % n_sectors


def geometry(radar, env, horizon, terrain, x, y, alt, rcs, speed):
    """radar: dict(sx, sy, alt); env: dict of scalars or arrays (max_range_m, az_center, az_width,
    el_max, active). Returns (DataFrame of features, azimuth array)."""
    dx, dy = np.asarray(x, float) - radar["sx"], np.asarray(y, float) - radar["sy"]
    slant, az, el = polar(dx, dy, alt, radar["alt"])
    ground = np.hypot(dx, dy)
    rcs = np.maximum(np.asarray(rcs, float), 1e-6)
    r_max = np.asarray(env["max_range_m"], float)
    azw = np.asarray(env["az_width"], float)
    full = azw >= 2 * np.pi - 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        az_off = np.where(full, 0.0, angle_diff(az, env["az_center"]) / (azw / 2))
        f = pd.DataFrame({
            "slant_km": slant / 1000.0,
            "el_deg": np.degrees(el),
            "los_margin_deg": np.degrees(el - horizon.threshold(az, ground)),
            "alt_m": np.asarray(alt, float),
            "agl_m": np.asarray(alt, float) - terrain.at_xy(x, y),
            "log10_rcs": np.log10(rcs),
            "speed_mps": np.asarray(speed, float),
            "snr_geom_db": 10 * np.log10(rcs) - 40 * np.log10(np.maximum(slant, 100.0) / 1000.0),
            "range_frac": slant / r_max,
            "az_off_frac": az_off,
            "el_frac": el / np.asarray(env["el_max"], float),
            "env_max_range_km": r_max / 1000.0 + np.zeros(len(slant)),
            "env_az_width_deg": np.degrees(azw) + np.zeros(len(slant)),
            "era_active": np.asarray(env["active"], float) + np.zeros(len(slant)),
        })
    return f, az


def calendar(t_ms):
    ts = pd.to_datetime(np.asarray(t_ms), unit="ms", utc=True)
    return pd.DataFrame({"minute_of_day": (ts.hour * 60 + ts.minute).to_numpy(float),
                         "day_of_week": ts.dayofweek.to_numpy(float)})


def inside_envelope(df, use_los=True):
    """Rows where 'not detected' is NOT trivially explained by the declared envelope:
    within range, inside the sector, inside the elevation band, radar on (and, with
    use_los, a clear line of sight). Unknown envelope parts (NaN) do not exclude a row."""
    m = ((df["range_frac"] <= 1)
         & (df["az_off_frac"].fillna(0) <= 1)
         & (df["el_frac"].fillna(0.5) >= 0) & (df["el_frac"].fillna(0.5) <= 1)
         & (df["era_active"] == 1))
    if use_los:
        m &= df["los_margin_deg"] > 0
    return m.to_numpy()


def estimate_envelope(az, declared_width=np.nan, n_bins=72, min_plots=50):
    """Sector from one day of plot azimuths: the occupied arc is everything except the
    largest run of (nearly) empty azimuth bins. Returns (center_rad, width_rad) or None."""
    az = np.asarray(az, float)
    az = az[np.isfinite(az)]
    if len(az) < min_plots:
        return None
    h, _ = np.histogram(np.mod(az, 2 * np.pi), bins=n_bins, range=(0, 2 * np.pi))
    empty = h < max(1.0, 0.02 * h.max())
    if not empty.any():
        return 0.0, 2 * np.pi
    # longest circular run of empty bins
    e2 = np.concatenate([empty, empty])
    best_len, best_end, run = 0, 0, 0
    for i, v in enumerate(e2):
        run = run + 1 if v else 0
        if run > best_len:
            best_len, best_end = min(run, n_bins), i
    if best_len * 360 / n_bins < 15:      # no real gap -> all-round radar
        return 0.0, 2 * np.pi
    bw = 2 * np.pi / n_bins
    gap_start = (best_end - best_len + 1) * bw
    occ_start, occ_width = gap_start + best_len * bw, (n_bins - best_len) * bw
    center = (occ_start + occ_width / 2) % (2 * np.pi)
    width = declared_width if np.isfinite(declared_width) else occ_width
    return center, width
