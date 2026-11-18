"""Step 1 - build the training table from product tables.

  python -m covgbm.prepare --config config.yaml [key=value ...]

Output: <work>/examples/day=NNN.parquet, one row per (radar, body, 30 s window) for every
body within range_factor x max range of the radar, plus <work>/radar_static/ (positions,
horizons, latest envelope and load) that the query tool needs later.

Bodies and labels: the bodies are the fused system tracks (collectionname TRACK-50__*).
Radar R "saw" body b in window w if R has a plot within witness_gate_m of one of b's track
updates (time difference counted as distance, witness_time_scale_mps). Only bodies some
sensor tracked exist in this table.
Day day_start-1 is read only as warm-up (previous-day rates, envelope estimate).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config as C
from . import io
from .features import calendar, estimate_envelope, geometry, sector_of
from .geo import EARTH_RADIUS_M, Frame, plot_xy
from .terrain import Horizon, Terrain

CTX = {}


# ───────────────────────── setup (once, shared by worker processes) ─────────────────────────
def setup(cfg):
    corpus = Path(cfg.paths.corpus)
    systems = io.read_systems(corpus)
    frame = Frame(systems.lat.mean(), systems.lon.mean())
    terrain = Terrain.from_h3_parquet(corpus / "terrain_h3.parquet", frame, cfg.data.terrain_cell_match_m)
    radars = {}
    t0 = time.time()
    for s in systems.itertuples():
        sx, sy = frame.xy(s.lat, s.lon)
        r_any = s.max_range_m
        radars[s.sid] = dict(sid=s.sid, name=s.name, family=s.family, sx=float(sx), sy=float(sy), alt=float(s.alt),
                             lat=float(s.lat), lon=float(s.lon), max_range_m=float(s.max_range_m),
                             az_width=float(s.az_width), el_max=float(s.el_max), r_any=float(r_any),
                             horizon=Horizon.build(terrain, sx, sy, s.alt, r_any * cfg.data.range_factor,
                                                   cfg.data.horizon_az_step_deg, cfg.data.horizon_range_step_m))
    print(f"{len(radars)} radars, horizons built in {time.time() - t0:.0f}s", flush=True)
    CTX.update(cfg=cfg, corpus=corpus, frame=frame, terrain=terrain, radars=radars, days=io.list_days(corpus))


# ───────────────────────── per-day radar statistics (load + estimated sector) ─────────────────────────
def bad_intervals(date, sid):
    """Time intervals where the radar's data is missing or incomplete (batch absent/incomplete,
    or the radar absent from the batch roster) - inside them no detections can be trusted."""
    d0 = io.day0_ms(date)
    good = [(a, b) for a, b, roster in io.good_spans(CTX["corpus"], date) if roster is None or sid in roster]
    out, cur = [], d0
    for a, b in sorted(good):
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < d0 + io.DAY_MS:
        out.append((cur, d0 + io.DAY_MS))
    return out, sum(b - a for a, b in good) / 60_000


def overlaps(lo, hi, intervals):
    m = np.zeros(len(lo), bool)
    for a, b in intervals:
        m |= (lo < b) & (hi > a)
    return m


def day_stats(date, plots=None):
    cfg = CTX["cfg"]
    if plots is None:
        plots = io.read_plots(CTX["corpus"], date)
    ns = cfg.data.n_sectors
    out = {}
    for sid, R in CTX["radars"].items():
        P = plots[plots.sid == sid]
        bad, _ = bad_intervals(date, sid)
        tm = P.t.to_numpy()
        tm = tm[~overlaps(tm, tm + 1, bad)]
        # minutes in which the radar produced anything = minutes it was operating
        live_min = len(np.unique(tm // 60_000))
        if live_min == 0:
            out[sid] = dict(rate_all=0.0, rate_sec=[0.0] * ns, env=None)
            continue
        sec = sector_of(P.entity_m_azimuth.to_numpy(), ns)
        out[sid] = dict(rate_all=len(P) / live_min,
                        rate_sec=(np.bincount(sec, minlength=ns) / live_min).tolist(),
                        env=estimate_envelope(P.entity_m_azimuth.to_numpy(), R["az_width"]))
    return out


# ───────────────────────── bodies ─────────────────────────
def keep_hash(ids, frac):
    if frac >= 1:
        return np.ones(len(ids), bool)
    return ((np.asarray(ids, np.int64) * 2654435761) % 1_000_003) < frac * 1_000_003


def fused_bodies(date, wins_ok, d0, bin_ms):
    cfg = CTX["cfg"]
    tr = io.read_fused_tracks(CTX["corpus"], date, cfg.data.fused_collection_regex)
    if tr.empty:
        return None, None
    tr["x"], tr["y"] = CTX["frame"].xy(tr.lat.to_numpy(), tr.lon.to_numpy())
    # every update, dead-reckoned along its velocity to +-witness_extrapolate_s, for witness matching
    # (fused updates can be seconds apart; a 250 m/s body moves kilometres between them)
    ex, st = float(cfg.data.witness_extrapolate_s), float(cfg.data.witness_extrapolate_step_s)
    offs = np.arange(-ex, ex + 1e-9, st) if ex > 0 else np.array([0.0])
    upd = pd.DataFrame({"body": np.repeat(tr.body.to_numpy(), len(offs)),
                        "t": (tr.t.to_numpy()[:, None] + offs[None, :] * 1000).ravel(),
                        "x": (tr.x.to_numpy()[:, None] + tr.vx.to_numpy()[:, None] * offs[None, :]).ravel(),
                        "y": (tr.y.to_numpy()[:, None] + tr.vy.to_numpy()[:, None] * offs[None, :]).ravel()})
    tr["win"] = (tr.t - d0) // bin_ms
    tr = tr[tr.win.isin(wins_ok) & keep_hash(tr.body, cfg.data.body_frac)]
    tr["t_mid"] = d0 + tr.win * bin_ms + bin_ms // 2
    tr["dt"] = (tr.t - tr.t_mid).abs()
    rcs = tr.groupby(["body", "win"]).rcs.median().rename("rcs_win")
    B = tr.sort_values("dt").drop_duplicates(["body", "win"]).join(rcs, on=["body", "win"])
    B = pd.DataFrame(dict(body=B.body, win=B.win, t_mid=B.t_mid, x=B.x, y=B.y, alt=B.alt,
                          rcs=B.rcs_win.fillna(B.rcs), speed=np.hypot(B.vx, B.vy)))
    return B.reset_index(drop=True), upd


# ───────────────────────── envelope per row ─────────────────────────
def envelope_for(R, t_mid, prev):
    cfg = CTX["cfg"]
    # sector from the previous day's plots (unknown -> NaN, which never excludes a row)
    st = prev.get(R["sid"]) if prev else None
    est = st["env"] if st else None
    return dict(max_range_m=R["max_range_m"],
                az_center=est[0] if est else np.nan,
                az_width=est[1] if est else (R["az_width"] if np.isfinite(R["az_width"]) else np.nan),
                el_max=R["el_max"], active=0.0 if (st and st["rate_all"] == 0) else 1.0)


# ───────────────────────── one day ─────────────────────────
def process_day(k):
    cfg, corpus = CTX["cfg"], CTX["corpus"]
    days = CTX["days"]
    date = days[k]
    out_file = Path(cfg.paths.work) / "examples" / f"day={k:03d}.parquet"
    if out_file.exists():
        return f"day {k} exists, skipped"
    if k == 0:
        return "day 0 is warm-up only"
    t0 = time.time()
    bin_ms = int(cfg.data.window_s * 1000)
    hist_ms = int(cfg.data.history_min * 60_000)
    ns = cfg.data.n_sectors
    d0 = io.day0_ms(date)
    prev = day_stats(days[k - 1])

    plots = io.read_plots(corpus, date)
    if plots.empty:
        return f"day {k}: no plots"
    spans = io.good_spans(corpus, date)
    wins_ok = sorted({w for a, b, _ in spans for w in range((a - d0) // bin_ms, (b - d0) // bin_ms)})
    B, upd = fused_bodies(date, wins_ok, d0, bin_ms)
    if B is None or B.empty:
        return f"day {k}: no bodies"
    B["agl"] = B.alt - CTX["terrain"].at_xy(B.x.to_numpy(), B.y.to_numpy())
    if cfg.data.min_agl_m is not None:
        B = B[B.agl >= cfg.data.min_agl_m].reset_index(drop=True)
    cal = calendar(B.t_mid.to_numpy())

    # plot positions from the radar's own polar measurement + its reported position
    sys_pos = {sid: (R["sx"], R["sy"]) for sid, R in CTX["radars"].items()}
    lat_s, lon_s = plots.entity_sensorposition_latitude.to_numpy(), plots.entity_sensorposition_longitude.to_numpy()
    sx, sy = CTX["frame"].xy(lat_s, lon_s)
    miss = ~np.isfinite(sx)
    if miss.any():
        sx[miss] = [sys_pos.get(s, (np.nan, np.nan))[0] for s in plots.sid.to_numpy()[miss]]
        sy[miss] = [sys_pos.get(s, (np.nan, np.nan))[1] for s in plots.sid.to_numpy()[miss]]
    plots["px"], plots["py"] = plot_xy(plots.entity_m_range, plots.entity_m_azimuth, plots.entity_m_elevation, sx, sy)
    plots["sector"] = sector_of(plots.entity_m_azimuth.to_numpy(), ns)

    # which body (or bodies) each plot belongs to -> "pairs" (one row per plot x body)
    pcols = ["sid", "t", "sector"]
    # a plot belongs to EVERY fused track within the gate (duplicate tracks of one object are common;
    # crediting only the nearest one would mark the others 'not detected')
    sc = cfg.data.witness_time_scale_mps / 1000.0
    tree = cKDTree(np.c_[upd.x.to_numpy(), upd.y.to_numpy(), upd.t.to_numpy() * sc])
    ok = np.flatnonzero(np.isfinite(plots.px.to_numpy()))
    # gate grows with range (angular errors): max(witness_gate_m, witness_gate_range_frac x range)
    gate = np.maximum(cfg.data.witness_gate_m, cfg.data.witness_gate_range_frac * plots.entity_m_range.to_numpy()[ok])
    kq, ub = int(cfg.data.witness_k), float(np.nanmax(gate)) if len(gate) else 1.0
    Q = np.c_[plots.px.to_numpy()[ok], plots.py.to_numpy()[ok], plots.t.to_numpy()[ok] * sc]
    PI, BO = [], []
    ubody = upd.body.to_numpy()
    for c0 in range(0, len(ok), 500_000):          # chunks keep memory flat on 40M-plot days
        dist, idx = tree.query(Q[c0:c0 + 500_000], k=kq, distance_upper_bound=ub)
        dist, idx = dist.reshape(-1, kq), idx.reshape(-1, kq)
        pi, kj = np.nonzero(np.isfinite(dist) & (dist <= gate[c0:c0 + 500_000, None]))
        PI.append(pi + c0); BO.append(ubody[idx[pi, kj]])
    pi = np.concatenate(PI) if PI else np.zeros(0, np.int64)
    pairs = plots.iloc[ok[pi]][pcols].reset_index(drop=True)
    pairs["own"] = np.concatenate(BO) if BO else np.zeros(0, np.int64)
    pairs = pairs.assign(_p=ok[pi]).drop_duplicates(["_p", "own"]).drop(columns="_p")
    t_lab = pairs.t
    pairs["win_lab"] = (np.asarray(t_lab) - d0) // bin_ms
    pairs = pairs.sort_values("t", kind="stable")
    pairs_by_sid = pairs.groupby("sid").indices
    frames = []
    for sid, R in CTX["radars"].items():
        near = np.hypot(B.x.to_numpy() - R["sx"], B.y.to_numpy() - R["sy"]) <= cfg.data.range_factor * R["r_any"]
        if not near.any():
            continue
        d = B[near].reset_index(drop=True)
        env = envelope_for(R, d.t_mid.to_numpy(), prev)
        f, az = geometry(R, env, R["horizon"], CTX["terrain"], d.x.to_numpy(), d.y.to_numpy(), d.alt.to_numpy(),
                         d.rcs.to_numpy(), d.speed.to_numpy())
        f = pd.concat([f, cal[near].reset_index(drop=True)], axis=1)
        P = plots[plots.sid == sid]
        sec = sector_of(az, ns)

        # label: the radar has >= 1 plot of this body in this window
        j = pairs_by_sid.get(sid)
        own = pairs.iloc[j] if j is not None else pairs.iloc[:0]
        lab_keys = pd.MultiIndex.from_arrays([own.own.to_numpy(), own.win_lab.to_numpy()])
        f["label"] = pd.MultiIndex.from_arrays([d.body.to_numpy(), d.win.to_numpy()]).isin(lab_keys).astype(np.int8)

        # recent load, this body's own plots removed
        bstart = d0 + d.win.to_numpy() * bin_ms
        lo, hi = np.maximum(bstart - hist_ms, d0), bstart
        mins = (hi - lo) / 60_000
        tP, secP = P.t.to_numpy(), P.sector.to_numpy()
        order = np.argsort(tP)
        tP, secP = tP[order], secP[order]
        cnt = lambda ts, a, b: np.searchsorted(ts, b) - np.searchsorted(ts, a)
        n_all = cnt(tP, lo, hi)
        n_sec = np.zeros(len(d), np.int64)
        for s_ in np.unique(sec):
            m = sec == s_
            n_sec[m] = cnt(tP[secP == s_], lo[m], hi[m])
        n_own, n_own_sec = np.zeros(len(d), np.int64), np.zeros(len(d), np.int64)
        own_t = own
        og = own_t.groupby("own").indices
        for body, idx in d.groupby("body").indices.items():
            j = og.get(body)
            if j is None:
                continue
            o = own_t.iloc[j]
            n_own[idx] = cnt(o.t.to_numpy(), lo[idx], hi[idx])
            for s_ in np.unique(sec[idx]):
                jj = idx[sec[idx] == s_]
                n_own_sec[jj] = cnt(o.t.to_numpy()[o.sector.to_numpy() == s_], lo[jj], hi[jj])
        bad, _ = bad_intervals(date, sid)
        hist_ok = (mins >= 1) & ~overlaps(lo, hi, bad)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["hist_rate_all"] = np.where(hist_ok, (n_all - n_own) / mins, np.nan)
            f["hist_rate_sector"] = np.where(hist_ok, (n_sec - n_own_sec) / mins, np.nan)
        st = prev.get(sid)
        f["prevday_rate_all"] = st["rate_all"] if st else np.nan
        f["prevday_rate_sector"] = np.array(st["rate_sec"])[sec] if st else np.nan

        f["sid"], f["family"], f["body"], f["win"], f["day"] = sid, R["family"], d.body.to_numpy(), d.win.to_numpy(), k
        f["t_mid"] = d.t_mid.to_numpy()
        f["tgt_x_km"], f["tgt_y_km"] = (d.x / 1000).astype("float32"), (d.y / 1000).astype("float32")  # analysis only
        f = f[~overlaps(bstart, bstart + bin_ms, bad)]
        frames.append(f)
    if not frames:
        return f"day {k}: no rows"
    out = pd.concat(frames, ignore_index=True)
    for c in out.select_dtypes("float64").columns:
        out[c] = out[c].astype("float32")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.rename(out_file)
    return f"day {k} {date}: {len(out):,} rows, detected {out.label.mean():.3f}, {time.time() - t0:.0f}s"


# ───────────────────────── radar bundle for the query tool ─────────────────────────
def save_static(last_k):
    cfg = CTX["cfg"]
    out = Path(cfg.paths.work) / "radar_static"
    out.mkdir(parents=True, exist_ok=True)
    days = CTX["days"]
    last = day_stats(days[last_k])
    radars = []
    hz = {}
    for sid, R in CTX["radars"].items():
        st = last.get(sid)
        est = (st or {}).get("env")
        env = dict(max_range_m=R["max_range_m"], az_center=est[0] if est else None,
                   az_width=est[1] if est else (R["az_width"] if np.isfinite(R["az_width"]) else None),
                   el_max=R["el_max"] if np.isfinite(R["el_max"]) else None, active=1.0 if st and st["rate_all"] > 0 else 0.0)
        radars.append({k: v for k, v in R.items() if k != "horizon"} | dict(
            env=env, prevday_rate_all=st["rate_all"] if st else None, prevday_rate_sec=st["rate_sec"] if st else None))
        hz[f"s{sid}"] = R["horizon"].min_el
    (out / "radars.json").write_text(json.dumps(dict(
        frame=CTX["frame"].to_dict(), last_day=days[last_k], window_s=cfg.data.window_s, n_sectors=cfg.data.n_sectors,
        horizon_az_step_deg=cfg.data.horizon_az_step_deg, horizon_range_step_m=cfg.data.horizon_range_step_m,
        terrain_cell_match_m=cfg.data.terrain_cell_match_m, radars=radars), indent=1, default=float))
    np.savez_compressed(out / "horizons.npz", **hz)
    src = CTX["corpus"] / "terrain_h3.parquet"
    if src.exists():
        (out / "terrain_h3.parquet").write_bytes(src.read_bytes())
    print(f"radar bundle -> {out}", flush=True)


def main(argv=None):
    cfg, _ = C.from_argv(argv)
    C.save(cfg, cfg.paths.work, "prepare")
    # finished days are skipped on re-runs, so the examples must all come from the same data settings
    ex = Path(cfg.paths.work) / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    h = C.cfg_hash({"data": {k: v for k, v in cfg["data"].items() if k not in ("workers", "day_start", "day_end")},
                    "corpus": str(Path(cfg.paths.corpus).resolve())})
    hf = ex / "_data_settings_hash.txt"
    if hf.exists() and hf.read_text().strip() != h and any(ex.glob("day=*.parquet")):
        raise SystemExit(f"{ex} was built with different data settings or another corpus.\n"
                         f"Delete that folder (or set paths.work to a new folder) and run again.")
    hf.write_text(h)
    setup(cfg)
    n = len(CTX["days"])
    k0 = max(1, int(cfg.data.day_start or 0))
    k1 = min(n, int(cfg.data.day_end) if cfg.data.day_end is not None else n)
    ks = list(range(k0, k1))
    print(f"{n} days in corpus; building days {k0}..{k1 - 1}", flush=True)
    w = int(cfg.data.workers)
    if w > 1:
        with mp.get_context("fork").Pool(w) as pool:
            for msg in pool.imap(process_day, ks):
                print(msg, flush=True)
    else:
        for k in ks:
            print(process_day(k), flush=True)
    save_static(k1 - 1)


if __name__ == "__main__":
    main()
