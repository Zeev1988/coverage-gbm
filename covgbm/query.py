"""Ask the trained model questions - offline, from results/model only (no corpus needed).

  python -m covgbm.query radars
  python -m covgbm.query point   --radar 101 --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1
  python -m covgbm.query pass    --radar 101 --from 31.50,34.80 --to 31.70,35.10 --agl 300 --rcs 0.1 --speed 30
  python -m covgbm.query polygon --radar 101 --polygon "31.5,34.8;31.6,34.8;31.6,34.9;31.5,34.9" --agl 300 --rcs 0.1
  python -m covgbm.query map     --radar 101 --agl 300 --rcs 0.1 --out map.png
  python -m covgbm.query best    --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1        (which radars see this point best)

Common options: --model results/model   --time 14:30 (UTC, default 12:00)   --dow 2 (0=Mon)
  --load quiet|typical|busy (how busy the radar is; default from config)   --msl ALT (instead of --agl)
Coordinates in degrees, heights in metres, RCS in m^2, speed in m/s.

Every answer is a probability PER 30-SECOND WINDOW that the radar produces at least one
plot of the body. For a pass (several windows) the tool prints the expected number of
detected windows and P(at least one window) assuming independent windows, which is an
UPPER estimate: consecutive windows are correlated, so the true pass probability is
between the best single window and that number.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import calendar, geometry, sector_of
from .geo import Frame
from .terrain import Horizon, Terrain


class Model:
    def __init__(self, bundle):
        b = Path(bundle)
        if not (b / "model.txt").exists():
            raise SystemExit(f"no model in {b} - run train first (or pass --model)")
        self.booster = lgb.Booster(model_file=str(b / "model.txt"))
        self.meta = json.loads((b / "meta.json").read_text())
        st = json.loads((b / "radars.json").read_text())
        self.static = st
        self.frame = Frame(st["frame"]["lat0"], st["frame"]["lon0"])
        self.terrain = Terrain.from_h3_parquet(b / "terrain_h3.parquet", self.frame, st["terrain_cell_match_m"])
        hz = np.load(b / "horizons.npz")
        self.radars = {}
        for r in st["radars"]:
            r["horizon"] = Horizon(np.radians(st["horizon_az_step_deg"]), st["horizon_range_step_m"], hz[f"s{r['sid']}"])
            self.radars[int(r["sid"])] = r
        self.ns = st["n_sectors"]
        self.window_s = st["window_s"]

    def radar(self, sid):
        if int(sid) not in self.radars:
            raise SystemExit(f"unknown radar {sid}; known: {sorted(self.radars)}")
        return self.radars[int(sid)]

    def predict(self, sid, lat_deg, lon_deg, alt_agl=None, alt_msl=None, rcs=1.0, speed=30.0, t_ms=None, load="typical"):
        """Vectorised over points. Returns (p, features DataFrame)."""
        R = self.radar(sid)
        lat, lon = np.radians(np.atleast_1d(lat_deg).astype(float)), np.radians(np.atleast_1d(lon_deg).astype(float))
        x, y = self.frame.xy(lat, lon)
        if alt_msl is None:
            alt = self.terrain.at_xy(x, y) + np.broadcast_to(np.asarray(alt_agl, float), x.shape)
        else:
            alt = np.broadcast_to(np.asarray(alt_msl, float), x.shape).astype(float)
        e = R["env"]
        env = dict(max_range_m=e["max_range_m"], az_center=np.nan if e["az_center"] is None else e["az_center"],
                   az_width=np.nan if e["az_width"] is None else e["az_width"],
                   el_max=np.nan if e["el_max"] is None else e["el_max"], active=e["active"])
        f, az = geometry(R, env, R["horizon"], self.terrain, x, y, alt, np.broadcast_to(rcs, x.shape),
                         np.broadcast_to(speed, x.shape))
        t = np.broadcast_to(t_ms if t_ms is not None else 12 * 3600_000, x.shape)
        f = pd.concat([f, calendar(t)], axis=1)
        sec = sector_of(az, self.ns)
        lv = self.meta["load_levels"].get(str(int(sid)), {}).get(load)
        pa, ps = R.get("prevday_rate_all"), R.get("prevday_rate_sec")
        f["prevday_rate_all"] = pa if pa is not None else np.nan
        f["prevday_rate_sector"] = np.array(ps)[sec] if ps is not None else np.nan
        f["hist_rate_all"] = lv if lv is not None else (pa if pa is not None else np.nan)
        # sector load = this load level spread over sectors like yesterday's traffic
        with np.errstate(invalid="ignore", divide="ignore"):
            f["hist_rate_sector"] = (f.prevday_rate_sector * (f.hist_rate_all / pa)) if pa else np.nan
        f["family"] = pd.Categorical([R["family"]] * len(f), categories=self.meta["families"])
        if "sid" in self.meta["features"]:
            f["sid"] = pd.Categorical([int(sid)] * len(f), categories=self.meta["sids"])
        p = self.booster.predict(f[self.meta["features"]])
        return p, f


def parse_ll(s):
    a, b = s.split(",")
    return float(a), float(b)


def t_of(args):
    h, m = (args.time or "12:00").split(":")
    # a Monday + dow days, so day_of_week comes out right
    return int(pd.Timestamp("2024-01-01", tz="UTC").value // 10**6 + int(args.dow) * 86_400_000
               + (int(h) * 60 + int(m)) * 60_000)


def interp_line(a, b, step_m, frame):
    xa, ya = frame.xy(np.radians(a[0]), np.radians(a[1]))
    xb, yb = frame.xy(np.radians(b[0]), np.radians(b[1]))
    L = float(np.hypot(xb - xa, yb - ya))
    n = max(1, int(np.ceil(L / step_m)))
    s = (np.arange(n) + 0.5) / n
    lat, lon = frame.latlon(xa + s * (xb - xa), ya + s * (yb - ya))
    return np.degrees(lat), np.degrees(lon), L


def pass_summary(p):
    p = np.asarray(p)
    return dict(windows=len(p), expected_detected_windows=float(p.sum()), best_window=float(p.max()),
                p_at_least_one_upper=float(1 - np.prod(1 - p)))


def in_poly(x, y, px, py):
    inside = np.zeros(len(x), bool)
    j = len(px) - 1
    for i in range(len(px)):
        c = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        inside ^= c
        j = i
    return inside


def describe(f, p):
    r = f.iloc[0]
    flags = []
    if r.range_frac > 1: flags.append("beyond max range")
    if np.isfinite(r.az_off_frac) and r.az_off_frac > 1: flags.append("outside the radar's sector")
    if np.isfinite(r.el_frac) and (r.el_frac > 1 or r.el_frac < 0): flags.append("outside the elevation band")
    if r.los_margin_deg <= 0: flags.append("hidden behind terrain")
    if r.era_active != 1: flags.append("radar was off on its last day")
    return (f"P(detected in a 30 s window) = {p[0]:.3f}\n"
            f"  range {r.slant_km:.1f} km ({100 * r.range_frac:.0f}% of max), elevation {r.el_deg:.2f} deg, "
            f"terrain clearance {r.los_margin_deg:+.2f} deg, height {r.agl_m:.0f} m AGL / {r.alt_m:.0f} m MSL, "
            f"RCS {10 ** r.log10_rcs:.3g} m^2\n"
            + ("  NOTE: " + "; ".join(flags) + " -> the model was evaluated only inside the envelope; "
               "outside it, low values are expected and were not scored." if flags else "  inside the envelope"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["radars", "point", "pass", "polygon", "map", "best"])
    ap.add_argument("--model", default="results/model")
    ap.add_argument("--radar", type=int)
    ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
    ap.add_argument("--from", dest="frm"); ap.add_argument("--to")
    ap.add_argument("--polygon")
    ap.add_argument("--agl", type=float); ap.add_argument("--msl", type=float)
    ap.add_argument("--rcs", type=float, default=1.0)
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--time"); ap.add_argument("--dow", type=int, default=2)
    ap.add_argument("--load", default="typical", choices=["quiet", "typical", "busy"])
    ap.add_argument("--crossings", type=int, default=300)
    ap.add_argument("--step-km", type=float, default=2.0)
    ap.add_argument("--radius-km", type=float)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    M = Model(a.model)
    if a.agl is None and a.msl is None:
        a.agl = 300.0
    alt = dict(alt_agl=a.agl) if a.msl is None else dict(alt_msl=a.msl)
    t = t_of(a)

    if a.what == "radars":
        print(f"{'radar':>6} {'family':8} {'lat':>8} {'lon':>8} {'range_km':>8} {'sector':>14}  load(plots/min q/t/b)")
        for sid, R in sorted(M.radars.items()):
            e = R["env"]
            sec = "all-round" if e["az_width"] and e["az_width"] >= 2 * np.pi - 1e-6 else (
                f"{np.degrees(e['az_center']):.0f}±{np.degrees(e['az_width']) / 2:.0f}" if e["az_center"] is not None else "unknown")
            lv = M.meta["load_levels"].get(str(sid), {})
            print(f"{sid:>6} {R['family']:8} {np.degrees(R['lat']):8.3f} {np.degrees(R['lon']):8.3f} "
                  f"{e['max_range_m'] / 1000:8.0f} {sec:>14}  {lv.get('quiet', float('nan')):.1f}/"
                  f"{lv.get('typical', float('nan')):.1f}/{lv.get('busy', float('nan')):.1f}"
                  + ("" if e["active"] == 1 else "  (off on last day)"))
        print(f"\nradar state as of {M.static['last_day']}; model trained on days {M.meta['trained_on_days']}")
        return

    if a.what == "best":
        rows = []
        for sid in sorted(M.radars):
            p, f = M.predict(sid, a.lat, a.lon, rcs=a.rcs, speed=a.speed, t_ms=t, load=a.load, **alt)
            rows.append((p[0], sid, f.iloc[0]))
        rows.sort(key=lambda r: -r[0])
        print(f"{'radar':>6} {'P/window':>9} {'range_km':>9} {'clearance_deg':>13}  note")
        for p, sid, r in rows[:15]:
            note = "beyond range" if r.range_frac > 1 else ("outside sector" if np.isfinite(r.az_off_frac) and r.az_off_frac > 1
                                                            else ("terrain" if r.los_margin_deg <= 0 else ""))
            print(f"{sid:>6} {p:9.3f} {r.slant_km:9.1f} {r.los_margin_deg:13.2f}  {note}")
        return

    if a.radar is None:
        raise SystemExit("--radar is required")

    if a.what == "point":
        p, f = M.predict(a.radar, a.lat, a.lon, rcs=a.rcs, speed=a.speed, t_ms=t, load=a.load, **alt)
        print(describe(f, p))

    elif a.what == "pass":
        step = a.speed * M.window_s
        lat, lon, L = interp_line(parse_ll(a.frm), parse_ll(a.to), step, M.frame)
        tt = t + (np.arange(len(lat)) * M.window_s * 1000)
        p, f = M.predict(a.radar, lat, lon, rcs=a.rcs, speed=a.speed, t_ms=tt, load=a.load, **alt)
        s = pass_summary(p)
        print(f"path {L / 1000:.1f} km at {a.speed} m/s = {s['windows']} windows of {M.window_s} s")
        print(f"  expected detected windows {s['expected_detected_windows']:.1f}; best window {s['best_window']:.3f}; "
              f"P(at least one) <= {s['p_at_least_one_upper']:.3f} (independence upper estimate)")
        if a.out:
            pd.DataFrame(dict(lat=lat, lon=lon, p=p, range_km=f.slant_km, clearance_deg=f.los_margin_deg)).to_csv(a.out, index=False)
            print(f"  per-window table -> {a.out}")

    elif a.what == "polygon":
        pts = [parse_ll(s) for s in a.polygon.split(";")]
        px, py = M.frame.xy(np.radians([q[0] for q in pts]), np.radians([q[1] for q in pts]))
        cx, cy, rad = px.mean(), py.mean(), np.hypot(px - px.mean(), py - py.mean()).max()
        rng = np.random.default_rng(0)
        res = []
        step = a.speed * M.window_s
        for _ in range(a.crossings):
            th, off = rng.uniform(0, np.pi), rng.uniform(-rad, rad)
            d = np.array([np.cos(th), np.sin(th)]); n = np.array([-d[1], d[0]])
            s = np.arange(-rad, rad, step)
            x, y = cx + off * n[0] + s * d[0], cy + off * n[1] + s * d[1]
            m = in_poly(x, y, px, py)
            if m.sum() == 0:
                continue
            lat, lon = M.frame.latlon(x[m], y[m])
            tt = t + np.arange(m.sum()) * M.window_s * 1000
            p, _ = M.predict(a.radar, np.degrees(lat), np.degrees(lon), rcs=a.rcs, speed=a.speed, t_ms=tt, load=a.load, **alt)
            res.append(pass_summary(p))
        R = pd.DataFrame(res)
        print(f"{len(R)} random straight crossings of the polygon at {a.speed} m/s:")
        print(f"  windows inside polygon: median {R.windows.median():.0f}")
        print(f"  best single window      : mean {R.best_window.mean():.3f}  (10%: {R.best_window.quantile(.1):.3f}, 90%: {R.best_window.quantile(.9):.3f})")
        print(f"  P(at least one), upper  : mean {R.p_at_least_one_upper.mean():.3f}  (10%: {R.p_at_least_one_upper.quantile(.1):.3f}, 90%: {R.p_at_least_one_upper.quantile(.9):.3f})")
        print("  the true crossing probability lies between the two lines above")

    elif a.what == "map":
        R = M.radar(a.radar)
        rad = (a.radius_km or R["env"]["max_range_m"] / 1000 * 1.1) * 1000
        g = np.arange(-rad, rad + 1, a.step_km * 1000)
        X, Y = np.meshgrid(R["sx"] + g, R["sy"] + g)
        lat, lon = M.frame.latlon(X.ravel(), Y.ravel())
        p, f = M.predict(a.radar, np.degrees(lat), np.degrees(lon), rcs=a.rcs, speed=a.speed, t_ms=t, load=a.load, **alt)
        out = a.out or f"map_radar{a.radar}_agl{a.agl}_rcs{a.rcs}.png"
        save_map(out, g / 1000, p.reshape(X.shape), f.range_frac.to_numpy().reshape(X.shape),
                 f"radar {a.radar} ({R['family']}), body {a.rcs} m^2 at {a.agl if a.msl is None else a.msl} m "
                 f"{'AGL' if a.msl is None else 'MSL'}, {a.time or '12:00'} UTC")
        pd.DataFrame(dict(lat=np.degrees(lat), lon=np.degrees(lon), p=p)).to_csv(Path(out).with_suffix(".csv"), index=False)
        print(f"map -> {out} (+ .csv)")


def save_map(path, g_km, P, rfrac, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=110)
    im = ax.pcolormesh(g_km, g_km, P, vmin=0, vmax=1, cmap="viridis", shading="auto")
    ax.contour(g_km, g_km, rfrac, levels=[1.0], colors="white", linewidths=0.8, linestyles="--")
    ax.plot(0, 0, "r^", ms=8)
    ax.set_aspect("equal"); ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, label="P(detected in a 30 s window)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
