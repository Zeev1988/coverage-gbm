"""Metrics. Every number reported is computed ONLY on rows inside the radar's envelope
(see features.inside_envelope): outside it 'not detected' is trivial and would inflate
AUC. The share of rows that were excluded is reported next to each block."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .features import inside_envelope


def basic(y, p):
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = dict(n=int(len(y)), detected_rate=float(y.mean()) if len(y) else None, predicted_rate=float(p.mean()) if len(y) else None)
    if len(y) == 0 or y.min() == y.max():
        return out | dict(auc=None, brier=None, brier_skill=None, logloss=None)
    base = np.full_like(p, y.mean())
    b, b0 = brier_score_loss(y, p), brier_score_loss(y, base)
    return out | dict(auc=float(roc_auc_score(y, p)), brier=float(b), brier_skill=float(1 - b / b0),
                      logloss=float(log_loss(y, p)), logloss_const=float(log_loss(y, base)))


def calibration(y, p, bins=10):
    b = np.clip((np.asarray(p) * bins).astype(int), 0, bins - 1)
    g = pd.DataFrame(dict(y=y, p=p, b=b)).groupby("b").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    return g.reset_index().round(4).to_dict("records")


def by(df, col, bins=None, labels=None, min_n=200):
    key = pd.cut(df[col], bins, labels=labels) if bins is not None else df[col]
    out = {}
    for k, g in df.groupby(key, observed=True):
        if len(g) >= min_n:
            out[str(k)] = basic(g.label.to_numpy(), g.p.to_numpy())
    return out


def hard_report(te, use_los=True):
    """te must have label, p and the envelope columns."""
    m_env = inside_envelope(te, use_los=False)
    m = inside_envelope(te, use_los=use_los)
    g = te[m]
    rep = dict(definition=("inside envelope: range_frac<=1, az_off_frac<=1, 0<=el_frac<=1, radar on"
                           + (", clear line of sight (los_margin_deg>0)" if use_los else "")),
               rows_total=int(len(te)), share_outside_excluded=float(1 - m.mean()),
               share_inside_envelope_without_los_rule=float(m_env.mean()),
               overall=basic(g.label.to_numpy(), g.p.to_numpy()),
               calibration=calibration(g.label.to_numpy(), g.p.to_numpy()))
    ge = te[m_env]
    if ge.label.nunique() == 2:
        rep["baseline_los_rule_auc"] = float(roc_auc_score(ge.label, (ge.los_margin_deg > 0).astype(float)))
        rep["model_auc_same_rows_as_baseline"] = float(roc_auc_score(ge.label, ge.p))
    rep["by_family"] = by(g, "family")
    if "sid" in g:
        rep["by_radar"] = by(g, "sid", min_n=50)
    rep["by_rcs_m2"] = by(g, "log10_rcs", bins=[-9, -1.5, -1, -0.5, 0, 0.5, 1, 9],
                          labels=["<0.03", "0.03-0.1", "0.1-0.3", "0.3-1", "1-3", "3-10", ">10"])
    rep["by_range_frac"] = by(g, "range_frac", bins=[0, .25, .5, .75, .9, 1.0001])
    rep["by_agl_m"] = by(g, "agl_m", bins=[-1e9, 50, 150, 300, 1000, 3000, 1e9],
                         labels=["<50", "50-150", "150-300", "300-1000", "1000-3000", ">3000"])
    if "archetype" in g and g.archetype.nunique() > 1:
        rep["by_archetype"] = by(g, "archetype")
    return rep, m


def pass_level(te, mask):
    """A pass = (day, radar, body). Detected if >= 1 window detected. Prediction 1 - prod(1 - p)
    treats windows as independent, so it OVERESTIMATES (consecutive windows are correlated)."""
    g = te[mask].groupby(["day", "sid", "body"], observed=True)
    P = pd.DataFrame(dict(actual=g.label.max(), n_windows=g.size(),
                          p_indep=g.p.apply(lambda s: 1 - np.prod(1 - s.to_numpy())), p_max=g.p.max()))
    return dict(n_passes=int(len(P)), actual_rate=float(P.actual.mean()), predicted_rate_indep=float(P.p_indep.mean()),
                predicted_rate_max_window=float(P.p_max.mean()),
                auc_indep=basic(P.actual.to_numpy(), P.p_indep.to_numpy())["auc"],
                calibration_indep=calibration(P.actual.to_numpy(), P.p_indep.to_numpy()))
