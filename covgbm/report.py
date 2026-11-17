"""Step 4 - static HTML report (no internet needed to open it): results/report.html

  python -m covgbm.report --config config.yaml
"""
import base64
import io as _io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .features import DESCRIPTIONS


def img(fig):
    b = _io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'


def fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def table(rows, cols, heads=None):
    h = "".join(f"<th>{c}</th>" for c in (heads or cols))
    body = "".join("<tr>" + "".join(f"<td>{fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def breakdown(d, title):
    if not d:
        return ""
    rows = [dict(group=k, **v) for k, v in d.items()]
    return (f"<h4>{title}</h4>" + table(rows, ["group", "n", "detected_rate", "predicted_rate", "auc", "brier_skill"],
                                        ["קבוצה", "שורות", "שיעור זיהוי בפועל", "שיעור חזוי", "AUC", "Brier skill"]))


def calib_fig(cal, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.DataFrame(cal)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.scatter(c.predicted, c.actual, s=np.clip(c.n / c.n.max() * 300, 10, 300), alpha=.7)
    ax.plot(c.predicted, c.actual, lw=1)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("actual detected share"); ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return img(fig)


def main(argv=None):
    cfg, _ = C.from_argv(argv)
    res = Path(cfg.paths.results)
    M = res / "metrics"
    load = lambda n: json.loads((M / n).read_text()) if (M / n).exists() else None
    main_ = load("main.json")
    if main_ is None:
        raise SystemExit("no results/metrics/main.json - run train first")
    hid, spa, den = load("hidden_radars.json"), load("spatial_holdout.json"), load("density.json")
    meta = json.loads((res / "model" / "meta.json").read_text())
    radars = json.loads((res / "model" / "radars.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = main_["inside_envelope"]
    o = H["overall"]
    parts = []
    parts.append(f"""
<h1>מודל כיסוי ראדר – דוח תוצאות</h1>
<p class="sub">config hash <code>{meta['config_hash']}</code> · label <code>{meta['label']}</code> ·
bodies <code>{meta['bodies']}</code> · envelope <code>{meta['envelope']}</code> · מצב ראדרים נכון ל־{radars['last_day']}</p>
<div class="box">
<b>השאלה:</b> האם ראדר R יזהה גוף בגודל X (RCS) שעובר בנקודה/פוליגון בגובה H?<br>
<b>התשובה של המודל:</b> הסתברות, לכל חלון של {radars['window_s']} שניות, שהראדר יפיק לפחות <code>plot</code> אחד של הגוף.<br>
<b>שורה בטבלת האימון:</b> (ראדר, גוף, חלון {radars['window_s']} ש') לכל גוף בטווח של עד {cfg.data.range_factor}× הטווח המקסימלי של הראדר.<br>
<b>חשוב:</b> כל המדדים בדוח מחושבים <u>רק על שורות בתוך מעטפת הראדר</u> (בטווח, בגזרה, ברצועת הגובה, ראדר פעיל{', קו ראייה פתוח' if cfg.evaluate.use_los_in_hard else ''}).
מחוץ למעטפת "לא זוהה" הוא טריוויאלי ומנפח AUC, ולכן {100 * H['share_outside_excluded']:.0f}% משורות המבחן הוצאו מכל המדדים.
</div>""")

    parts.append(f"""
<h2>1. התוצאה המרכזית (ימי מבחן, בתוך המעטפת)</h2>
<div class="kpis">
<div><span>{fmt(o['auc'])}</span>AUC של המודל</div>
<div><span>{fmt(H.get('baseline_los_rule_auc'))}</span>AUC של כלל "קו ראייה פתוח ⇒ מזוהה"</div>
<div><span>{fmt(o['brier_skill'])}</span>Brier skill (0 = לא טוב מקבוע)</div>
<div><span>{fmt(o['detected_rate'])}</span>שיעור זיהוי בפועל</div>
<div><span>{fmt(o['predicted_rate'])}</span>שיעור חזוי ממוצע</div>
<div><span>{o['n']:,}</span>שורות נמדדו</div>
</div>
<p>חלוקת ימים: אימון {main_['split']['train_days']}, ולידציה {main_['split']['val_days']}, מבחן {main_['split']['test_days']}.
{main_['trees']} עצים. השוואה הוגנת לכלל קו־הראייה: על אותן שורות (בתוך המעטפת ללא דרישת LOS) ה־AUC של המודל הוא {fmt(H.get('model_auc_same_rows_as_baseline'))}.</p>
<div class="row">{calib_fig(H['calibration'], 'calibration, 30 s windows, inside envelope')}
<div><h4>כיול</h4><p>כל נקודה = קבוצת חלונות עם הסתברות חזויה דומה. על הקו המקווקו = ההסתברות "אומרת אמת".
נקודה מתחת לקו = המודל אופטימי מדי.</p></div></div>""")

    P = main_["pass_level"]
    parts.append(f"""
<h3>ברמת מעבר (pass)</h3>
<p>מעבר = (יום, ראדר, גוף). זוהה אם לפחות חלון אחד זוהה. {P['n_passes']:,} מעברים בתוך המעטפת.
בפועל זוהו {fmt(P['actual_rate'])}. החיזוי <code>1−∏(1−p)</code> נותן {fmt(P['predicted_rate_indep'])}
(<b>מגזים</b>, כי חלונות סמוכים תלויים), והחלון הטוב ביותר נותן {fmt(P['predicted_rate_max_window'])} (חסם תחתון).
האמת נמצאת ביניהם. AUC ברמת מעבר: {fmt(P['auc_indep'])}.</p>""")

    W = main_.get("witness_label_check")
    if W:
        parts.append(f"""<h3>בדיקת תוויות witness מול אמת (רק בדאטא סינתטי)</h3>
<p>אותן שורות, תווית witness (מה שאפשר לחשב בדאטא אמיתי) מול תווית האמת של הסימולטור: הסכמה {fmt(W['agreement'])};
מתוך מה ש־witness קרא "זוהה", {fmt(W['witness_says_detected_but_truth_not'])} לא זוהו באמת;
מתוך הזיהויים האמיתיים, witness פספס {fmt(W['truth_detected_but_witness_missed'])}.
AUC של המודל מול תווית האמת: <b>{fmt(W['model_auc_vs_truth_label'])}</b> (מול witness: {fmt(W['model_auc_vs_witness_label'])}).</p>""")
    parts.append("<h2>2. איפה המודל טוב ואיפה פחות</h2>")
    parts.append(breakdown(H.get("by_family"), "לפי סוג ראדר (family)"))
    parts.append(breakdown(H.get("by_rcs_m2"), "לפי גודל הגוף (RCS, m²)"))
    parts.append(breakdown(H.get("by_range_frac"), "לפי מרחק כחלק מהטווח המקסימלי"))
    parts.append(breakdown(H.get("by_agl_m"), "לפי גובה מעל הקרקע (m)"))
    parts.append(breakdown(H.get("by_archetype"), "לפי סוג גוף (רק בדאטא סינתטי)"))
    if H.get("by_radar"):
        rows = sorted([dict(radar=k, **v) for k, v in H["by_radar"].items()], key=lambda r: (r["auc"] is None, r["auc"] or 0))
        parts.append("<h4>לפי ראדר (מהחלש לחזק)</h4><div class='scroll'>" + table(
            rows, ["radar", "n", "detected_rate", "predicted_rate", "auc", "brier_skill"],
            ["ראדר", "שורות", "זיהוי בפועל", "חזוי", "AUC", "Brier skill"]) + "</div>")

    imp = main_["importance_gain_pct"]
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(imp) + 0.6))
    ks = list(imp)[::-1]
    ax.barh(ks, [imp[k] for k in ks]); ax.set_xlabel("% of total gain")
    parts.append("<h2>3. על מה המודל נשען (feature importance)</h2><div class='row'>" + img(fig) + "<div>"
                 + table([dict(f=k, d=DESCRIPTIONS.get(k, ""), g=imp[k]) for k in imp], ["f", "g", "d"],
                         ["feature", "% gain", "משמעות"]) + "</div></div>")

    parts.append("<h2>4. האם המודל מכליל, או זוכר?</h2>")
    if hid:
        a, b = hid["hidden_model"], hid.get("main_model_same_rows") or {}
        parts.append(f"""<h3>ראדרים שהמודל לא ראה (hidden radars)</h3>
<p>{hid['n_radars']} ראדרים חולקו ל־{len(hid['folds'])} קבוצות; כל קבוצה הוסתרה מהאימון בתורה (מודל בלי מזהה ראדר).
AUC על ראדרים מוסתרים: <b>{fmt(a['auc'])}</b> מול {fmt(b.get('auc'))} של המודל הראשי על אותן שורות.
פער קטן = המודל לומד פיזיקה ותצורה, לא ראדרים ספציפיים. שימוש: ראדר חדש שאין לו היסטוריה.</p>""")
        pr = pd.DataFrame(hid["by_radar"])
        if len(pr) and pr.auc_main.notna().any():
            fig, ax = plt.subplots(figsize=(4.2, 4))
            for fam, g in pr.groupby("family"):
                ax.scatter(g.auc_main, g.auc_hidden, s=18, label=fam)
            lo = float(min(pr.auc_main.min(), pr.auc_hidden.min())) - .01
            ax.plot([lo, 1], [lo, 1], "k:", lw=1); ax.legend(fontsize=8)
            ax.set_xlabel("AUC main model (radar seen)"); ax.set_ylabel("AUC radar hidden")
            parts.append("<div class='row'>" + img(fig) + "<p>כל נקודה = ראדר. מתחת לאלכסון = המודל מרוויח מכך שראה את הראדר הזה באימון.</p></div>")
    if spa:
        parts.append(f"""<h3>אזורים שהמודל לא ראה (spatial holdout)</h3>
<p>השטח חולק לתאים של {spa['cell_km']:.0f}×{spa['cell_km']:.0f} ק"מ; {100 * spa['holdout_frac']:.0f}% מהתאים ({spa['n_cells_held']}) הוצאו מהאימון לגמרי.
AUC בתאים שלא נראו: <b>{fmt(spa['model_never_saw_these_cells']['auc'])}</b> מול {fmt(spa['main_model_same_rows']['auc'])} למודל שאומן גם עליהם.</p>""")
    if den:
        d = pd.DataFrame(den["deciles"])
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(d.decile, d.auc, "o-"); ax.set_xlabel("training density decile of the cell (0 = sparsest)"); ax.set_ylabel("AUC")
        parts.append("<h3>צפיפות אימון</h3><div class='row'>" + img(fig) + f"<p>AUC לפי כמה דוגמאות אימון היו באותו תא ({den['cell_km']:.0f} ק\"מ). "
                     "קו שטוח = אין שינון של מקומות.</p></div>")

    # coverage maps
    from .query import Model, save_map
    try:
        Q = Model(res / "model")
        fams = {}
        for sid, R in sorted(Q.radars.items()):
            fams.setdefault(R["family"], []).append(sid)
        pick = [v[0] for v in fams.values()]
        pick += [s for s in sorted(Q.radars) if s not in pick]
        pick = pick[: int(cfg.report.map_radars)]
        maps = []
        mdir = res / "maps"; mdir.mkdir(exist_ok=True)
        for sid in pick:
            R = Q.radars[sid]
            rad = R["env"]["max_range_m"] * 1.1
            g = np.arange(-rad, rad + 1, cfg.report.map_step_km * 1000)
            X, Y = np.meshgrid(R["sx"] + g, R["sy"] + g)
            lat, lon = Q.frame.latlon(X.ravel(), Y.ravel())
            for h in cfg.report.map_alt_agl_m:
                for rcs in cfg.report.map_rcs_m2:
                    p, f = Q.predict(sid, np.degrees(lat), np.degrees(lon), alt_agl=h, rcs=rcs)
                    fn = mdir / f"radar{sid}_agl{h}_rcs{rcs}.png"
                    save_map(fn, g / 1000, p.reshape(X.shape), f.range_frac.to_numpy().reshape(X.shape),
                             f"radar {sid} ({R['family']}), {rcs} m^2 at {h} m AGL")
                    maps.append(f'<img class="map" src="data:image/png;base64,{base64.b64encode(fn.read_bytes()).decode()}">')
        parts.append("<h2>5. מפות כיסוי לדוגמה</h2><p>הסתברות זיהוי בחלון 30 ש' סביב הראדר (משולש אדום). הקו הלבן המקווקו = הטווח המקסימלי. "
                     "שעה 12:00 UTC, עומס ראדר טיפוסי. מפות לכל ראדר/גובה/גודל: <code>query map</code>.</p><div class='maps'>"
                     + "".join(maps) + "</div>")
    except Exception as e:  # report still useful without maps
        parts.append(f"<p>maps skipped: {e}</p>")

    parts.append(f"""
<h2>6. איך שואלים את המודל שאלות (בלי אינטרנט)</h2>
<pre>./ask.sh radars                                                          # רשימת ראדרים, גזרה, טווח, עומס
./ask.sh point   --radar 101 --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1      # נקודה אחת
./ask.sh pass    --radar 101 --from 31.5,34.8 --to 31.7,35.1 --agl 300 --rcs 0.1 --speed 30
./ask.sh polygon --radar 101 --polygon "31.5,34.8;31.6,34.8;31.6,34.9;31.5,34.9" --agl 300 --rcs 0.1
./ask.sh map     --radar 101 --agl 300 --rcs 0.1 --out map.png
./ask.sh best    --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1                  # מי רואה את הנקודה הכי טוב
אפשרויות: --time 14:30 (UTC)  --dow 0..6  --load quiet|typical|busy  --msl 900 (במקום --agl)</pre>

<h2>7. מילון מונחים</h2>
<dl>
<dt>AUC</dt><dd>הסיכוי שחלון שבו הגוף זוהה יקבל הסתברות גבוהה יותר מחלון שבו לא זוהה. 0.5 = ניחוש, 1 = מושלם. מודד דירוג, לא כיול.</dd>
<dt>Brier skill</dt><dd>1 − Brier(מודל)/Brier(קבוע). כמה טובות ההסתברויות לעומת ניחוש אותו שיעור לכל שורה. 0 = לא מוסיף, 1 = מושלם.</dd>
<dt>כיול (calibration)</dt><dd>האם "70%" אכן קורה ב־70% מהמקרים.</dd>
<dt>בתוך המעטפת (inside envelope)</dt><dd>range_frac≤1, az_off_frac≤1, 0≤el_frac≤1, ראדר פעיל{', los_margin_deg>0' if cfg.evaluate.use_los_in_hard else ''}. שם התשובה לא טריוויאלית.</dd>
<dt>כלל קו ראייה (LOS rule)</dt><dd>קו הבסיס: "אם אין שטח שמסתיר – מזוהה". המודל צריך להיות טוב ממנו משמעותית.</dd>
<dt>witness label</dt><dd>בדאטא אמיתי: ראדר R "ראה" את הגוף אם יש לו plot במרחק עד {cfg.data.witness_gate_m:.0f} מ' מעדכון של המסלול המאוחד (fused track) של הגוף. גופים שאף חיישן לא עקב אחריהם לא קיימים בטבלה.</dd>
<dt>RCS</dt><dd>שטח החתך המכ"מי, m². רחפן קטן ~0.01–0.1, מטוס קל ~1–2.</dd>
</dl>
<h2>8. מגבלות</h2>
<ul>
<li>ההסתברות היא לחלון של {radars['window_s']} ש'. לחיבור למעבר שלם אין נוסחה מדויקת – ה־query מדפיס את שני החסמים.</li>
<li>עם <code>label=witness</code> המודל לומד "האם R ראה גוף שמישהו אחר עקב אחריו"; גופים שאף אחד לא ראה חסרים, ולכן המודל אופטימי עבור גופים קשים במיוחד.</li>
<li>הגזרה (sector) של ראדר מוערכת מה־plots של היום הקודם כשאין מעטפת אמת; ראדר שזז או שינה גזרה היום ייראה נכון רק מחר.</li>
<li>שאלות על ראדר שלא היה בדאטא: המודל הראשי משתמש במזהה ראדר; לראדר חדש השתמשו בתוצאות "hidden radars" כאמדן איכות.</li>
</ul>""")

    css = """body{font-family:Arial,Helvetica,sans-serif;direction:rtl;max-width:1100px;margin:auto;padding:16px;color:#222}
code,pre,img,table{direction:ltr}pre{background:#f4f4f4;padding:10px;text-align:left;overflow-x:auto}
table{border-collapse:collapse;margin:6px 0;font-size:13px}td,th{border:1px solid #ccc;padding:3px 7px;text-align:center}
th{background:#eef}.box{background:#f6f8ff;border:1px solid #cdd;padding:10px;line-height:1.6}.sub{color:#666;font-size:13px}
.kpis{display:flex;flex-wrap:wrap;gap:10px}.kpis div{border:1px solid #ccd;padding:8px 14px;text-align:center;font-size:12px;min-width:120px}
.kpis span{display:block;font-size:24px;font-weight:bold}.row{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap}
.scroll{max-height:380px;overflow-y:auto}.maps img{width:48%}dt{font-weight:bold;margin-top:6px}"""
    html = f"<!doctype html><html lang='he'><head><meta charset='utf-8'><title>coverage model report</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"
    (res / "report.html").write_text(html, encoding="utf-8")
    print(f"report -> {res / 'report.html'}")


if __name__ == "__main__":
    main()
