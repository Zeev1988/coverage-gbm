# coverage-gbm: מודל כיסוי ראדר, מוכן לרשת סגורה

**השאלה:** האם ראדר R יזהה גוף בגודל X (`RCS`) שעובר בנקודה/קו/פוליגון בגובה H?
**התשובה:** הסתברות לכל חלון של 30 שניות שהראדר יפיק לפחות `plot` אחד של הגוף.

כל מה שצריך כדי להכין דאטא, לאמן, להעריך, להפיק דוח ולשאול שאלות נמצא כאן. אין צורך באינטרנט או ב־LLM.
אחרי ההרצה: `results/report.html` (דוח עם הסברים ומילון מונחים) ו־`./ask.sh` (שאלות).

---

## 0. התקנה (פעם אחת)

```bash
./install.sh            # uv sync --frozen  (או pip -r requirements.txt אם אין uv)
./smoke_test.sh         # בדיקה מקצה לקצה על קורפוס סינתטי זעיר, כמה דקות
```
- מירור פנימי: `export UV_INDEX_URL=...` (או `PIP_INDEX_URL`) לפני `./install.sh`.
- אין מירור בכלל: מכינים בחוץ `WHEELS=1 ./make_bundle.sh`, ובפנים `WHEELS=vendor/wheels ./install.sh`.
- Python ≥ 3.13. הגרסאות הנעולות ב־`uv.lock`; `requirements.txt` מיוצא ממנו.

## 1. איפה שמים את הדאטא

ב־`config.yaml`: `paths.corpus` (ברירת מחדל `data/corpus`). המבנה הוא חוזה המוצר (`docs/SYNTH-COLUMNS.md` בריפו):
```
data/corpus/date=YYYY-MM-DD/batch=BB/*.parquet  (+ manifest.json)   plots
data/corpus/tracks/datetime_partition=YYYY-MM-DD/*.parquet          tracks (צריך את המסלולים המאוחדים TRACK-50__*)
data/corpus/systems.parquet                                           ראדרים
data/corpus/terrain_h3.parquet                                        h3_cell -> terrain_ft
```
עמודות שבשימוש:
- `plots`: `entity_m_systemid`, `timestamp`, `entity_m_range/azimuth/elevation`, `entity_sensorposition_latitude/longitude`.
- `tracks`: `collectionname`, `entity_m_header_sendtimemillis`, `entity_statevector_position_*`, `entity_statevector_velocity_vx/vy`, `entity_radar_crosssection` (או `entity_radarcrosssection`), `entity_trackid`.
- `systems`: `system_id`, `position_*`, `max_range_m`, ואופציונלית `family`, `coverage_azimuth`, `coverage_elevation`.

חסר `terrain_h3.parquet`? הכל רץ עם קרקע 0 מ' (אזהרה בלוג), אבל `los_margin_deg` ו־`agl_m` יאבדו משמעות.

## 2. הרצה

```bash
./run.sh                                   # prepare -> train -> evaluate -> report
./run.sh data.body_frac=0.3                # כל הגדרה ב־config.yaml אפשר לדרוס כך
STEPS="train report" ./run.sh              # רק חלק מהשלבים
```
| שלב | מה עושה | פלט |
|---|---|---|
| `prepare` | טבלת אימון: שורה = (ראדר, גוף, חלון 30 ש') לכל גוף בטווח ≤1.3× טווח הראדר | `work/examples/day=NNN.parquet` |
| `train` | חלוקה לפי ימים (אימון / ולידציה / מבחן), LightGBM, מדדים | `results/model/`, `results/metrics/main.json` |
| `evaluate` | ראדרים מוסתרים, אזורים מוסתרים, צפיפות אימון | `results/metrics/*.json` |
| `report` | דוח HTML סטטי עם גרפים, מפות כיסוי, מילון | `results/report.html` |

- **ממשיך מאיפה שנעצר:** ימים שכבר נבנו מדולגים. אם משנים הגדרה תחת `data.*`, ‏`prepare` יסרב לערבב ויבקש למחוק את `work/examples` (או לכוון `paths.work` לתיקייה חדשה).
- **זיכרון:** בערך 10 GB ל־`worker` ביום של כ־40M plots. בהתאם קובעים `data.workers`. אם האימון חונק את הזיכרון: `model.max_fit_rows=5000000`.
- כל פלט נשמר עם ה־config המלא וה־hash שלו (`*.config.json`), כך שאפשר לשחזר כל תוצאה.

## 3. שאלות (offline, רק מתוך `results/model`)

```bash
./ask.sh radars                                                       # ראדרים: סוג, טווח, גזרה, עומס
./ask.sh point   --radar 101 --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1
./ask.sh pass    --radar 101 --from 31.5,34.8 --to 31.7,35.1 --agl 300 --rcs 0.1 --speed 30 --out pass.csv
./ask.sh polygon --radar 101 --polygon "31.5,34.8;31.6,34.8;31.6,34.9;31.5,34.9" --agl 300 --rcs 0.1
./ask.sh map     --radar 101 --agl 300 --rcs 0.1 --out map.png       # מפה + csv
./ask.sh best    --lat 31.5 --lon 34.9 --agl 300 --rcs 0.1           # אילו ראדרים רואים הכי טוב
```
אפשרויות: `--time 14:30` (UTC), `--dow 0..6`, `--load quiet|typical|busy` (עומס הראדר), `--msl 900` (גובה מעל פני הים במקום `--agl`).
מעלות עשרוניות, מטרים, m², m/s.

**איך לקרוא תשובה:**
- `point`: הסתברות לחלון 30 ש'. אם הנקודה מחוץ למעטפת (מעבר לטווח, מחוץ לגזרה, מוסתרת בשטח), יודפס `NOTE`: שם המודל לא נמדד.
- `pass` / `polygon`: מודפסים שני חסמים. "החלון הטוב ביותר" הוא חסם תחתון, ו־`P(at least one)` (חלונות בלתי תלויים) הוא חסם עליון. האמת ביניהם, כי חלונות סמוכים תלויים זה בזה.

## 4. שני מצבי תיוג

| | דאטא אמיתי (ברירת מחדל) | דאטא סינתטי עם אמת |
|---|---|---|
| `data.bodies` | `fused_tracks`: המסלולים המאוחדים | `truth`: המטרות מ־`plan.json` |
| `data.label` | `witness`: ל־R יש plot ליד עדכון של המסלול | `truth`: `src_target_id` |
| `data.envelope` | `estimated`: גזרה מה־plots של אתמול, טווח/גובה מ־systems | `truth`: `envelopes.json` |

`witness` בפירוט: plot של R במרחק ≤ `max(1500 m, 1% מהטווח)` מעדכון של מסלול מאוחד, כשכל עדכון מוארך ±6 ש' לפי המהירות שלו. ה־plot נזקף לכל המסלולים בתוך השער, כי מסלולים כפולים לאותו גוף נפוצים.
על דאטא סינתטי `witness` מחושב לצד האמת, והדוח מראה את ההסכמה. ב־smoke: הסכמה 95%, `witness` מפספס כ־5% מהזיהויים האמיתיים, וכמעט אף פעם לא ממציא זיהוי.

**מגבלה חשובה של `witness`:** בטבלה יש רק גופים שמישהו עקב אחריהם. גופים שאף חיישן לא ראה חסרים, ולכן המודל אופטימי לגופים קשים במיוחד.

## 5. המדדים: רק בתוך המעטפת

מחוץ למעטפת "לא זוהה" טריוויאלי ומנפח את `AUC`, ולכן כל מדד בדוח מחושב רק על שורות עם `range_frac≤1`, `az_off_frac≤1`, `0≤el_frac≤1`, ראדר פעיל, וקו ראייה פתוח (`los_margin_deg>0`). הדוח מציין כמה אחוז הוצאו.
קו הבסיס: הכלל "קו ראייה פתוח ⇒ מזוהה". מילון מלא בתוך `report.html`.

## 5א. קריאת הדוח לפי סדר

1. המספר המרכזי: `AUC` מול `AUC` של כלל ה־LOS, ו־`Brier skill`.
2. כיול: הנקודות על האלכסון?
3. פירוקים: לפי סוג ראדר, `RCS`, טווח, גובה, ראדר.
4. הכללה: `hidden radars` ו־`spatial holdout`. פער קטן = לומד פיזיקה, לא זוכר מקומות.
5. מפות כיסוי.

## 6. קבצים

```
config.yaml          כל ההגדרות (ברירות מחדל מלאות: covgbm/config.py)
covgbm/prepare.py    דאטא -> טבלת אימון (גופים, תוויות, פיצ'רים)
covgbm/features.py   הפיצ'רים, משותפים לאימון ולשאלות; inside_envelope
covgbm/train.py      אימון + מדדים;  evaluate.py  בדיקות הכללה;  metrics.py
covgbm/report.py     דוח HTML;  query.py  כלי השאלות
covgbm/terrain.py    קרקע מ־h3 + אופק לכל ראדר;  geo.py  קואורדינטות (4/3 earth)
generator/smoke.yaml תרחיש זעיר לבדיקה
make_bundle.sh       אריזה להעברה לרשת הסגורה (כולל המחולל srd)
```

## 7. העברה לרשת הסגורה

```bash
./make_bundle.sh          # -> dist/coverage-gbm-<sha>.tar.gz  (+ WHEELS=1 אם אין שם מירור)
```
בצד השני: לחלץ, `./install.sh`, `./smoke_test.sh`, לשים דאטא ב־`data/corpus`, `./run.sh`.
