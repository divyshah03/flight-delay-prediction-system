# CLAUDE.md — Flight Delay Prediction

## What this project is

Predict whether a US domestic flight will be delayed 15+ minutes, using only
information known a few hours before scheduled departure.

This is a portfolio project. Its job is to prove real ML judgment: careful
point-in-time features, no leakage, a real baseline, honest evaluation, and a
model that's actually served. It is NOT a Kaggle-style "train XGBoost, report
accuracy" project — the whole point is doing the parts most people skip.

Prediction framing: given a flight and a cutoff time of **T-4 hours before
scheduled departure**, output two things:
1. The probability it departs 15+ minutes late (classification)
2. If delayed, the expected number of minutes late (regression)

Together these produce output like "78% chance of delay, expected ~22
minutes late" — this combined framing is the target, not classification
alone. Every feature must only use information available at or before that
cutoff, for both models.

## Scope (strict)

Build exactly these pieces. Do not add anything beyond this list without
asking first — this project has a documented history of scope creep and we
are deliberately keeping it tight.

Status key: `[x]` done & verified · `[~]` in progress / partial · `[ ]` not started.
Updated 2026-09-23 — see Build order below for the step-by-step breakdown.

1. [x] Data ingestion: BTS On-Time Performance + NOAA weather
2. [x] Point-in-time feature pipeline (see Features below — this is the core of
   the project)
3. [x] Baseline models (historical delay rate lookup for classification;
   historical average delay minutes for regression)
4. [x] Two trained classification models (XGBoost/LightGBM and a PyTorch
   neural net) predicting delay probability, PLUS one regression model
   (XGBoost/LightGBM is fine — do not build a second PyTorch model for
   this) predicting expected delay minutes, trained only on flights that
   were actually delayed
5. [x] Evaluation: time-based split, ROC-AUC + PR-AUC + calibration for
   classification, MAE/RMSE for regression, SHAP for both
6. [x] Serving: a FastAPI `/predict` endpoint
7. [x] MLflow experiment tracking
8. [x] Tests: no-leakage checks + API tests
9. [x] CI (GitHub Actions running tests on push)
10. [x] Docker (nice-to-have polish, not core) — built and run end-to-end on
    2026-09-23: `docker build`, then `docker run` serving real traffic;
    `/health` and `/predict` verified to return identical output to the
    non-containerized API (29.3% delay probability, ~68.8 min expected).

**Explicitly out of scope. Do NOT build these unless the user asks:**
- Any frontend (no React, no web UI). A Streamlit demo is optional
  nice-to-have, not required.
- Crew connection risk / tight-turn modeling (crew scheduling data isn't
  realistically available)
- Crosswind component via runway-heading trigonometry
- Weather forecast divergence deltas (rate of change of forecasts)
- Continuous holiday-proximity curves (a simple binary `is_holiday` /
  `days_to_holiday` capped feature is fine if trivial, nothing elaborate)
- Airport capacity tiering / scheduled volume modeling beyond what's a
  natural byproduct of the hub backlog feature
- Multi-airline crew/aircraft rotation network graphs
- Real-time live flight tracking or streaming (this is a batch/historical
  project, not the flight-streaming-platform project)

The rule for any feature idea that comes up mid-build: if it can't be
explained correctly in one sentence without notes, it doesn't belong in a
resume project. Ask before adding anything not on the Features list below.

## Tech stack

- Python 3.11+
- **Polars** for data ingestion and feature engineering (not pandas — the
  tail-number self-joins and multi-year aggregations are the kind of thing
  pandas gets slow and memory-heavy on; Polars handles it with lazy
  evaluation). Pandas is fine for small, final, in-memory frames going into
  sklearn/XGBoost/PyTorch, but the heavy pipeline work is Polars.
- XGBoost or LightGBM for the tree model
- PyTorch for the neural net (embeddings for airport/carrier/route + dense
  features)
- scikit-learn for baseline utilities, metrics, calibration
- SHAP for explainability
- MLflow for experiment tracking
- FastAPI + Uvicorn for serving
- pytest for tests
- Docker (last, optional)

## Data

- **BTS On-Time Performance** (US DOT/BTS): historical US domestic flights —
  scheduled/actual departure & arrival times, delay minutes, tail number,
  carrier, origin/destination.
- **NOAA weather** (hourly METAR/ISD): weather observations per airport per
  hour.
- Scope: top 15 US hub airports, 1 year of data (2024-01 to 2024-12,
  narrowed from the original 2023-2024 range on 2026-09-22 to keep scope
  tight). This is still millions of rows — do not expand beyond this
  without asking.
- Target: `departure_delay >= 15` (binary). Expect roughly an 80/20
  on-time/delayed split — this is real class imbalance, not a bug (see
  Evaluation).

## The weather look-ahead trap (critical — do not get this wrong)

You CANNOT use the actual weather observation at flight time to predict a
flight before it departs — that's the weather that *caused or coincided
with* the delay, and using it is leakage.

Fix: every weather feature must come from data available at or before
**T-4 hours**. Concretely: for a flight scheduled to depart at 4:00 PM, only
use the METAR observation timestamped 12:00 PM or earlier (or a TAF forecast
issued by then, if you choose to incorporate forecasts — using actual T-4h
observations is simpler and sufficient for this project; TAFs are a
stretch goal, not required).

## Features

Point-in-time correctness is the core engineering discipline of this
project. General rule: for any historical/aggregate feature, only include
records where `actual_time < cutoff_time`, where `cutoff_time` =
scheduled departure time minus 4 hours.

Build these:

1. **Calendar/schedule**: scheduled departure time, carrier, route,
   origin/destination airport.
2. **Cyclical time encoding**: transform `hour_of_day` and `day_of_year`
   into sin/cos pairs instead of one-hot encoding. This is a small, cheap,
   correctness-minded detail worth including.
3. **Weather at T-4h**: temperature, wind speed, precipitation, visibility,
   cloud ceiling, at both origin and destination, using the look-ahead rule
   above.
4. **IFR flag**: binary feature — 1 if visibility < 0.5 mi or ceiling <
   200 ft at T-4h, else 0. Cheap, aviation-specific, and meaningful (low
   visibility forces Instrument Flight Rules, which cuts airport arrival
   rate).
5. **Tail-number delay propagation**: was this specific aircraft's most
   recent completed flight (by tail number, before the cutoff) delayed, and
   by how much? This is the single most important feature in the project —
   implement it carefully and test it directly.
6. **Hub network backlog index**: percentage of flights that departed the
   same origin airport delayed, over the 3 hours before cutoff. This is the
   "delays cascade through the airport system" signal and is worth building
   even though it's more work than #5 — it's the best complexity-to-value
   feature that was considered for this project.
7. **Historical route/hour delay rate (moving-window target encoding)**:
   for high-cardinality fields like `route` and `carrier`, encode using
   historical delay rate computed strictly from data before the cutoff
   (moving-window, not global — this is the same point-in-time discipline
   as everything else, just applied to encoding). This also IS your
   baseline model (see below) — reuse the computation.
8. **is_holiday**: simple binary flag for major US holidays. Keep it
   binary/simple; do not build a continuous proximity curve.

## Models

### Classification (delay probability)

- **Baseline**: predicted probability = the historical route/hour delay
  rate (feature #7 above, used directly as a prediction with no model on
  top). Both trained models must beat this, or the project has no story —
  make this comparison prominent in the README.
- **XGBoost or LightGBM**: primary tabular model, expected to perform best.
- **PyTorch neural net**: embedding layers for airport/carrier/route
  concatenated with the dense numeric features, feeding into a small MLP.
  This is what gets PyTorch onto the resume — it needs to be a real,
  reasonably-tuned model, not a token effort.

### Regression (expected delay minutes)

- **Baseline**: historical average delay minutes for that route/hour
  (among historically-delayed flights), same moving-window discipline as
  the classification baseline.
- **One trained model**: XGBoost/LightGBM regressor. Do not build a second
  PyTorch model for this — one solid regression model is enough; the
  resume value of PyTorch is already covered by the classifier.
- **Training data**: fit only on flights where the actual delay was 15+
  minutes (i.e. condition the regression on "given it's delayed, how
  late?"). Do not try to predict delay minutes for on-time flights — that's
  not a meaningful target.
- **Combining the two models at inference**: the API returns both numbers
  together (probability from the classifier, expected-minutes from the
  regressor, computed regardless of the classifier's output) and the
  caller reads them as a pair — e.g. "78% chance of delay, expected ~22
  minutes late." Do not multiply or otherwise mathematically combine the
  two outputs into a single number.

## Evaluation

- **Time-based split only.** Train on earlier months, test on later months.
  Never shuffle randomly across time — this would leak future information
  into training.
- **Report both ROC-AUC and PR-AUC.** With ~80/20 class imbalance, ROC-AUC
  alone can look good on a weak model — PR-AUC is the more honest metric
  here and must be reported alongside it, not instead of it.
- **Calibration**: check that predicted probabilities are honest (a plot or
  table — of the flights predicted ~30% likely to be delayed, are ~30%
  actually delayed?).
- **SHAP**: feature importance chart for the tree model. This becomes a
  README chart.
- **Plain-language metric**: something like "correctly flags X% of real
  delays at a Y% false-alarm rate," in addition to the AUC numbers.
- **Regression metrics**: MAE and RMSE for expected delay minutes, on the
  same time-based split, compared against the historical-average baseline.
  Report MAE as the plain-language number (e.g. "predicts delay length to
  within X minutes on average").
- Optional stretch: a drift check — train on year 1, test on year 2, show
  how much performance decays without retraining.

## Serving

- FastAPI app with `POST /predict`: takes raw feature inputs, returns both
  the delay probability (classifier) and expected delay minutes
  (regressor) in one response, e.g.
  `{"delay_probability": 0.78, "expected_delay_minutes": 22}`.
- No frontend required. If a demo is wanted later, a single-page Streamlit
  form is an acceptable optional add-on — do not build a React/web frontend.
- MLflow tracks every training run (params, metrics, model artifacts) so
  runs are comparable.

## Testing requirements

- **Leakage tests are mandatory, not optional.** For every point-in-time
  feature (tail-number propagation, hub backlog, target encoding, weather),
  write a test that asserts no record used in computing the feature has a
  timestamp after the cutoff. This is the most important test category in
  the project — it's also your best interview talking point, so it needs
  to actually exist and pass, not just be claimed in the README.
- API tests for the `/predict` endpoint (valid input → valid probability
  output; invalid input → clean error).
- A basic model sanity test (e.g. classifier predictions are in [0,1],
  regressor predictions are non-negative, both baselines are beaten on
  held-out data).

## Repo layout

```
flight-delay-prediction/
├── CLAUDE.md
├── README.md
├── data/               # scripts to pull/cache BTS + NOAA data
├── features/           # point-in-time feature engineering (Polars)
├── models/             # baseline, xgboost, pytorch training scripts
├── evaluation/         # metrics, calibration, SHAP, results tables
├── api/                # FastAPI serving app
├── tests/              # leakage tests, API tests, model sanity tests
├── mlruns/             # MLflow tracking output (gitignored, local)
├── Dockerfile           # optional, do last
└── requirements.txt
```

## Build order

Work in this order. Get each step correct and tested before moving to the
next — the feature pipeline is the part most likely to have subtle bugs, so
don't rush it.

Status key: `[x]` done & verified · `[~]` in progress / partial · `[ ]` not started.
Updated 2026-09-22.

1. [x] Data ingestion (BTS + NOAA, cached locally, scoped to chosen hub airports
   and date range) — hub list (ATL, DFW, DEN, ORD, LAX, JFK, LAS, MCO, MIA,
   CLT, SEA, PHX, EWR, SFO, IAH) confirmed with user. Date range: **2024-01–
   2024-12 only** (12 months), confirmed to stay on 2024 on 2026-09-23 after
   checking 2025 — BTS On-Time Performance itself looks fully published for
   2025 (TranStats reports data current through 2026-07), but NOAA LCD
   weather data for 2025 currently stops at 2025-08-25 across all 15 hub
   stations, so 2025 isn't usable as a full year yet; revisit once NOAA
   catches up. `data/ingest_bts.py` and `data/ingest_noaa.py` run clean
   against 2024-01–2024-12, confirmed complete and verified: `bts_ontime.parquet`
   (4,995,321 rows, all 12 months) and `noaa_weather.parquet` (152,846 rows,
   all 15 stations). `data/ingest_bts.py` also had a real timestamp-parsing
   bug fixed here (see step 2) and was rerun after that fix.
2. [x] Point-in-time feature pipeline in Polars — already used UTC (`*_utc`
   columns) correctly by the time it was checked on 2026-09-23; the "known
   UTC gap" above was stale. Running it against real 2024 data (for the
   first time) surfaced two real bugs, both fixed with regression tests
   added:
   (a) `data/ingest_bts.py`: an empty DepTime/ArrTime (cancelled flight)
   zfilled to "0000" and parsed as a fake midnight departure/arrival instead
   of null, making cancelled flights look like completed history.
   (b) A rarer BTS case — Cancelled=1 rows that still have a real DepTime
   (taxi/pushback before the cancellation) — has DepDelayMinutes null, which
   silently poisons the cum_sum-based windowed joins in `hub_backlog.py` and
   `target_encoding.py` at that exact row (Polars emits null, not the
   carried-forward total, at a null cum_sum input); fixed by explicitly
   excluding Cancelled=1 from history in both. `tail_propagation.py` also
   now excludes Diverted=1 from history (BTS keeps `Dest` as the originally
   scheduled airport even when the aircraft actually landed elsewhere).
   Verified on the real 2024 build (2,926,854 modeling rows): hub_backlog_pct,
   route_hour_delay_rate, carrier_delay_rate all now confirmed in [0, 1]
   (previously up to ~3% of rows were wildly out of range, e.g. 640.0).
   9 leakage/regression tests passing (was 8, all synthetic-only).
3. [x] Classification baseline + evaluation harness (time-split, ROC-AUC,
   PR-AUC, calibration) — run against real 2024 data. Time split: train
   2024-01-01–2024-09-15 (2,048,797 rows), test 2024-09-15–2025-01-01
   (878,057 rows). Baseline: ROC-AUC=0.626, PR-AUC=0.261, Brier=0.147,
   catch-rate@20%FPR=36.2%.
4. [x] XGBoost classifier: ROC-AUC=0.667, PR-AUC=0.320, Brier=0.138,
   catch-rate@20%FPR=41.2% — beats baseline on every metric.
   PyTorch classifier: ROC-AUC=0.677, PR-AUC=0.333, Brier=0.184,
   catch-rate@20%FPR=42.6% — beats baseline and XGBoost on ranking metrics,
   but has a *worse* Brier than both (poorly calibrated/overconfident; no
   calibration step like Platt scaling was applied). Real, reported
   tradeoff — see README.
5. [x] Evaluation run for real: calibration tables generated for both
   classifiers (XGBoost tracks true rate closely; PyTorch is overconfident
   at every decile). SHAP generated for both tree models (classifier:
   route_hour_delay_rate/carrier_delay_rate/hour_sin/hub_backlog_pct lead;
   regressor: doy_sin/hub_backlog_pct/carrier_delay_rate/tail_prior_delay_minutes
   lead).
6. [x] XGBoost regressor: MAE=45.41, RMSE=83.11 (n=156,567 delayed test
   flights) vs. baseline MAE=45.55, RMSE=85.15 — beats baseline, modestly
   (delay-length variance is dominated by information out of scope here,
   e.g. mechanical/crew/reactionary delays — see README).
7. [x] SHAP explainability — see step 5.
8. [x] MLflow tracking — verified working: `mlruns/0/` has one run per
   trained model (xgboost_classifier, xgboost_regressor, pytorch_classifier)
   with params/metrics/artifacts logged.
9. [x] FastAPI serving — `api/main.py` already implemented (not actually an
   empty placeholder; only `api/__init__.py` is). Tested end-to-end via
   TestClient against real trained artifacts: POST /predict with a real
   feature vector returned `{"delay_probability": 0.293,
   "expected_delay_minutes": 68.8}`; GET /health confirms models loaded.
10. [x] CI — `.github/workflows/tests.yml` already implemented (was
    incorrectly marked `[ ]`); runs `pytest tests/` on push/PR. The
    real-data-only tests (test_model_sanity.py) skip cleanly in CI since a
    fresh checkout has no ingested data/trained artifacts.
11. [x] Docker — `Dockerfile` already implemented (was incorrectly marked
    "empty placeholder"); built and run end-to-end on 2026-09-23 (Docker
    Desktop started for this specifically): `docker build` succeeded,
    `docker run` served real traffic, and `/health` + `/predict` returned
    output identical to the non-containerized API. Cleaned up after
    (container/image removed).
12. [x] README written: problem statement, weather look-ahead trap
    (including the BTS-DST-vs-NOAA-fixed-offset variant), tail-propagation
    and hub-backlog write-ups, classification + regression results tables,
    real SHAP charts and a calibration reliability chart (generated via
    `evaluation/report_xgboost.py` + `evaluation/report_torch.py`, run as
    two separate scripts — importing xgboost/shap and torch in the same
    process segfaults on this machine), example /predict output, and a
    "what was left out" section reusing the Scope list above.

Note on this file's own accuracy: steps 4, 6, 7, 8, 9, 10, 11 above were
previously marked `[ ]`/`[~]` despite `models/xgboost_classifier.py`,
`models/xgboost_regressor.py`, `models/torch_classifier.py`,
`evaluation/shap_report.py`, `api/main.py`, `.github/workflows/tests.yml`,
and `Dockerfile` already existing with real (not stub) implementations —
apparently from a prior session's work that was never reflected back into
this status tracker. Worth remembering: this file's checklist can drift from
the actual repo state; when in doubt, check the filesystem and run the
tests rather than trusting the checklist alone.

## Code style

- Type hints everywhere.
- Polars for pipeline/aggregation code; pandas only for small frames
  handed to sklearn/XGBoost/PyTorch if needed.
- Every point-in-time feature function should have a docstring stating its
  cutoff rule explicitly.
- No dependencies beyond the stack above without asking.

## README must include

- One-paragraph problem statement and the T-4h prediction framing
- Explanation of the weather look-ahead trap and how it was avoided
- Explanation of tail-number propagation and hub backlog index (the two
  headline features)
- Classification results table: baseline vs. XGBoost vs. neural net, with
  ROC-AUC, PR-AUC, and the plain-language delay-catch-rate number
- Regression results table: baseline vs. XGBoost/LightGBM, with MAE and
  RMSE for expected delay minutes
- SHAP charts and calibration plot
- Example output shown in the "78% chance of delay, expected ~22 minutes
  late" combined format
- A short "what I deliberately left out and why" note (crew connection
  risk, crosswind modeling, etc.) — this itself is a good signal of
  judgment, not a weakness to hide

## Working rules for Claude Code

- Stay inside the Features list and the scope list above. If an idea for a
  new feature or model comes up, stop and ask before building it.
- Write the leakage test for a feature in the same step you build the
  feature, not afterward.
- Run tests after each build step; don't move on with failing tests.
- If a design question comes up that this file doesn't answer, ask instead
  of guessing.