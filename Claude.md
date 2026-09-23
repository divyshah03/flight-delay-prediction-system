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

1. Data ingestion: BTS On-Time Performance + NOAA weather
2. Point-in-time feature pipeline (see Features below — this is the core of
   the project)
3. Baseline models (historical delay rate lookup for classification;
   historical average delay minutes for regression)
4. Two trained classification models (XGBoost/LightGBM and a PyTorch
   neural net) predicting delay probability, PLUS one regression model
   (XGBoost/LightGBM is fine — do not build a second PyTorch model for
   this) predicting expected delay minutes, trained only on flights that
   were actually delayed
5. Evaluation: time-based split, ROC-AUC + PR-AUC + calibration for
   classification, MAE/RMSE for regression, SHAP for both
6. Serving: a FastAPI `/predict` endpoint
7. MLflow experiment tracking
8. Tests: no-leakage checks + API tests
9. CI (GitHub Actions running tests on push)
10. Docker (nice-to-have polish, not core — do this last, and skip it
    entirely if time is short)

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
- Scope: top 10-15 US hub airports, 1-2 years of data. This is still
  millions of rows — do not expand beyond this without asking.
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

1. Data ingestion (BTS + NOAA, cached locally, scoped to chosen hub airports
   and date range)
2. Point-in-time feature pipeline in Polars, with leakage tests written
   alongside each feature as it's built (not after)
3. Classification baseline + evaluation harness (time-split, ROC-AUC,
   PR-AUC, calibration)
4. XGBoost/LightGBM classifier, compared against baseline
5. PyTorch neural net classifier, compared against both
6. Regression baseline + XGBoost/LightGBM regressor for delay minutes
   (trained only on delayed flights), evaluated with MAE/RMSE against its
   baseline
7. SHAP explainability for the classifier and the regressor
8. MLflow tracking wired into all of the above
9. FastAPI serving returning both outputs together
10. CI (GitHub Actions running pytest)
11. Docker (optional, last)
12. README with results tables (classification + regression), SHAP charts,
    calibration plot, and a short write-up of the tail-number and
    hub-backlog features and how leakage was avoided

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