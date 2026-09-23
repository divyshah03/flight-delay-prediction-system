# ✈️ Flight Delay Prediction

Predicts whether a US domestic flight will be delayed 15+ minutes — and if so, by how many — using only information available **4 hours before scheduled departure (T-4h)**.

> 🔮 **78% chance of delay, expected ~22 minutes late**

A portfolio project built to show real ML judgment, not a Kaggle-style "train XGBoost, report accuracy" exercise: point-in-time feature engineering with no leakage, a real baseline both trained models must beat, honest evaluation, and a model actually served behind an API.

**Scope:** top 15 US hub airports, full calendar year 2024, ~2.9M flights.

## 🌦️ The weather look-ahead trap

You can't use the weather observation *at* flight time to predict a flight before it departs — that's the weather that caused the delay, so using it is leakage.

**Fix:** every weather feature comes from the METAR observation at or before T-4h, joined via `join_asof(strategy="backward")` against the cutoff — never the actual departure time.

**Second trap:** BTS flight times are DST-aware local time; NOAA's LCD weather product is fixed local *standard* time year-round. Treating them as the same clock misaligns the weather join by an hour for ~8 months of the year. Both are converted to UTC independently (BTS via IANA timezone, NOAA via fixed offset) — see `data/ingest_bts.py` / `data/airports.py`.

## 🛫 The two headline features

**Tail-number delay propagation** — was this aircraft's most recent completed flight delayed, and by how much? An as-of join on `(Tail_Number, airport)`, matching a prior flight's `Dest` to the current flight's `Origin`, gated on the prior flight's actual *arrival* time (guarantees it's a known, final delay). Diverted flights are excluded from history — BTS keeps `Dest` as the originally-scheduled airport even when the plane actually landed elsewhere.

**Hub backlog index** — % of flights that departed the same origin airport delayed in the 3 hours before cutoff ("are delays cascading right now"). Computed as a cumulative-count as-of join, not a self-join (self-join is O(n²) per airport and doesn't finish at BTS scale).

Both features — plus the moving-window target encodings — are where the project's real bugs surfaced. See [What running this against real data caught](#-what-running-this-against-real-data-caught).

## 📊 Results

### Classification — delay probability

Time-based split: train Jan 1–Sep 15 2024 (70%, ~2.05M), test Sep 15 2024–Jan 1 2025 (30%, ~878K). Never shuffled across time.

| Model | ROC-AUC | PR-AUC | Brier | Catch-rate @ 20% FPR |
|---|---|---|---|---|
| Baseline (route/hour rate) | 0.626 | 0.261 | 0.147 | 36.2% |
| XGBoost | 0.667 | 0.320 | **0.138** | 41.2% |
| PyTorch (embeddings + MLP) | **0.670** | 0.320 | 0.198 | **41.4%** |

🎯 **Plain language:** at a 1-in-5 false-alarm rate, both trained models catch ~41% of real delays vs. 36% for "just use the historical rate."

PyTorch narrowly edges XGBoost on ROC-AUC/catch-rate and ties on PR-AUC, but has a clearly worse Brier score (0.198 vs 0.138) — its probabilities are less trustworthy even though it separates classes about as well. *(PyTorch trained with a fixed seed — `torch`/`numpy`/`random` + a seeded `DataLoader` generator — so these numbers reproduce exactly on rerun; confirmed via two independent runs producing bit-identical metrics.)*

<img src="evaluation/plots/calibration_classifiers.png" alt="Calibration: XGBoost vs PyTorch" width="420">

XGBoost tracks the diagonal closely; PyTorch sits below it at every decile — it systematically overstates risk.

<details>
<summary><b>XGBoost calibration deciles</b></summary>

| Predicted | Actual | n |
|---|---|---|
| 0.050 | 0.064 | 87,806 |
| 0.070 | 0.091 | 87,806 |
| 0.086 | 0.115 | 87,806 |
| 0.101 | 0.131 | 87,806 |
| 0.117 | 0.151 | 87,806 |
| 0.134 | 0.172 | 87,806 |
| 0.155 | 0.192 | 87,806 |
| 0.184 | 0.219 | 87,805 |
| 0.229 | 0.263 | 87,805 |
| 0.367 | 0.384 | 87,805 |

</details>

<details>
<summary><b>PyTorch calibration deciles</b></summary>

| Predicted | Actual | n |
|---|---|---|
| 0.141 | 0.061 | 87,806 |
| 0.215 | 0.091 | 87,806 |
| 0.267 | 0.115 | 87,806 |
| 0.313 | 0.131 | 87,806 |
| 0.361 | 0.150 | 87,806 |
| 0.413 | 0.169 | 87,806 |
| 0.470 | 0.193 | 87,806 |
| 0.534 | 0.224 | 87,805 |
| 0.612 | 0.265 | 87,805 |
| 0.747 | 0.384 | 87,805 |

</details>

No calibration step (e.g. Platt scaling) was applied to PyTorch — consistent with its worse Brier score. For a real deployment showing the probability *number* to a person, that argues for serving XGBoost's probability despite PyTorch's marginal edge on ranking — a genuine tradeoff, not a clean win either way.

### Regression — expected delay minutes (given a delay)

Trained/evaluated only on flights with `DepDelayMinutes >= 15` (156,567 in test) — predicting delay length for an on-time flight isn't meaningful.

| Model | MAE (min) | RMSE (min) |
|---|---|---|
| Baseline (route/hour average) | 45.55 | 85.15 |
| XGBoost | **45.41** | **83.11** |

🎯 **Plain language:** given a flight is delayed, the model predicts how late to within ~45 minutes — a modest edge over the 46-minute baseline. Once a flight is delayed at all, most remaining variance (mechanical, crew, reactionary delays) is information this project deliberately doesn't model — so this small edge is the honest result, not a shortfall.

### 🔍 SHAP feature importance

**Classifier**

<img src="evaluation/plots/shap_classifier.png" alt="SHAP — XGBoost classifier" width="380">

| Feature | Mean \|SHAP\| |
|---|---|
| route_hour_delay_rate | 0.334 |
| carrier_delay_rate | 0.258 |
| hour_sin | 0.247 |
| hub_backlog_pct | 0.210 |
| doy_cos / doy_sin | 0.141 |
| origin_ceiling_ft | 0.093 |
| tail_prior_delay_minutes | 0.074 |

**Regressor**

<img src="evaluation/plots/shap_regressor.png" alt="SHAP — XGBoost regressor" width="380">

| Feature | Mean \|SHAP\| |
|---|---|
| doy_sin | 5.08 |
| hub_backlog_pct | 4.90 |
| carrier_delay_rate | 4.20 |
| tail_prior_delay_minutes | 3.97 |
| doy_cos | 3.61 |
| route_hour_delay_rate | 2.76 |
| origin_temperature_f | 2.47 |
| origin_ceiling_ft | 1.61 |

Target encodings dominate classification (unsurprising — same signal as the baseline, refined). For *how late*, the two headline features and seasonality matter more than the route.

## 🚀 Example output

```
POST /predict
{
  "hour_sin": 0.5, "hour_cos": 0.87, "doy_sin": 0.1, "doy_cos": 0.99,
  "is_holiday": 0,
  "origin_temperature_f": 45.0, "origin_wind_speed_kt": 12.0,
  "origin_precip_in": 0.0, "origin_visibility_mi": 10.0, "origin_ceiling_ft": 5000.0,
  "ifr_flag": 0, "tail_prior_delay_minutes": 25.0, "tail_prior_delayed": 1,
  "hub_backlog_pct": 0.3, "route_hour_delay_rate": 0.25, "carrier_delay_rate": 0.22
}

200 OK
{ "delay_probability": 0.293, "expected_delay_minutes": 68.8 }
```

→ "29% chance of delay, expected ~69 minutes late." The two numbers are computed independently and returned together.

## 🐛 What running this against real data caught

Two real bugs surfaced running the pipeline against real 2024 BTS data for the first time:

1. **Cancelled-flight parsing**: BTS represents a cancelled flight's `DepTime` as an empty string. `str.zfill(4)` on `""` → `"0000"`, silently parsed as a real midnight departure instead of null — making cancelled flights look like completed history to every downstream feature.
2. **Null-poisoned cumulative joins**: some `Cancelled=1` rows still carry a real `DepTime` but null `DepDelayMinutes`. Polars' `cum_sum()` emits `null` (not the carried-forward total) at that exact row — silently resetting the running count for hub backlog / target encoding. Result: `hub_backlog_pct` (should be in [0, 1]) hit values as high as 640 for ~3% of flights before the fix.

Both fixed (exclude `Cancelled=1` from all historical aggregates) and covered by regression tests. Diverted-flight handling in tail propagation was found and fixed the same way.

## ✂️ What I deliberately left out, and why

- **Crew connection risk / tight-turn modeling** — crew scheduling data isn't realistically available outside an airline.
- **Crosswind via runway-heading trigonometry** — real signal, but out of scope for this size project.
- **Weather forecast divergence deltas** — legitimate stretch goal, not required.
- **Continuous holiday-proximity curves** — a binary `is_holiday` flag is enough.
- **Airport capacity tiering** beyond what's a byproduct of hub backlog.
- **Multi-airline crew/aircraft rotation graphs** — a materially larger project.
- **Real-time/streaming inference** — this is a batch, historical project.

Rule applied: if a feature idea couldn't be explained correctly in one sentence, it didn't belong here.

## 🏗️ Architecture

```
data/         BTS + NOAA ingestion (Polars)
features/     Point-in-time pipeline: calendar, weather-at-cutoff,
              tail propagation, hub backlog, target encoding
models/       Baseline, XGBoost classifier + regressor, PyTorch classifier
evaluation/   Time-based split, metrics, SHAP
api/          FastAPI POST /predict
tests/        Leakage tests + API tests + model sanity tests
```

```bash
python -m data.ingest_bts
python -m data.ingest_noaa
python -m features.build
python -m models.xgboost_classifier
python -m models.xgboost_regressor
python -m models.torch_classifier
python -m evaluation.report_xgboost   # SHAP charts + XGBoost calibration
python -m evaluation.report_torch     # PyTorch calibration + combined chart
uvicorn api.main:app --reload
```

`report_xgboost` and `report_torch` are deliberately two separate scripts,
run in either order: importing `xgboost`/`shap` and `torch` in the same
process segfaults on at least one dev machine (a native duplicate-OpenMP-
runtime conflict between the two libraries, unrelated to this project's own
code). Each writes its model's calibration table to `artifacts/calibration/`;
whichever script runs second finds the other's table there and produces the
combined `evaluation/plots/calibration_classifiers.png`.

MLflow tracks every training run's params, metrics, and model artifacts
(`mlruns/`, local file store). Run `mlflow ui` from the project root to
browse them.
