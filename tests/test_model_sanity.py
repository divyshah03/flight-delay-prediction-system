"""Model sanity checks against the real trained artifacts and real ingested
data. Skipped (not failed) when that data/those artifacts don't exist -- a
fresh clone's CI run doesn't download the multi-GB BTS+NOAA dataset or train
real models, so these are integration checks for local runs after the full
pipeline has executed, not part of the fast per-push suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from data.config import DELAY_THRESHOLD_MINUTES, PROCESSED_DIR
from evaluation.metrics import FEATURE_COLUMNS, classification_metrics, regression_metrics, time_based_split
from models.baseline import classification_baseline_proba, regression_baseline_minutes

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
CLASSIFIER_PATH = ARTIFACT_DIR / "xgboost_classifier.json"
REGRESSOR_PATH = ARTIFACT_DIR / "xgboost_regressor.json"

pytestmark = pytest.mark.skipif(
    not (FEATURES_PATH.exists() and CLASSIFIER_PATH.exists() and REGRESSOR_PATH.exists()),
    reason="real features/model artifacts not present -- run the full ingestion+feature+training pipeline first",
)


@pytest.fixture(scope="module")
def held_out_split():
    frame = pl.read_parquet(FEATURES_PATH)
    _train, test = time_based_split(frame)
    return test


def test_classifier_predictions_in_unit_interval(held_out_split) -> None:
    clf = xgb.XGBClassifier()
    clf.load_model(CLASSIFIER_PATH)
    X = held_out_split.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    proba = clf.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_regressor_predictions_non_negative(held_out_split) -> None:
    reg = xgb.XGBRegressor()
    reg.load_model(REGRESSOR_PATH)
    delayed = held_out_split.filter(pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES)
    X = delayed.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    pred = reg.predict(X)
    assert (pred >= 0.0).all()


def test_classifier_beats_baseline_on_held_out_data(held_out_split) -> None:
    clf = xgb.XGBClassifier()
    clf.load_model(CLASSIFIER_PATH)
    X = held_out_split.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    y = held_out_split["is_delayed"].to_numpy()

    model_report = classification_metrics(y, clf.predict_proba(X)[:, 1])
    baseline_report = classification_metrics(y, classification_baseline_proba(held_out_split))

    assert model_report.pr_auc >= baseline_report.pr_auc
    assert model_report.roc_auc >= baseline_report.roc_auc


def test_regressor_beats_baseline_on_held_out_data(held_out_split) -> None:
    reg = xgb.XGBRegressor()
    reg.load_model(REGRESSOR_PATH)
    delayed = held_out_split.filter(pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES)
    X = delayed.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    y = delayed["DepDelayMinutes"].to_numpy()

    model_report = regression_metrics(y, reg.predict(X))
    baseline_report = regression_metrics(y, regression_baseline_minutes(delayed))

    assert model_report.mae <= baseline_report.mae
