"""API tests for POST /predict.

Uses tiny synthetic XGBoost models (not the real trained artifacts) so this
suite has no dependency on the multi-GB ingested dataset and can run in CI on
every push. See tests/test_model_sanity.py for the real-artifact integration
checks, which are skipped when the real pipeline hasn't been run.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from fastapi.testclient import TestClient

import api.main as api_main
from evaluation.metrics import FEATURE_COLUMNS

VALID_PAYLOAD = {
    "hour_sin": 0.5,
    "hour_cos": 0.86,
    "doy_sin": 0.1,
    "doy_cos": 0.99,
    "is_holiday": 0,
    "origin_temperature_f": 72.0,
    "origin_wind_speed_kt": 10.0,
    "origin_precip_in": 0.0,
    "origin_visibility_mi": 10.0,
    "origin_ceiling_ft": 5000.0,
    "dest_temperature_f": 68.0,
    "dest_wind_speed_kt": 8.0,
    "dest_precip_in": 0.0,
    "dest_visibility_mi": 10.0,
    "dest_ceiling_ft": 4000.0,
    "ifr_flag": 0,
    "tail_prior_delay_minutes": 5.0,
    "tail_prior_delayed": 0,
    "hub_backlog_pct": 0.1,
    "route_hour_delay_rate": 0.2,
    "carrier_delay_rate": 0.18,
}


def _make_dummy_models() -> tuple[xgb.XGBClassifier, xgb.XGBRegressor]:
    rng = np.random.default_rng(0)
    n = 64
    X = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    y_cls = rng.integers(0, 2, size=n)
    y_reg = rng.uniform(0, 60, size=n)

    clf = xgb.XGBClassifier(n_estimators=5, max_depth=2)
    clf.fit(X, y_cls)
    reg = xgb.XGBRegressor(n_estimators=5, max_depth=2)
    reg.fit(X, y_reg)
    return clf, reg


def _client_with_dummy_models() -> TestClient:
    clf, reg = _make_dummy_models()
    api_main._classifier = clf
    api_main._regressor = reg
    return TestClient(api_main.app)


def test_predict_valid_input_returns_bounded_probability() -> None:
    client = _client_with_dummy_models()
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["delay_probability"] <= 1.0
    assert body["expected_delay_minutes"] >= 0.0


def test_predict_missing_required_field_returns_422() -> None:
    client = _client_with_dummy_models()
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["hour_sin"]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_out_of_range_field_returns_422() -> None:
    client = _client_with_dummy_models()
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["hub_backlog_pct"] = 1.5  # must be in [0, 1]
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_predict_wrong_type_returns_422() -> None:
    client = _client_with_dummy_models()
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["is_holiday"] = "not-a-number"
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422


def test_health_endpoint() -> None:
    client = _client_with_dummy_models()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["models_loaded"] is True
