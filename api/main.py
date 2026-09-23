"""FastAPI serving app: POST /predict returns delay probability (XGBoost
classifier, the best-performing model per evaluation) and expected delay
minutes (XGBoost regressor) together, computed independently of each other.

Scope note: this endpoint takes an already-computed feature vector (the same
FEATURE_COLUMNS used in training), not raw booking info -- computing the
point-in-time features live (tail-number history, hub backlog, T-4h weather)
would need an online feature store and live data feeds, which is a separate,
much larger productionization project outside this one's scope.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evaluation.metrics import FEATURE_COLUMNS

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

_classifier: xgb.XGBClassifier | None = None
_regressor: xgb.XGBRegressor | None = None


def _load_models() -> None:
    global _classifier, _regressor
    clf_path = ARTIFACT_DIR / "xgboost_classifier.json"
    reg_path = ARTIFACT_DIR / "xgboost_regressor.json"
    if not clf_path.exists() or not reg_path.exists():
        raise RuntimeError(
            f"Model artifacts not found in {ARTIFACT_DIR}. Run "
            "`python -m models.xgboost_classifier` and `python -m models.xgboost_regressor` first."
        )
    clf = xgb.XGBClassifier()
    clf.load_model(clf_path)
    reg = xgb.XGBRegressor()
    reg.load_model(reg_path)
    _classifier, _regressor = clf, reg


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(title="Flight Delay Prediction API", lifespan=lifespan)


class PredictRequest(BaseModel):
    hour_sin: float
    hour_cos: float
    doy_sin: float
    doy_cos: float
    is_holiday: int = Field(ge=0, le=1)
    origin_temperature_f: float | None = None
    origin_wind_speed_kt: float | None = None
    origin_precip_in: float | None = None
    origin_visibility_mi: float | None = None
    origin_ceiling_ft: float | None = None
    dest_temperature_f: float | None = None
    dest_wind_speed_kt: float | None = None
    dest_precip_in: float | None = None
    dest_visibility_mi: float | None = None
    dest_ceiling_ft: float | None = None
    ifr_flag: int = Field(ge=0, le=1)
    tail_prior_delay_minutes: float
    tail_prior_delayed: int = Field(ge=0, le=1)
    hub_backlog_pct: float = Field(ge=0.0, le=1.0)
    route_hour_delay_rate: float = Field(ge=0.0, le=1.0)
    carrier_delay_rate: float = Field(ge=0.0, le=1.0)

    def to_row(self) -> np.ndarray:
        values = [getattr(self, col) for col in FEATURE_COLUMNS]
        return np.array([[v if v is not None else np.nan for v in values]], dtype=np.float64)


class PredictResponse(BaseModel):
    delay_probability: float
    expected_delay_minutes: float


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _classifier is None or _regressor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    row = request.to_row()
    proba = float(_classifier.predict_proba(row)[0, 1])
    expected_minutes = float(_regressor.predict(row)[0])
    return PredictResponse(
        delay_probability=proba,
        expected_delay_minutes=max(expected_minutes, 0.0),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": _classifier is not None and _regressor is not None}
