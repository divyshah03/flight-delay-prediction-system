"""XGBoost regressor: expected delay minutes, conditioned on the flight
actually being delayed. Trained only on flights with DepDelayMinutes >= 15 --
predicting delay length for an on-time flight isn't a meaningful target.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import xgboost as xgb

from data.config import DELAY_THRESHOLD_MINUTES, PROCESSED_DIR
from evaluation.metrics import FEATURE_COLUMNS, regression_metrics, time_based_split
from models.baseline import regression_baseline_minutes

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method": "hist",
}


def _matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)


def train_regressor(features_path: Path = PROCESSED_DIR / "features.parquet") -> dict:
    frame = pl.read_parquet(features_path)
    train, test = time_based_split(frame)

    train_delayed = train.filter(pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES)
    test_delayed = test.filter(pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES)

    X_train, y_train = _matrix(train_delayed), train_delayed["DepDelayMinutes"].to_numpy()
    X_test, y_test = _matrix(test_delayed), test_delayed["DepDelayMinutes"].to_numpy()

    model = xgb.XGBRegressor(**XGB_PARAMS, missing=np.nan)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    report = regression_metrics(y_test, pred)
    baseline_pred = regression_baseline_minutes(test_delayed)
    baseline_report = regression_metrics(y_test, baseline_pred)

    with mlflow.start_run(run_name="xgboost_regressor"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_metrics(
            {
                "mae": report.mae,
                "rmse": report.rmse,
                "baseline_mae": baseline_report.mae,
                "baseline_rmse": baseline_report.rmse,
            }
        )
        model_path = ARTIFACT_DIR / "xgboost_regressor.json"
        model.save_model(model_path)
        mlflow.log_artifact(str(model_path))

    print(f"XGBoost regressor: MAE={report.mae:.2f} RMSE={report.rmse:.2f} (n_delayed_test={len(y_test):,})")
    print(f"Baseline:          MAE={baseline_report.mae:.2f} RMSE={baseline_report.rmse:.2f}")

    return {
        "model": model,
        "report": report,
        "baseline_report": baseline_report,
        "test_delayed": test_delayed,
        "y_test": y_test,
        "pred": pred,
    }


if __name__ == "__main__":
    train_regressor()
