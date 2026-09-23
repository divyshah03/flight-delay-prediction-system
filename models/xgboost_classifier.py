"""XGBoost classifier: primary tabular model for delay probability.

Trained and evaluated on a strict time-based split (never shuffled across
time -- see evaluation/metrics.py:time_based_split) and compared against the
route/hour historical delay rate baseline, which must be beaten for the
project to have a story.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import xgboost as xgb

from data.config import PROCESSED_DIR
from evaluation.metrics import FEATURE_COLUMNS, classification_metrics, matrix_from_frame, time_based_split
from models.baseline import classification_baseline_proba

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "missing": np.nan,
}


def _xgboost_matrix(frame: pl.DataFrame) -> np.ndarray:
    """Like evaluation.metrics.matrix_from_frame, but keeps nulls as NaN --
    XGBoost's native missing-value handling is a better fit than the
    zero-fill matrix_from_frame uses for other consumers, since 0 is a valid
    value for several of these features (e.g. precip_in, ifr_flag)."""
    return frame.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)


def train_classifier(features_path: Path = PROCESSED_DIR / "features.parquet") -> dict:
    frame = pl.read_parquet(features_path)
    train, test = time_based_split(frame)

    X_train, y_train = _xgboost_matrix(train), train["is_delayed"].to_numpy()
    X_test, y_test = _xgboost_matrix(test), test["is_delayed"].to_numpy()

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    report = classification_metrics(y_test, proba)
    baseline_proba = classification_baseline_proba(test)
    baseline_report = classification_metrics(y_test, baseline_proba)

    with mlflow.start_run(run_name="xgboost_classifier"):
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_metrics(
            {
                "roc_auc": report.roc_auc,
                "pr_auc": report.pr_auc,
                "brier": report.brier,
                "catch_rate_at_20pct_fpr": report.catch_rate_at_fpr,
                "baseline_roc_auc": baseline_report.roc_auc,
                "baseline_pr_auc": baseline_report.pr_auc,
            }
        )
        model_path = ARTIFACT_DIR / "xgboost_classifier.json"
        model.save_model(model_path)
        mlflow.log_artifact(str(model_path))

    print(f"XGBoost classifier: ROC-AUC={report.roc_auc:.4f} PR-AUC={report.pr_auc:.4f}")
    print(f"Baseline:            ROC-AUC={baseline_report.roc_auc:.4f} PR-AUC={baseline_report.pr_auc:.4f}")
    print(f"Catch rate @20% FPR: model={report.catch_rate_at_fpr:.3f} baseline={baseline_report.catch_rate_at_fpr:.3f}")

    return {"model": model, "report": report, "baseline_report": baseline_report, "test": test, "y_test": y_test, "proba": proba}


if __name__ == "__main__":
    train_classifier()
