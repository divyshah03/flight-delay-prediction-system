"""SHAP explainability for the tree models (classifier + regressor).

Per project scope, SHAP is only required for "the tree model" -- the PyTorch
classifier isn't included here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import shap
import xgboost as xgb

from evaluation.metrics import FEATURE_COLUMNS

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def shap_values_for_model(model: xgb.XGBModel, frame: pl.DataFrame, sample_size: int = 5000) -> tuple[np.ndarray, pl.DataFrame]:
    """TreeExplainer SHAP values for an XGBoost model, on a sample of `frame`
    (SHAP on the full multi-million-row test set is unnecessary for a
    feature-importance chart and considerably slower)."""
    sample = frame.sample(n=min(sample_size, frame.height), seed=42)
    X = sample.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return values, sample


def mean_abs_shap_table(values: np.ndarray) -> pl.DataFrame:
    """Feature -> mean(|SHAP value|), sorted descending -- the table the
    README's SHAP chart is built from."""
    importance = np.abs(values).mean(axis=0)
    return pl.DataFrame(
        {"feature": list(FEATURE_COLUMNS), "mean_abs_shap": importance}
    ).sort("mean_abs_shap", descending=True)


def save_shap_report(model: xgb.XGBModel, frame: pl.DataFrame, name: str) -> Path:
    values, _sample = shap_values_for_model(model, frame)
    table = mean_abs_shap_table(values)
    out_path = ARTIFACT_DIR / f"shap_{name}.parquet"
    table.write_parquet(out_path)
    print(f"[shap] {name} top features:\n{table.head(8)}")
    return out_path
