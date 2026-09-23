"""Generate SHAP charts for both XGBoost models and the XGBoost classifier's
calibration table.

Kept separate from report_torch.py deliberately: loading xgboost/shap and
torch in the same process segfaults on at least one dev machine (a duplicate-
OpenMP-runtime conflict between the two libraries' native extensions, not a
bug in this project's code). Each report script only imports one framework;
evaluation/plots.py's combined calibration chart is assembled by whichever
script runs second, from the other's saved table.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xgboost as xgb

from data.config import DELAY_THRESHOLD_MINUTES, PROCESSED_DIR
from evaluation.metrics import FEATURE_COLUMNS, calibration_table, time_based_split
from evaluation.plots import ARTIFACT_CAL_DIR, plot_calibration, plot_shap_bar
from evaluation.shap_report import mean_abs_shap_table, shap_values_for_model

ARTIFACT_DIR = ARTIFACT_CAL_DIR.parent


def main() -> None:
    frame = pl.read_parquet(PROCESSED_DIR / "features.parquet")
    _train, test = time_based_split(frame)

    clf = xgb.XGBClassifier()
    clf.load_model(ARTIFACT_DIR / "xgboost_classifier.json")
    values, _sample = shap_values_for_model(clf, test)
    plot_shap_bar(
        mean_abs_shap_table(values),
        "SHAP feature importance — XGBoost classifier",
        "shap_classifier.png",
    )

    X = test.select(list(FEATURE_COLUMNS)).to_numpy().astype(np.float64)
    proba = clf.predict_proba(X)[:, 1]
    y = test["is_delayed"].to_numpy()
    xgb_cal = calibration_table(y, proba, n_bins=10)
    xgb_cal.write_parquet(ARTIFACT_CAL_DIR / "calibration_xgboost.parquet")

    reg = xgb.XGBRegressor()
    reg.load_model(ARTIFACT_DIR / "xgboost_regressor.json")
    test_delayed = test.filter(pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES)
    rvalues, _rsample = shap_values_for_model(reg, test_delayed)
    plot_shap_bar(
        mean_abs_shap_table(rvalues),
        "SHAP feature importance — XGBoost regressor",
        "shap_regressor.png",
    )
    print("[report_xgboost] wrote shap_classifier.png, shap_regressor.png, calibration_xgboost.parquet")

    torch_cal_path = ARTIFACT_CAL_DIR / "calibration_pytorch.parquet"
    if torch_cal_path.exists():
        torch_cal = pl.read_parquet(torch_cal_path)
        plot_calibration({"XGBoost": xgb_cal, "PyTorch": torch_cal}, "calibration_classifiers.png")
        print("[report_xgboost] wrote calibration_classifiers.png")
    else:
        print("[report_xgboost] calibration_pytorch.parquet not found yet -- run report_torch.py to complete the combined chart")


if __name__ == "__main__":
    main()
