"""Time-based split helpers and classification/regression metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationReport:
    roc_auc: float
    pr_auc: float
    brier: float
    catch_rate_at_fpr: float
    fpr_target: float


@dataclass(frozen=True)
class RegressionReport:
    mae: float
    rmse: float


def time_based_split(
    frame: pl.DataFrame,
    time_col: str = "scheduled_dep_dt",
    train_frac: float = 0.7,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split by chronological order — never shuffle across time."""
    ordered = frame.sort(time_col)
    n_train = int(ordered.height * train_frac)
    return ordered.head(n_train), ordered.tail(ordered.height - n_train)


def classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fpr_target: float = 0.20,
) -> ClassificationReport:
    """ROC-AUC, PR-AUC, Brier, and delay catch-rate at a fixed false-alarm rate."""
    roc = float(roc_auc_score(y_true, y_proba))
    pr = float(average_precision_score(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))
    catch = _catch_rate_at_fpr(y_true, y_proba, fpr_target=fpr_target)
    return ClassificationReport(
        roc_auc=roc,
        pr_auc=pr,
        brier=brier,
        catch_rate_at_fpr=catch,
        fpr_target=fpr_target,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionReport:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return RegressionReport(mae=mae, rmse=rmse)


def _catch_rate_at_fpr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fpr_target: float,
) -> float:
    """Share of true delays caught when operating at approximately fpr_target FPR."""
    neg = y_proba[y_true == 0]
    if neg.size == 0:
        return 0.0
    threshold = float(np.quantile(neg, 1.0 - fpr_target))
    preds = y_proba >= threshold
    positives = y_true == 1
    if positives.sum() == 0:
        return 0.0
    return float(preds[positives].mean())


def calibration_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Mean predicted probability vs. empirical delay rate per quantile bin."""
    order = np.argsort(y_proba)
    bins = np.array_split(order, n_bins)
    rows: list[dict[str, float]] = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": float(i),
                "mean_predicted": float(y_proba[idx].mean()),
                "frac_delayed": float(y_true[idx].mean()),
                "n": float(len(idx)),
            }
        )
    return pl.DataFrame(rows)


FEATURE_COLUMNS: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "is_holiday",
    "origin_temp_c",
    "origin_wind_speed_kt",
    "origin_precip_in",
    "origin_visibility_mi",
    "origin_ceiling_ft",
    "dest_temp_c",
    "dest_wind_speed_kt",
    "dest_precip_in",
    "dest_visibility_mi",
    "dest_ceiling_ft",
    "ifr_flag",
    "tail_prior_delay_minutes",
    "tail_prior_delayed",
    "hub_backlog_pct",
    "route_hour_delay_rate",
    "carrier_delay_rate",
)


def matrix_from_frame(
    frame: pl.DataFrame,
    columns: Iterable[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    cols = list(columns)
    return frame.select(cols).fill_null(0.0).to_numpy().astype(np.float32)
