"""Classification and regression baselines from historical delay rates."""

from __future__ import annotations

import numpy as np
import polars as pl

from features.target_encoding import windowed_rate


def classification_baseline_proba(frame: pl.DataFrame) -> np.ndarray:
    """Predict delay probability as the route/hour historical delay rate.

    Requires column `route_hour_delay_rate` from features.target_encoding.
    """
    if "route_hour_delay_rate" not in frame.columns:
        raise ValueError("frame must include route_hour_delay_rate")
    return frame["route_hour_delay_rate"].to_numpy().astype(np.float64)


def regression_baseline_minutes(
    frame: pl.DataFrame,
    route_hour_avg_col: str = "route_hour_avg_delay_minutes",
) -> np.ndarray:
    """Predict expected delay minutes from historical route/hour averages.

    Intended for delayed flights only; callers should filter DepDelayMinutes >= 15
    before scoring. Missing averages fall back to 30 minutes.
    """
    if route_hour_avg_col not in frame.columns:
        raise ValueError(f"frame must include {route_hour_avg_col}")
    vals = frame[route_hour_avg_col].fill_null(30.0).to_numpy().astype(np.float64)
    return vals


def add_route_hour_avg_delay_minutes(
    flights: pl.DataFrame,
    lookback_days: int = 28,
    delay_threshold: float = 15.0,
) -> pl.DataFrame:
    """Moving-window average delay minutes among historically delayed peers.

    Cutoff rule: only delayed flights with actual_dep_dt < cutoff_dt contribute.

    Uses the same cumulative-sum as-of-join technique as
    features/target_encoding.py (see its module docstring) rather than a
    self-join, which is O(n^2) per route/hour group and doesn't finish at
    real BTS scale.
    """
    delayed_history = flights.filter(pl.col("DepDelayMinutes") >= delay_threshold)
    avgs = windowed_rate(
        flights,
        ["route", "hour_of_day"],
        "DepDelayMinutes",
        lookback_days,
        "route_hour_avg_delay_minutes",
        default=30.0,
        history_source=delayed_history,
    )
    return flights.join(avgs, on="flight_id", how="left")
