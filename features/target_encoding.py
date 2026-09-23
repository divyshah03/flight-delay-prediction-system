"""Moving-window target encoding for route and carrier delay rates.

This computation is also the classification baseline: predicted probability =
historical route/hour delay rate using only pre-cutoff history.
"""

from __future__ import annotations

import polars as pl

from data.config import DELAY_THRESHOLD_MINUTES


def add_moving_target_encoding(
    flights: pl.DataFrame,
    lookback_days: int = 28,
) -> pl.DataFrame:
    """Encode route and carrier with pre-cutoff historical delay rates.

    Cutoff rule: only flights with actual_dep_dt < cutoff_dt and within the
    lookback window contribute to the rate for the current flight. Rates are
    also conditioned on scheduled hour_of_day for the route/hour baseline.
    """
    required = {
        "flight_id",
        "route",
        "Reporting_Airline",
        "hour_of_day",
        "cutoff_dt",
        "actual_dep_dt",
        "DepDelayMinutes",
    }
    missing = required - set(flights.columns)
    if missing:
        raise ValueError(f"flights missing columns for target encoding: {sorted(missing)}")

    labeled = flights.with_columns(
        (pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES).cast(pl.Int8).alias("is_delayed")
    )

    current = labeled.select(
        ["flight_id", "route", "Reporting_Airline", "hour_of_day", "cutoff_dt"]
    )
    history = labeled.select(
        [
            pl.col("route"),
            pl.col("Reporting_Airline"),
            pl.col("hour_of_day"),
            pl.col("actual_dep_dt"),
            pl.col("is_delayed"),
            pl.col("flight_id").alias("hist_flight_id"),
        ]
    )

    route_hour = (
        current.join(history, on=["route", "hour_of_day"], how="left")
        .filter(
            pl.col("actual_dep_dt").is_not_null()
            & (pl.col("actual_dep_dt") < pl.col("cutoff_dt"))
            & (
                pl.col("actual_dep_dt")
                >= (pl.col("cutoff_dt") - pl.duration(days=lookback_days))
            )
            & (pl.col("hist_flight_id") != pl.col("flight_id"))
        )
        .group_by("flight_id")
        .agg(pl.col("is_delayed").mean().alias("route_hour_delay_rate"))
    )

    carrier = (
        current.join(history, on="Reporting_Airline", how="left")
        .filter(
            pl.col("actual_dep_dt").is_not_null()
            & (pl.col("actual_dep_dt") < pl.col("cutoff_dt"))
            & (
                pl.col("actual_dep_dt")
                >= (pl.col("cutoff_dt") - pl.duration(days=lookback_days))
            )
            & (pl.col("hist_flight_id") != pl.col("flight_id"))
        )
        .group_by("flight_id")
        .agg(pl.col("is_delayed").mean().alias("carrier_delay_rate"))
    )

    return (
        flights.join(route_hour, on="flight_id", how="left")
        .join(carrier, on="flight_id", how="left")
        .with_columns(
            pl.col("route_hour_delay_rate").fill_null(0.2),
            pl.col("carrier_delay_rate").fill_null(0.2),
        )
    )
