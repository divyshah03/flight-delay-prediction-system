"""Hub network backlog index.

Percentage of flights that departed the same origin airport delayed during the
3 hours immediately before the current flight's cutoff.
"""

from __future__ import annotations

import polars as pl

from data.config import DELAY_THRESHOLD_MINUTES


def add_hub_backlog(flights: pl.DataFrame, window_hours: int = 3) -> pl.DataFrame:
    """Add hub_backlog_pct: delayed share at origin in [cutoff - window, cutoff).

    Cutoff rule: only flights with actual_dep_dt < cutoff_dt and
    actual_dep_dt >= cutoff_dt - window_hours at the same Origin contribute.
    """
    required = {"flight_id", "Origin", "cutoff_dt", "actual_dep_dt", "DepDelayMinutes"}
    missing = required - set(flights.columns)
    if missing:
        raise ValueError(f"flights missing columns for hub backlog: {sorted(missing)}")

    current = flights.select(["flight_id", "Origin", "cutoff_dt"])
    history = flights.select(
        [
            pl.col("Origin"),
            pl.col("actual_dep_dt"),
            pl.col("DepDelayMinutes"),
            pl.col("flight_id").alias("hist_flight_id"),
        ]
    )

    window = (
        current.join(history, on="Origin", how="left")
        .filter(
            pl.col("actual_dep_dt").is_not_null()
            & (pl.col("actual_dep_dt") < pl.col("cutoff_dt"))
            & (
                pl.col("actual_dep_dt")
                >= (pl.col("cutoff_dt") - pl.duration(hours=window_hours))
            )
            & (pl.col("hist_flight_id") != pl.col("flight_id"))
        )
        .with_columns(
            (pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES).cast(pl.Int8).alias("was_delayed")
        )
        .group_by("flight_id")
        .agg(
            pl.len().alias("hub_window_n"),
            pl.col("was_delayed").mean().alias("hub_backlog_pct"),
        )
    )

    return flights.join(window, on="flight_id", how="left").with_columns(
        pl.col("hub_backlog_pct").fill_null(0.0),
        pl.col("hub_window_n").fill_null(0),
    )
