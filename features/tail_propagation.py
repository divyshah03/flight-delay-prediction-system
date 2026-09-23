"""Tail-number delay propagation feature.

For each flight, look up the most recent completed flight of the same aircraft
(Tail_Number) whose actual departure was strictly before this flight's cutoff.
"""

from __future__ import annotations

import polars as pl

from data.config import DELAY_THRESHOLD_MINUTES


def add_tail_propagation(flights: pl.DataFrame) -> pl.DataFrame:
    """Add prior-tail delay flag and minutes for each flight.

    Cutoff rule: only prior flights with actual_dep_dt < cutoff_dt (of the
    current flight) may contribute. Implemented via a self join filtered by
    that inequality, then taking the latest prior per flight.
    """
    required = {"flight_id", "Tail_Number", "cutoff_dt", "actual_dep_dt", "DepDelayMinutes"}
    missing = required - set(flights.columns)
    if missing:
        raise ValueError(f"flights missing columns for tail propagation: {sorted(missing)}")

    current = flights.select(
        [
            pl.col("flight_id"),
            pl.col("Tail_Number"),
            pl.col("cutoff_dt"),
        ]
    )

    prior = flights.select(
        [
            pl.col("flight_id").alias("prior_flight_id"),
            pl.col("Tail_Number"),
            pl.col("actual_dep_dt").alias("prior_actual_dep_dt"),
            pl.col("DepDelayMinutes").alias("prior_delay_minutes"),
        ]
    )

    candidates = (
        current.join(prior, on="Tail_Number", how="left")
        .filter(
            pl.col("prior_actual_dep_dt").is_not_null()
            & (pl.col("prior_actual_dep_dt") < pl.col("cutoff_dt"))
            & (pl.col("prior_flight_id") != pl.col("flight_id"))
        )
        .sort(["flight_id", "prior_actual_dep_dt"], descending=[False, True])
        .unique(subset=["flight_id"], keep="first")
        .select(
            [
                "flight_id",
                pl.col("prior_delay_minutes"),
                pl.col("prior_actual_dep_dt"),
            ]
        )
    )

    return (
        flights.join(candidates, on="flight_id", how="left")
        .with_columns(
            pl.col("prior_delay_minutes").fill_null(0.0).alias("tail_prior_delay_minutes"),
            (
                pl.col("prior_delay_minutes").fill_null(0.0) >= DELAY_THRESHOLD_MINUTES
            )
            .cast(pl.Int8)
            .alias("tail_prior_delayed"),
        )
        .drop(["prior_delay_minutes", "prior_actual_dep_dt"], strict=False)
    )
