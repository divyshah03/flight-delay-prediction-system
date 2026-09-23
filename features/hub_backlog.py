"""Hub network backlog index.

Percentage of flights that departed the same origin airport delayed during the
3 hours immediately before the current flight's cutoff.

Implemented as a cumulative-count as-of join rather than a self-join: a naive
`join(on="Origin").filter(window)` self-join is O(n^2) per origin -- at real
BTS scale (a busy hub sees hundreds of thousands of departures over 2 years)
that is not just slow, it doesn't finish. Instead: sort each origin's history
by departure time, take a running cumulative count and cumulative delayed
count, then for any cutoff-relative window [a, b) the answer is
cumulative(b) - cumulative(a), found via two backward as-of joins. This is
O(n log n) and gives bit-identical results to the self-join definition it
replaces.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from data.config import DELAY_THRESHOLD_MINUTES

# Subtracted from a query boundary before the as-of join so that "backward,
# nearest at or before" becomes "strictly before" -- join_asof has no
# exclusive-bound mode, and history timestamps are minute-resolution, so a
# microsecond epsilon can never mask a real match while still excluding a
# row exactly on the boundary.
_EPSILON = timedelta(microseconds=1)


def add_hub_backlog(flights: pl.DataFrame, window_hours: int = 3) -> pl.DataFrame:
    """Add hub_backlog_pct: delayed share at origin in [cutoff - window, cutoff).

    Cutoff rule: only flights with actual_dep_dt < cutoff_dt and
    actual_dep_dt >= cutoff_dt - window_hours at the same Origin contribute.
    """
    required = {"flight_id", "Origin", "cutoff_dt", "actual_dep_dt", "DepDelayMinutes", "Cancelled"}
    missing = required - set(flights.columns)
    if missing:
        raise ValueError(f"flights missing columns for hub backlog: {sorted(missing)}")

    # Cancelled=1 rows are excluded explicitly, not just via actual_dep_dt
    # being null: a small number of BTS rows record a real DepTime (taxi/
    # pushback) for a flight ultimately cancelled, leaving DepDelayMinutes
    # null even though actual_dep_dt is populated -- and a null value_col
    # entry poisons the cum_sum-based windowed aggregate at that exact row
    # (Polars' cum_sum emits null, not the carried-forward total, at a null
    # input), which then corrupts every window whose as-of match lands on it.
    history = (
        flights.filter(pl.col("actual_dep_dt").is_not_null() & (pl.col("Cancelled") == 0))
        .select(
            "Origin",
            "actual_dep_dt",
            (pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES).cast(pl.Int64).alias("was_delayed"),
        )
        .sort(["Origin", "actual_dep_dt"])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("Origin").alias("cum_count"),
            pl.col("was_delayed").cum_sum().over("Origin").alias("cum_delayed"),
        )
    )

    bounds = flights.select(
        "flight_id",
        "Origin",
        (pl.col("cutoff_dt") - _EPSILON).alias("_upper_q"),
        (pl.col("cutoff_dt") - pl.duration(hours=window_hours) - _EPSILON).alias("_lower_q"),
    )

    def cumulative_as_of(query_col: str) -> pl.DataFrame:
        left = (
            bounds.select("flight_id", "Origin", query_col)
            .sort(["Origin", query_col])
        )
        joined = left.join_asof(
            history,
            left_on=query_col,
            right_on="actual_dep_dt",
            by="Origin",
            strategy="backward",
        )
        return joined.select(
            "flight_id",
            pl.col("cum_count").fill_null(0),
            pl.col("cum_delayed").fill_null(0),
        )
    upper = cumulative_as_of("_upper_q").rename({"cum_count": "upper_count", "cum_delayed": "upper_delayed"})
    lower = cumulative_as_of("_lower_q").rename({"cum_count": "lower_count", "cum_delayed": "lower_delayed"})

    window = (
        bounds.select("flight_id")
        .join(upper, on="flight_id", how="left")
        .join(lower, on="flight_id", how="left")
        .with_columns(
            (pl.col("upper_count") - pl.col("lower_count")).alias("hub_window_n"),
            (pl.col("upper_delayed") - pl.col("lower_delayed")).alias("hub_window_delayed"),
        )
        .with_columns(
            pl.when(pl.col("hub_window_n") > 0)
            .then(pl.col("hub_window_delayed") / pl.col("hub_window_n"))
            .otherwise(0.0)
            .alias("hub_backlog_pct")
        )
        .select("flight_id", "hub_window_n", "hub_backlog_pct")
    )

    return flights.join(window, on="flight_id", how="left").with_columns(
        pl.col("hub_backlog_pct").fill_null(0.0),
        pl.col("hub_window_n").fill_null(0),
    )
