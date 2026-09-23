"""Moving-window target encoding for route and carrier delay rates.

This computation is also the classification baseline: predicted probability =
historical route/hour delay rate using only pre-cutoff history.

Implemented as a cumulative-count as-of join rather than a self-join, for the
same reason as features/hub_backlog.py: a naive self-join on route+hour or on
carrier is O(n^2) per group, and a popular route/carrier over 2 years of BTS
data makes that join intractable. See hub_backlog.py's module docstring for
the general technique; this applies it with a day-scale lookback window
instead of an hour-scale one.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from data.config import DELAY_THRESHOLD_MINUTES

_EPSILON = timedelta(microseconds=1)


def windowed_rate(
    flights: pl.DataFrame,
    group_cols: list[str],
    value_col: str,
    lookback_days: int,
    out_col: str,
    default: float,
    history_source: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Mean of `value_col` over `group_cols` history in
    [cutoff_dt - lookback_days, cutoff_dt), via cumulative-sum as-of joins.

    `history_source` lets the aggregated population differ from the current
    population being scored -- e.g. the regression baseline in
    models/baseline.py averages only over historically-DELAYED peers, not
    every peer. Defaults to `flights` itself (target encoding's case).
    """
    # Cancelled=1 rows are excluded explicitly, not just via actual_dep_dt
    # being null -- see features/hub_backlog.py's module comment on the same
    # filter: a handful of BTS rows have a real DepTime despite Cancelled=1,
    # leaving value_col null, which poisons the cum_sum-based aggregate at
    # that row for every window whose as-of match lands on it.
    history_source = flights if history_source is None else history_source
    history = (
        history_source.filter(pl.col("actual_dep_dt").is_not_null() & (pl.col("Cancelled") == 0))
        .select(*group_cols, "actual_dep_dt", value_col)
        .sort([*group_cols, "actual_dep_dt"])
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(group_cols).alias("cum_count"),
            pl.col(value_col).cum_sum().over(group_cols).alias("cum_value"),
        )
    )

    bounds = flights.select(
        "flight_id",
        *group_cols,
        (pl.col("cutoff_dt") - _EPSILON).alias("_upper_q"),
        (pl.col("cutoff_dt") - pl.duration(days=lookback_days) - _EPSILON).alias("_lower_q"),
    )

    def cumulative_as_of(query_col: str) -> pl.DataFrame:
        left = bounds.select("flight_id", *group_cols, query_col).sort([*group_cols, query_col])
        joined = left.join_asof(
            history,
            left_on=query_col,
            right_on="actual_dep_dt",
            by=group_cols,
            strategy="backward",
        )
        return joined.select(
            "flight_id",
            pl.col("cum_count").fill_null(0),
            pl.col("cum_value").fill_null(0.0),
        )

    upper = cumulative_as_of("_upper_q").rename({"cum_count": "upper_n", "cum_value": "upper_v"})
    lower = cumulative_as_of("_lower_q").rename({"cum_count": "lower_n", "cum_value": "lower_v"})

    result = (
        bounds.select("flight_id")
        .join(upper, on="flight_id", how="left")
        .join(lower, on="flight_id", how="left")
        .with_columns(
            (pl.col("upper_n") - pl.col("lower_n")).alias("_n"),
            (pl.col("upper_v") - pl.col("lower_v")).alias("_v"),
        )
        .with_columns(
            pl.when(pl.col("_n") > 0).then(pl.col("_v") / pl.col("_n")).otherwise(default).alias(out_col)
        )
        .select("flight_id", out_col)
    )
    return result


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
        (pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES).cast(pl.Float64).alias("is_delayed")
    )

    route_hour = windowed_rate(
        labeled, ["route", "hour_of_day"], "is_delayed", lookback_days, "route_hour_delay_rate", default=0.2
    )
    carrier = windowed_rate(
        labeled, ["Reporting_Airline"], "is_delayed", lookback_days, "carrier_delay_rate", default=0.2
    )

    return flights.join(route_hour, on="flight_id", how="left").join(carrier, on="flight_id", how="left")
