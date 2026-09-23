"""Calendar / schedule features and cyclical time encodings."""

from __future__ import annotations

import math
from datetime import date

import polars as pl

# Major US federal holidays used for a simple binary flag (year-agnostic month/day).
_US_HOLIDAY_MD: set[tuple[int, int]] = {
    (1, 1),  # New Year's Day
    (7, 4),  # Independence Day
    (11, 11),  # Veterans Day
    (12, 25),  # Christmas
}


def add_cutoff_time(flights: pl.DataFrame, hours_before: int = 4) -> pl.DataFrame:
    """Attach cutoff_dt = scheduled_dep_dt - hours_before.

    Cutoff rule: every historical/aggregate feature may only use records with
    actual_time < cutoff_dt.
    """
    return flights.with_columns(
        (pl.col("scheduled_dep_dt") - pl.duration(hours=hours_before)).alias("cutoff_dt")
    )


def add_calendar_features(flights: pl.DataFrame, local_dt_col: str = "scheduled_dep_dt") -> pl.DataFrame:
    """Add route, hour, day-of-week, and cyclical sin/cos encodings.

    `local_dt_col` must be LOCAL wall-clock scheduled departure time, not a
    UTC-converted timestamp -- cyclical hour/day-of-year patterns are a local
    phenomenon (rush hours, daylight), and using the UTC clock hour would
    scramble that signal differently for every airport's timezone. The
    default assumes a single-timezone frame (true for the synthetic test
    fixtures); the real pipeline (features/build.py) passes the explicit
    local column since `scheduled_dep_dt` there holds UTC.

    Cutoff rule: these are known from the published schedule, so they are
    available at booking time and do not depend on post-cutoff observations.
    """
    return flights.with_columns(
        (pl.col("Origin") + pl.lit("-") + pl.col("Dest")).alias("route"),
        pl.col(local_dt_col).dt.hour().alias("hour_of_day"),
        pl.col(local_dt_col).dt.weekday().alias("day_of_week"),
        pl.col(local_dt_col).dt.ordinal_day().alias("day_of_year"),
    ).with_columns(
        (2 * math.pi * pl.col("hour_of_day") / 24).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("hour_of_day") / 24).cos().alias("hour_cos"),
        (2 * math.pi * pl.col("day_of_year") / 366).sin().alias("doy_sin"),
        (2 * math.pi * pl.col("day_of_year") / 366).cos().alias("doy_cos"),
    )


def is_us_holiday(d: date) -> bool:
    """Return True for a small fixed set of major US holidays (month/day match)."""
    return (d.month, d.day) in _US_HOLIDAY_MD


def add_is_holiday(flights: pl.DataFrame, local_dt_col: str = "scheduled_dep_dt") -> pl.DataFrame:
    """Binary is_holiday from scheduled departure date.

    `local_dt_col` should be local wall-clock date, same reasoning as
    add_calendar_features -- a UTC date can fall on the wrong side of
    midnight relative to the holiday as observed locally.

    Cutoff rule: derived only from the scheduled calendar date.
    """
    return flights.with_columns(
        pl.col(local_dt_col)
        .dt.date()
        .map_elements(is_us_holiday, return_dtype=pl.Boolean)
        .cast(pl.Int8)
        .alias("is_holiday")
    )
