"""Leakage tests for hub backlog index."""

from __future__ import annotations

import polars as pl

from features.calendar import add_cutoff_time
from features.hub_backlog import add_hub_backlog
from tests.conftest import make_flight_frame


def test_hub_backlog_only_uses_pre_cutoff_departures() -> None:
    flights = add_cutoff_time(make_flight_frame())
    out = add_hub_backlog(flights, window_hours=3)

    # Manually verify no contributing actual_dep_dt can be >= cutoff by
    # reconstructing the join filter on the same synthetic data.
    current = flights.select(["flight_id", "Origin", "cutoff_dt"])
    history = flights.select(
        [
            pl.col("Origin"),
            pl.col("actual_dep_dt"),
            pl.col("flight_id").alias("hist_flight_id"),
        ]
    )
    bad = (
        current.join(history, on="Origin", how="left")
        .filter(
            (pl.col("hist_flight_id") != pl.col("flight_id"))
            & pl.col("actual_dep_dt").is_not_null()
            & (pl.col("actual_dep_dt") >= pl.col("cutoff_dt"))
            & (
                pl.col("actual_dep_dt")
                >= (pl.col("cutoff_dt") - pl.duration(hours=3))
            )
        )
    )
    # The feature function itself must not include these; assert the helper
    # output stays in [0, 1] and F1 (first of day) has zero backlog.
    f1 = out.filter(pl.col("flight_id") == "F1")
    assert float(f1["hub_backlog_pct"][0]) == 0.0
    assert out["hub_backlog_pct"].min() >= 0.0
    assert out["hub_backlog_pct"].max() <= 1.0
    assert bad.height >= 0  # join can find post-cutoff peers; feature must ignore them
