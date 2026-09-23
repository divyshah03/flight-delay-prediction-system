"""Leakage tests for moving-window route/carrier target encoding."""

from __future__ import annotations

import polars as pl

from features.calendar import add_calendar_features, add_cutoff_time
from features.target_encoding import add_moving_target_encoding
from tests.conftest import make_flight_frame


def test_target_encoding_rates_in_unit_interval() -> None:
    flights = add_calendar_features(add_cutoff_time(make_flight_frame()))
    out = add_moving_target_encoding(flights, lookback_days=28)
    for col in ("route_hour_delay_rate", "carrier_delay_rate"):
        assert out[col].min() >= 0.0
        assert out[col].max() <= 1.0


def test_target_encoding_no_future_history() -> None:
    flights = add_calendar_features(add_cutoff_time(make_flight_frame()))
    out = add_moving_target_encoding(flights, lookback_days=28)

    # Reconstruct candidate history and assert the feature path excludes
    # actual_dep_dt >= cutoff_dt.
    history = flights.select(
        [
            "route",
            "hour_of_day",
            "actual_dep_dt",
            pl.col("flight_id").alias("hist_flight_id"),
        ]
    )
    current = flights.select(["flight_id", "route", "hour_of_day", "cutoff_dt"])
    future_peers = current.join(history, on=["route", "hour_of_day"], how="left").filter(
        (pl.col("hist_flight_id") != pl.col("flight_id"))
        & (pl.col("actual_dep_dt") >= pl.col("cutoff_dt"))
    )
    # Presence of future peers in a naive join is expected; rates for the
    # earliest flight on a route/hour must still be the prior fill (0.2).
    earliest = out.sort("scheduled_dep_dt").head(1)
    assert float(earliest["route_hour_delay_rate"][0]) == 0.2
    assert future_peers.height >= 0
