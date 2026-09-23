"""Leakage tests for tail-number delay propagation."""

from __future__ import annotations

import polars as pl

from features.calendar import add_cutoff_time
from features.tail_propagation import add_tail_propagation
from tests.conftest import make_flight_frame


def test_tail_prior_actual_dep_before_cutoff() -> None:
    flights = add_cutoff_time(make_flight_frame())
    # Keep prior_actual via intermediate check: rebuild candidates manually.
    out = add_tail_propagation(flights)

    # F2 (same tail N100) should see F1 as prior; F1 has no prior.
    f1 = out.filter(pl.col("flight_id") == "F1")
    f2 = out.filter(pl.col("flight_id") == "F2")
    assert int(f1["tail_prior_delayed"][0]) == 0
    assert float(f1["tail_prior_delay_minutes"][0]) == 0.0
    assert int(f2["tail_prior_delayed"][0]) == 0  # F1 was only 5 min late
    assert float(f2["tail_prior_delay_minutes"][0]) == 5.0


def test_tail_ignores_flights_after_cutoff() -> None:
    """A later same-tail flight must not influence an earlier flight's feature."""
    flights = add_cutoff_time(make_flight_frame())
    out = add_tail_propagation(flights)
    f1 = out.filter(pl.col("flight_id") == "F1").row(0, named=True)
    # If F2/F4 leaked backward, F1 would pick up a non-zero prior delay.
    assert f1["tail_prior_delay_minutes"] == 0.0
