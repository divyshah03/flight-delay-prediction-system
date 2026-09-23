"""Leakage tests for tail-number delay propagation.

Uses a local fixture (not tests.conftest.make_flight_frame) because the
current matching logic keys on actual ARRIVAL time + Dest==Origin (see
features/tail_propagation.py docstring for why), which needs an
actual_arr_dt column the shared flight fixture doesn't carry.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from features.calendar import add_cutoff_time
from features.tail_propagation import add_tail_propagation


def make_tail_frame() -> pl.DataFrame:
    base = datetime(2024, 6, 1, 16, 0, 0)  # F1 scheduled 4pm -> cutoff noon
    rows = [
        {  # inbound leg from a non-hub spoke, lands at ATL well before F1's cutoff
            "flight_id": "F0",
            "Tail_Number": "N100",
            "Origin": "TUL",
            "Dest": "ATL",
            "scheduled_dep_dt": None,
            "actual_dep_dt": None,
            "actual_arr_dt": base - timedelta(hours=6),  # 10:00
            "DepDelayMinutes": 20.0,
            "Diverted": 0,
        },
        {  # decoy: same tail, lands before cutoff too, but at the WRONG airport
            "flight_id": "FX",
            "Tail_Number": "N100",
            "Origin": "TUL",
            "Dest": "ORD",
            "scheduled_dep_dt": None,
            "actual_dep_dt": None,
            "actual_arr_dt": base - timedelta(hours=5),  # 11:00, more recent than F0
            "DepDelayMinutes": 90.0,
            "Diverted": 0,
        },
        {
            "flight_id": "F1",
            "Tail_Number": "N100",
            "Origin": "ATL",
            "Dest": "DFW",
            "scheduled_dep_dt": base,
            "actual_dep_dt": base + timedelta(minutes=5),
            "actual_arr_dt": base + timedelta(hours=2, minutes=5),
            "DepDelayMinutes": 5.0,
            "Diverted": 0,
        },
        {  # return leg, lands back at ATL before F2's cutoff
            "flight_id": "F1B",
            "Tail_Number": "N100",
            "Origin": "DFW",
            "Dest": "ATL",
            "scheduled_dep_dt": None,
            "actual_dep_dt": None,
            "actual_arr_dt": base + timedelta(hours=5),  # 21:00
            "DepDelayMinutes": 30.0,
            "Diverted": 0,
        },
        {
            "flight_id": "F2",
            "Tail_Number": "N100",
            "Origin": "ATL",
            "Dest": "DFW",
            "scheduled_dep_dt": base + timedelta(hours=10),  # 02:00 next day
            "actual_dep_dt": base + timedelta(hours=10, minutes=15),
            "actual_arr_dt": base + timedelta(hours=12, minutes=15),
            "DepDelayMinutes": 15.0,
        },
    ]
    return pl.DataFrame(
        rows,
        schema_overrides={"scheduled_dep_dt": pl.Datetime, "actual_dep_dt": pl.Datetime},
    )


def test_tail_prior_matches_by_arrival_airport_not_just_recency() -> None:
    """F1 (departs ATL) must pick up F0 (landed at ATL) as prior, not the
    more-recent FX (landed at ORD) -- this is the fix over a tail-only join
    with no airport check."""
    flights = add_cutoff_time(make_tail_frame())
    out = add_tail_propagation(flights)
    f1 = out.filter(pl.col("flight_id") == "F1").row(0, named=True)
    assert f1["tail_prior_delay_minutes"] == 20.0
    assert f1["tail_prior_delayed"] == 1


def test_tail_picks_most_recent_eligible_arrival() -> None:
    """F2 (departs ATL later) must pick up F1B (landed at ATL at 21:00),
    not the older F0 (landed at ATL at 10:00)."""
    flights = add_cutoff_time(make_tail_frame())
    out = add_tail_propagation(flights)
    f2 = out.filter(pl.col("flight_id") == "F2").row(0, named=True)
    assert f2["tail_prior_delay_minutes"] == 30.0
    assert f2["tail_prior_delayed"] == 1


def test_tail_no_prior_when_nothing_qualifies() -> None:
    """F0 itself has no valid prior (nothing lands at TUL beforehand)."""
    flights = add_cutoff_time(make_tail_frame())
    out = add_tail_propagation(flights)
    f0 = out.filter(pl.col("flight_id") == "F0").row(0, named=True)
    assert f0["tail_prior_delay_minutes"] == 0.0
    assert f0["tail_prior_delayed"] == 0


def test_tail_ignores_flights_after_cutoff() -> None:
    """F1's prior must not be influenced by F1B/F2, which happen afterward."""
    flights = add_cutoff_time(make_tail_frame())
    out = add_tail_propagation(flights)
    f1 = out.filter(pl.col("flight_id") == "F1").row(0, named=True)
    assert f1["tail_prior_delay_minutes"] == 20.0  # not F1B's 30 or F2's 15
