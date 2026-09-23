"""Shared synthetic frames for leakage and feature unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl


def make_flight_frame() -> pl.DataFrame:
    """Small deterministic flight set spanning a few hours at one hub."""
    base = datetime(2024, 6, 1, 16, 0, 0)  # scheduled 4:00 PM → cutoff noon
    rows = [
        {
            "flight_id": "F1",
            "Tail_Number": "N100",
            "Reporting_Airline": "AA",
            "Origin": "ATL",
            "Dest": "DFW",
            "scheduled_dep_dt": base,
            "actual_dep_dt": base + timedelta(minutes=5),
            "DepDelayMinutes": 5.0,
            "Cancelled": 0,
        },
        {
            "flight_id": "F2",
            "Tail_Number": "N100",
            "Reporting_Airline": "AA",
            "Origin": "ATL",
            "Dest": "DFW",
            "scheduled_dep_dt": base + timedelta(hours=6),
            "actual_dep_dt": base + timedelta(hours=6, minutes=40),
            "DepDelayMinutes": 40.0,
        },
        {
            "flight_id": "F3",
            "Tail_Number": "N200",
            "Reporting_Airline": "DL",
            "Origin": "ATL",
            "Dest": "ORD",
            "scheduled_dep_dt": base + timedelta(hours=1),
            "actual_dep_dt": base + timedelta(hours=1, minutes=50),
            "DepDelayMinutes": 50.0,
        },
        {
            "flight_id": "F4",
            "Tail_Number": "N100",
            "Reporting_Airline": "AA",
            "Origin": "ATL",
            "Dest": "DFW",
            "scheduled_dep_dt": base + timedelta(hours=8),
            "actual_dep_dt": base + timedelta(hours=8, minutes=20),
            "DepDelayMinutes": 20.0,
        },
    ]
    return pl.DataFrame(rows)


def make_weather_frame() -> pl.DataFrame:
    """Hourly weather including a post-cutoff observation that must be ignored."""
    noon = datetime(2024, 6, 1, 12, 0, 0)
    return pl.DataFrame(
        [
            {
                "airport": "ATL",
                "obs_dt": noon - timedelta(hours=1),
                "temperature_f": 82.0,
                "wind_speed_kt": 8.0,
                "precip_in": 0.0,
                "visibility_mi": 10.0,
                "ceiling_ft": 5000.0,
            },
            {
                "airport": "ATL",
                "obs_dt": noon,
                "temperature_f": 84.0,
                "wind_speed_kt": 12.0,
                "precip_in": 0.1,
                "visibility_mi": 0.25,
                "ceiling_ft": 100.0,
            },
            {
                "airport": "ATL",
                "obs_dt": noon + timedelta(hours=3),  # after cutoff for F1 — must not leak
                "temperature_f": 64.0,
                "wind_speed_kt": 40.0,
                "precip_in": 1.0,
                "visibility_mi": 0.1,
                "ceiling_ft": 50.0,
            },
            {
                "airport": "DFW",
                "obs_dt": noon,
                "temperature_f": 90.0,
                "wind_speed_kt": 15.0,
                "precip_in": 0.0,
                "visibility_mi": 8.0,
                "ceiling_ft": 3000.0,
            },
            {
                "airport": "ORD",
                "obs_dt": noon,
                "temperature_f": 68.0,
                "wind_speed_kt": 10.0,
                "precip_in": 0.0,
                "visibility_mi": 6.0,
                "ceiling_ft": 2000.0,
            },
        ]
    )
