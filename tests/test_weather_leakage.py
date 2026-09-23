"""Leakage tests for weather T-4h joins and IFR flag."""

from __future__ import annotations

import polars as pl

from features.calendar import add_cutoff_time
from features.weather import add_ifr_flag, add_weather_at_cutoff
from tests.conftest import make_flight_frame, make_weather_frame


def _obs_after_cutoff(obs_col: str, cutoff_col: str) -> pl.Expr:
    return pl.col(obs_col).is_not_null() & (pl.col(obs_col) > pl.col(cutoff_col))


def test_weather_obs_never_after_cutoff() -> None:
    flights = add_cutoff_time(make_flight_frame())
    weather = make_weather_frame()
    out = add_weather_at_cutoff(flights, weather)

    assert out.filter(_obs_after_cutoff("origin_obs_dt", "cutoff_dt")).height == 0
    assert out.filter(_obs_after_cutoff("dest_obs_dt", "cutoff_dt")).height == 0


def test_weather_uses_latest_obs_at_or_before_cutoff() -> None:
    flights = add_cutoff_time(make_flight_frame()).filter(pl.col("flight_id") == "F1")
    weather = make_weather_frame()
    out = add_weather_at_cutoff(flights, weather)
    # F1 cutoff is noon; noon obs (vis 0.25) is allowed, post-cutoff storm is not.
    assert out["origin_visibility_mi"][0] == 0.25
    assert out["origin_temperature_f"][0] == 84.0


def test_ifr_flag_from_cutoff_weather() -> None:
    flights = add_cutoff_time(make_flight_frame()).filter(pl.col("flight_id") == "F1")
    out = add_ifr_flag(add_weather_at_cutoff(flights, make_weather_frame()))
    assert int(out["ifr_flag"][0]) == 1
