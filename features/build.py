"""Orchestrates the full point-in-time feature pipeline: loads ingested BTS +
NOAA parquet, wires every feature function together in the correct order,
and writes the final modeling table.

Column contract with the feature functions in this package: `scheduled_dep_dt`
/ `actual_dep_dt` / `actual_arr_dt` / `cutoff_dt` are all UTC (this is what
makes cross-timezone joins -- weather-at-cutoff, tail propagation, hub
backlog, target encoding -- correct; see data/airports.py and
data/ingest_bts.py docstrings for why naive local time is NOT safe here).
`scheduled_dep_local` is the separate naive local wall-clock time used only
for calendar/cyclical features, where local rhythm (not the UTC clock) is the
meaningful signal.

Feature functions themselves run over the full ingestion universe (Origin OR
Dest in a hub airport) because tail propagation and hub backlog need that
broader history; the final modeling population (what gets written out and
what the models train on) is filtered down to Origin-in-hub afterward, per
the Data section's framing of predicting delays for flights departing a hub.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from data.config import DELAY_THRESHOLD_MINUTES, HUB_AIRPORTS, PROCESSED_DIR
from features.calendar import add_calendar_features, add_cutoff_time, add_is_holiday
from features.hub_backlog import add_hub_backlog
from features.tail_propagation import add_tail_propagation
from features.target_encoding import add_moving_target_encoding
from features.weather import add_ifr_flag, add_weather_at_cutoff
from models.baseline import add_route_hour_avg_delay_minutes

BTS_PARQUET = PROCESSED_DIR / "bts_ontime.parquet"
NOAA_PARQUET = PROCESSED_DIR / "noaa_weather.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"


def _load_flights(bts_path: Path) -> pl.DataFrame:
    flights = pl.read_parquet(bts_path)
    flights = flights.with_row_index("flight_id").with_columns(
        pl.col("flight_id").cast(pl.Utf8)
    )
    return flights.rename(
        {
            "sched_dep_utc": "scheduled_dep_dt",
            "actual_dep_utc": "actual_dep_dt",
            "sched_arr_utc": "scheduled_arr_dt",
            "actual_arr_utc": "actual_arr_dt",
        }
    )


def _load_weather(noaa_path: Path) -> pl.DataFrame:
    weather = pl.read_parquet(noaa_path)
    return weather.rename({"obs_time_utc": "obs_dt"}).select(
        "airport", "obs_dt", "temperature_f", "wind_speed_kt", "precip_in", "visibility_mi", "ceiling_ft"
    )


def build_features(
    bts_path: Path = BTS_PARQUET,
    noaa_path: Path = NOAA_PARQUET,
    hub_airports: list[str] = HUB_AIRPORTS,
    out_path: Path = FEATURES_PARQUET,
) -> Path:
    if not bts_path.exists():
        raise FileNotFoundError(f"{bts_path} not found -- run data.run_ingestion first")
    if not noaa_path.exists():
        raise FileNotFoundError(f"{noaa_path} not found -- run data.run_ingestion first")

    flights = _load_flights(bts_path)
    weather = _load_weather(noaa_path)

    flights = add_cutoff_time(flights)
    flights = add_calendar_features(flights, local_dt_col="scheduled_dep_local")
    flights = add_is_holiday(flights, local_dt_col="scheduled_dep_local")
    flights = add_weather_at_cutoff(flights, weather)
    flights = add_ifr_flag(flights)
    flights = add_tail_propagation(flights)
    flights = add_hub_backlog(flights)
    flights = add_moving_target_encoding(flights)
    flights = add_route_hour_avg_delay_minutes(flights)

    # Modeling population: flights departing a hub, actually flown (a
    # cancelled or diverted flight has no well-defined departure-delay
    # outcome to predict).
    modeling = flights.filter(
        pl.col("Origin").is_in(hub_airports)
        & (pl.col("Cancelled") == 0)
        & (pl.col("Diverted") == 0)
        & pl.col("DepDelayMinutes").is_not_null()
    ).with_columns(
        (pl.col("DepDelayMinutes") >= DELAY_THRESHOLD_MINUTES).cast(pl.Int8).alias("is_delayed")
    )

    modeling.write_parquet(out_path)
    print(f"[features] wrote {modeling.height:,} rows ({flights.height:,} in the full ingestion universe) to {out_path}")
    return out_path


if __name__ == "__main__":
    build_features()
