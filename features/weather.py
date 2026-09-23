"""T-4h weather features and IFR flag.

CRITICAL: never join weather at scheduled departure time — that observation
coincides with (or causes) the delay. Always join the latest observation at or
before cutoff_dt for each airport.
"""

from __future__ import annotations

import polars as pl

IFR_VISIBILITY_MI = 0.5
IFR_CEILING_FT = 200.0


def _latest_weather_asof(
    flights: pl.DataFrame,
    weather: pl.DataFrame,
    airport_col: str,
    prefix: str,
) -> pl.DataFrame:
    """As-of join weather onto flights using cutoff_dt as the right bound.

    Cutoff rule: only weather rows with obs_dt <= cutoff_dt are eligible;
    Polars join_asof with strategy='backward' enforces that bound.
    """
    left = flights.select(
        [
            pl.col("flight_id"),
            pl.col(airport_col).alias("airport"),
            pl.col("cutoff_dt"),
        ]
    ).sort(["airport", "cutoff_dt"])

    right = weather.sort(["airport", "obs_dt"])

    joined = left.join_asof(
        right,
        left_on="cutoff_dt",
        right_on="obs_dt",
        by="airport",
        strategy="backward",
    )

    rename_map = {
        "temp_c": f"{prefix}_temp_c",
        "wind_speed_kt": f"{prefix}_wind_speed_kt",
        "precip_in": f"{prefix}_precip_in",
        "visibility_mi": f"{prefix}_visibility_mi",
        "ceiling_ft": f"{prefix}_ceiling_ft",
        "obs_dt": f"{prefix}_obs_dt",
    }
    weather_cols = [c for c in rename_map if c in joined.columns]
    slim = joined.select(["flight_id"] + weather_cols).rename(
        {c: rename_map[c] for c in weather_cols}
    )
    return flights.join(slim, on="flight_id", how="left")


def add_weather_at_cutoff(flights: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Attach origin and destination weather as of each flight's cutoff_dt.

    Cutoff rule: obs_dt <= cutoff_dt for both origin and destination joins.
    """
    out = _latest_weather_asof(flights, weather, "Origin", "origin")
    out = _latest_weather_asof(out, weather, "Dest", "dest")
    return out


def add_ifr_flag(flights: pl.DataFrame) -> pl.DataFrame:
    """Binary IFR flag from origin visibility/ceiling at T-4h.

    Cutoff rule: uses only origin_* weather columns already joined at cutoff.
    IFR = 1 when visibility < 0.5 mi OR ceiling < 200 ft.
    """
    return flights.with_columns(
        (
            (pl.col("origin_visibility_mi") < IFR_VISIBILITY_MI)
            | (pl.col("origin_ceiling_ft") < IFR_CEILING_FT)
        )
        .fill_null(False)
        .cast(pl.Int8)
        .alias("ifr_flag")
    )
