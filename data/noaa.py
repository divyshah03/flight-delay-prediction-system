"""NOAA ISD / METAR hourly weather ingestion helpers.

Weather features must use observations at or before the T-4h cutoff for each
flight. This module only loads and normalizes station-hour observations;
cutoff joining lives in features/weather.py.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from data.config import HUB_AIRPORTS

RAW_DIR = Path("data/raw/noaa")
CACHE_DIR = Path("data/cache")

# Minimal METAR-like schema we normalize into.
WEATHER_COLUMNS: tuple[str, ...] = (
    "airport",
    "obs_dt",
    "temp_c",
    "wind_speed_kt",
    "precip_in",
    "visibility_mi",
    "ceiling_ft",
)

# Rough ICAO station ids for the hub list (US METAR convention: K + IATA).
AIRPORT_TO_ICAO: dict[str, str] = {code: f"K{code}" for code in HUB_AIRPORTS}


def normalize_weather(frame: pl.DataFrame) -> pl.DataFrame:
    """Coerce a raw weather extract into the project schema."""
    required = set(WEATHER_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Weather frame missing columns: {sorted(missing)}")
    return (
        frame.select(list(WEATHER_COLUMNS))
        .with_columns(
            pl.col("obs_dt").cast(pl.Datetime),
            pl.col("temp_c").cast(pl.Float64),
            pl.col("wind_speed_kt").cast(pl.Float64),
            pl.col("precip_in").cast(pl.Float64),
            pl.col("visibility_mi").cast(pl.Float64),
            pl.col("ceiling_ft").cast(pl.Float64),
            pl.col("airport").cast(pl.Utf8).str.to_uppercase(),
        )
        .filter(pl.col("airport").is_in(list(HUB_AIRPORTS)))
        .sort(["airport", "obs_dt"])
    )


def load_weather_cache(path: Path | None = None) -> pl.DataFrame:
    """Load cached hourly weather parquet if present, else an empty schema frame."""
    path = path or (CACHE_DIR / "weather.parquet")
    if not path.exists():
        return pl.DataFrame(
            schema={
                "airport": pl.Utf8,
                "obs_dt": pl.Datetime,
                "temp_c": pl.Float64,
                "wind_speed_kt": pl.Float64,
                "precip_in": pl.Float64,
                "visibility_mi": pl.Float64,
                "ceiling_ft": pl.Float64,
            }
        )
    return normalize_weather(pl.read_parquet(path))


def cache_weather(frame: pl.DataFrame, name: str = "weather.parquet") -> Path:
    """Persist normalized weather to the local cache (gitignored)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / name
    normalize_weather(frame).write_parquet(out)
    return out
