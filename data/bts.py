"""BTS On-Time Performance ingestion helpers.

Downloads / caches monthly On-Time Performance files for the scoped hub
airports and date range. Raw files land under data/raw/bts (gitignored).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from data.config import END_DATE, HUB_AIRPORTS, START_DATE

# Columns we keep from BTS monthly CSVs (names match public BTS exports).
BTS_COLUMNS: tuple[str, ...] = (
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "CRSArrTime",
    "ArrDelay",
    "Cancelled",
    "Diverted",
)

RAW_DIR = Path("data/raw/bts")
CACHE_DIR = Path("data/cache")


def _month_range(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def expected_month_files() -> list[Path]:
    """Return expected local paths for each month in the project date window."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return [
        RAW_DIR / f"On_Time_Reporting_Carrier_On_Time_Performance_{year}_{month:02d}.csv"
        for year, month in _month_range(
            START_DATE.year, START_DATE.month, END_DATE.year, END_DATE.month
        )
    ]


def filter_hub_flights(frame: pl.DataFrame, hubs: Iterable[str] = HUB_AIRPORTS) -> pl.DataFrame:
    """Keep flights where origin or destination is in the hub set."""
    hub_list = list(hubs)
    return frame.filter(pl.col("Origin").is_in(hub_list) | pl.col("Dest").is_in(hub_list))


def parse_crs_dep_datetime(frame: pl.DataFrame) -> pl.DataFrame:
    """Combine FlightDate + CRSDepTime into a timezone-naive scheduled datetime."""
    # CRSDepTime is HHMM as int; pad to 4 digits then parse.
    return frame.with_columns(
        pl.concat_str(
            [
                pl.col("FlightDate").cast(pl.Utf8),
                pl.lit(" "),
                pl.col("CRSDepTime")
                .cast(pl.Int32)
                .fill_null(0)
                .map_elements(lambda t: f"{int(t):04d}", return_dtype=pl.Utf8),
            ]
        )
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False)
        .alias("scheduled_dep_dt")
    )


def load_bts_month(path: Path) -> pl.DataFrame:
    """Load one monthly BTS CSV, selecting known columns when present."""
    available = pl.scan_csv(path, infer_schema_length=10_000).collect_schema().names()
    cols = [c for c in BTS_COLUMNS if c in available]
    frame = pl.read_csv(path, columns=cols, infer_schema_length=10_000)
    return filter_hub_flights(frame)


def build_flight_table(paths: list[Path] | None = None) -> pl.DataFrame:
    """Concatenate available monthly files into a single flight table.

    Missing month files are skipped so local development can proceed with a
    partial cache. Callers should treat an empty result as "run the download
    step first."
    """
    paths = paths or expected_month_files()
    frames: list[pl.DataFrame] = []
    for path in paths:
        if path.exists():
            frames.append(load_bts_month(path))
    if not frames:
        return pl.DataFrame(schema={c: pl.Utf8 for c in BTS_COLUMNS})
    combined = pl.concat(frames, how="diagonal_relaxed")
    return parse_crs_dep_datetime(combined)


def cache_flights(frame: pl.DataFrame, name: str = "flights.parquet") -> Path:
    """Write a Polars frame to the local cache directory (gitignored)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / name
    frame.write_parquet(out)
    return out
