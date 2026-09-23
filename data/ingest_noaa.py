"""Ingest NOAA Local Climatological Data (LCD) hourly weather observations for
the configured hub airports and date range.

LCD is used (rather than raw ISD or ISD-Lite) because it's the NOAA product
built specifically for ASOS/AWOS stations at major airports, and already
carries the fields the Features section needs directly: temperature, wind
speed, precipitation, visibility, and sky-condition layers (from which
ceiling is derived).

Timezone note: LCD's DATE column is local standard time year-round, verified
empirically (SOD summary rows end at 23:59 on the matching calendar date, and
Sunrise/Sunset values match known local sunrise times for that station) --
NOT UTC and NOT DST-adjusted. See airports.py docstring for why this matters:
using each airport's IANA zone (which applies DST) here would misalign
weather-to-flight joins by an hour for roughly eight months of the year.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import polars as pl
import requests

from .airports import AIRPORTS
from .config import HUB_AIRPORTS, NOAA_RAW_DIR, PROCESSED_DIR, month_range

LCD_URL_TEMPLATE = "https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}/{station_id}.csv"

# Routine (FM-15) and special (FM-16) METAR reports are the actual hourly/
# sub-hourly observations. Other REPORT_TYPE values in the same file are
# daily/monthly summary rows (SOD, SOM, ...) with the Hourly* columns empty.
HOURLY_REPORT_TYPES = ["FM-15", "FM-16"]

_SKY_LAYER_RE = re.compile(r"([A-Z]{2,3}):\d+\s*(\d+)?")
# Cover codes that constitute a ceiling per aviation convention: broken/
# overcast cloud, or indefinite ceiling from vertical visibility (obscuration,
# e.g. dense fog). FEW/SCT/CLR do not create a ceiling.
_CEILING_COVER_CODES = {"BKN", "OVC", "VV"}


def _parse_ceiling_ft(sky_conditions: str | None) -> int | None:
    """Lowest BKN/OVC/VV layer height in feet, or None if no ceiling (clear,
    or only FEW/SCT layers -- i.e. unlimited ceiling)."""
    if not sky_conditions:
        return None
    heights = []
    for cover, height_hundreds_ft in _SKY_LAYER_RE.findall(sky_conditions):
        if cover in _CEILING_COVER_CODES and height_hundreds_ft:
            heights.append(int(height_hundreds_ft) * 100)
    return min(heights) if heights else None


def _strip_flag_suffix(expr: pl.Expr) -> pl.Expr:
    """LCD numeric fields sometimes carry a trailing flag letter (e.g. '0.50s'
    for a sensor-derived reading, '2.50V' for variable visibility) or the
    literal 'T' for trace precipitation. Extract the leading numeric part;
    trace becomes 0.0."""
    cleaned = expr.str.replace("^T$", "0.0")
    return cleaned.str.extract(r"^(\d+\.?\d*)", 1).cast(pl.Float64, strict=False)


def station_year_url(station_id: str, year: int) -> str:
    return LCD_URL_TEMPLATE.format(station_id=station_id, year=year)


def download_station_year(station_id: str, year: int, raw_dir: Path = NOAA_RAW_DIR) -> Path:
    dest = raw_dir / f"{station_id}_{year}.csv"
    if dest.exists():
        return dest

    url = station_year_url(station_id, year)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120, stream=True)
    resp.raise_for_status()

    tmp = dest.with_suffix(".csv.tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def parse_lcd_csv(path: Path, iata: str) -> pl.DataFrame:
    """Load one station-year LCD file, keep hourly obs, derive clean columns."""
    meta = AIRPORTS[iata]

    df = pl.read_csv(
        path,
        columns=[
            "DATE",
            "REPORT_TYPE",
            "HourlyDryBulbTemperature",
            "HourlyWindSpeed",
            "HourlyPrecipitation",
            "HourlyVisibility",
            "HourlySkyConditions",
        ],
        schema_overrides={
            "HourlyDryBulbTemperature": pl.Utf8,
            "HourlyWindSpeed": pl.Utf8,
            "HourlyPrecipitation": pl.Utf8,
            "HourlyVisibility": pl.Utf8,
        },
    )

    df = df.filter(pl.col("REPORT_TYPE").str.strip_chars().is_in(HOURLY_REPORT_TYPES))

    df = df.with_columns(
        pl.col("DATE").str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("obs_time_lst"),
        _strip_flag_suffix(pl.col("HourlyDryBulbTemperature")).alias("temperature_f"),
        _strip_flag_suffix(pl.col("HourlyWindSpeed")).alias("wind_speed_kt"),
        _strip_flag_suffix(pl.col("HourlyPrecipitation")).alias("precip_in"),
        _strip_flag_suffix(pl.col("HourlyVisibility")).alias("visibility_mi"),
        pl.col("HourlySkyConditions")
        .map_elements(_parse_ceiling_ft, return_dtype=pl.Int64)
        .alias("ceiling_ft"),
    )

    df = df.with_columns(
        (pl.col("obs_time_lst") - timedelta(hours=meta.std_utc_offset_hours))
        .dt.replace_time_zone("UTC")
        .alias("obs_time_utc"),
        pl.lit(iata).alias("airport"),
    )

    return df.select(
        "airport",
        "obs_time_utc",
        "obs_time_lst",
        "temperature_f",
        "wind_speed_kt",
        "precip_in",
        "visibility_mi",
        "ceiling_ft",
    )


def ingest_noaa(
    start=None,
    end=None,
    hub_airports: list[str] = HUB_AIRPORTS,
    raw_dir: Path = NOAA_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    overwrite: bool = False,
) -> Path:
    """Download + parse LCD data for every hub airport across every year
    touched by the date range, write one combined parquet."""
    months = month_range(start, end) if start or end else month_range()
    years = sorted({y for y, _ in months})
    out_path = processed_dir / "noaa_weather.parquet"
    if out_path.exists() and not overwrite:
        print(f"[noaa] {out_path} already exists, skipping (pass overwrite=True to rebuild)")
        return out_path

    frames = []
    for iata in hub_airports:
        if iata not in AIRPORTS:
            raise ValueError(f"no NOAA station metadata for airport {iata!r}; add it to airports.py")
        station_id = AIRPORTS[iata].noaa_station_id
        for year in years:
            print(f"[noaa] {iata} {year}: downloading...")
            path = download_station_year(station_id, year, raw_dir)
            print(f"[noaa] {iata} {year}: parsing...")
            frames.append(parse_lcd_csv(path, iata))

    combined = pl.concat(frames).sort("airport", "obs_time_utc")
    combined.write_parquet(out_path)
    print(f"[noaa] wrote {combined.height:,} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    ingest_noaa()
