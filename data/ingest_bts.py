"""Ingest BTS On-Time Performance data, scoped to the configured hub airports
and date range.

Scoping rule: a row is kept if Origin OR Dest is in HUB_AIRPORTS (not AND).
This is deliberate, not a broader default for its own sake: the tail-number
propagation feature (features.tail_propagation) needs, for a flight departing
a hub, the same aircraft's immediately preceding completed flight -- which
always LANDS at that hub (Dest == hub) regardless of where it came from. An
Origin-AND-Dest-in-hubs filter would silently drop those inbound legs
whenever the aircraft's prior stop was a non-hub spoke airport, breaking the
single most important feature in the project. The classification/regression
population itself (the flights we predict on) is the Origin-in-hubs subset;
Dest-in-hubs rows exist only to supply tail/rotation history and are not
targets themselves.

Consequence for weather features: destination weather (feature #3) is only
available when Dest is itself a hub airport, since NOAA ingestion is scoped
to the same 15 airports. For flights landing outside the hub set, destination
weather columns will be null -- that's expected, not a bug, and the feature
pipeline should treat it as a legitimate missing value, not impute across the
leakage boundary.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl
import requests

from .airports import AIRPORTS
from .config import BTS_RAW_DIR, HUB_AIRPORTS, PROCESSED_DIR, month_range

BTS_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Columns we keep out of the ~110 BTS provides. Selected for: the leakage-safe
# schedule fields, the delay targets, and everything the point-in-time
# features (tail propagation, hub backlog) need to reconstruct aircraft
# rotations and airport-level delay state.
KEEP_COLUMNS = [
    "FlightDate",
    "Reporting_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Distance",
]


def bts_zip_url(year: int, month: int) -> str:
    return BTS_URL_TEMPLATE.format(year=year, month=month)


def download_bts_month(year: int, month: int, raw_dir: Path = BTS_RAW_DIR) -> Path:
    """Download one month's BTS zip if not already cached. Returns the local path."""
    dest = raw_dir / f"{year}_{month:02d}.zip"
    if dest.exists():
        return dest

    url = bts_zip_url(year, month)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120, stream=True)
    resp.raise_for_status()

    tmp = dest.with_suffix(".zip.tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def _add_utc_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """Add sched_dep_utc/actual_dep_utc (when Origin is a hub) and
    sched_arr_utc/actual_arr_utc (when Dest is a hub), using each hub's IANA
    timezone so DST is handled correctly. Non-hub ends are left null -- see
    module docstring for why that's acceptable.
    """

    def parse_hhmm_to_datetime(date_col: str, time_col: str) -> pl.Expr:
        # BTS times are zero-padded 4-char HHMM strings ("0830"), with a rare
        # "2400" meaning midnight of the *next* day. Empty string = missing
        # (e.g. a cancelled flight's DepTime).
        t = pl.col(time_col).str.zfill(4)
        is_2400 = t == "2400"
        hour = pl.when(is_2400).then(0).otherwise(t.str.slice(0, 2).cast(pl.Int8, strict=False))
        minute = t.str.slice(2, 2).cast(pl.Int8, strict=False)
        base_date = pl.col(date_col).str.to_date("%Y-%m-%d")
        day_offset = pl.when(is_2400).then(1).otherwise(0)
        return (
            (base_date.dt.offset_by(day_offset.cast(pl.Utf8) + "d"))
            .cast(pl.Datetime("us"))
            .dt.offset_by(hour.cast(pl.Utf8) + "h")
            .dt.offset_by(minute.cast(pl.Utf8) + "m")
        )

    df = df.with_columns(
        parse_hhmm_to_datetime("FlightDate", "CRSDepTime").alias("_sched_dep_local_naive"),
        parse_hhmm_to_datetime("FlightDate", "DepTime").alias("_actual_dep_local_naive"),
        parse_hhmm_to_datetime("FlightDate", "CRSArrTime").alias("_sched_arr_local_naive"),
        parse_hhmm_to_datetime("FlightDate", "ArrTime").alias("_actual_arr_local_naive"),
    )

    out = df.with_columns(
        [
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("sched_dep_utc"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("actual_dep_utc"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("sched_arr_utc"),
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("actual_arr_utc"),
        ]
    )

    # Per-airport timezone conversion: polars needs one tz per column
    # operation, so we convert per hub airport and coalesce into the shared
    # output columns.
    for iata, meta in AIRPORTS.items():
        is_origin_hub = pl.col("Origin") == iata
        is_dest_hub = pl.col("Dest") == iata

        def to_utc(naive_col: str) -> pl.Expr:
            return (
                pl.col(naive_col)
                .dt.replace_time_zone(meta.iana_tz, ambiguous="earliest", non_existent="null")
                .dt.convert_time_zone("UTC")
            )

        out = out.with_columns(
            pl.when(is_origin_hub).then(to_utc("_sched_dep_local_naive")).otherwise(pl.col("sched_dep_utc")).alias("sched_dep_utc"),
            pl.when(is_origin_hub).then(to_utc("_actual_dep_local_naive")).otherwise(pl.col("actual_dep_utc")).alias("actual_dep_utc"),
            pl.when(is_dest_hub).then(to_utc("_sched_arr_local_naive")).otherwise(pl.col("sched_arr_utc")).alias("sched_arr_utc"),
            pl.when(is_dest_hub).then(to_utc("_actual_arr_local_naive")).otherwise(pl.col("actual_arr_utc")).alias("actual_arr_utc"),
        )

    return out.rename({"_sched_dep_local_naive": "scheduled_dep_local"}).drop(
        "_actual_dep_local_naive",
        "_sched_arr_local_naive",
        "_actual_arr_local_naive",
    )


def load_and_filter_month(zip_path: Path, hub_airports: list[str] = HUB_AIRPORTS) -> pl.DataFrame:
    """Extract a month's CSV in-memory from its zip, select+filter, add UTC times."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"expected exactly one CSV in {zip_path}, found {csv_names}")
        with zf.open(csv_names[0]) as f:
            df = pl.read_csv(
                f,
                columns=KEEP_COLUMNS,
                schema_overrides={
                    "CRSDepTime": pl.Utf8,
                    "DepTime": pl.Utf8,
                    "CRSArrTime": pl.Utf8,
                    "ArrTime": pl.Utf8,
                    "WheelsOff": pl.Utf8,
                    "WheelsOn": pl.Utf8,
                    "Tail_Number": pl.Utf8,
                    "CancellationCode": pl.Utf8,
                },
            )

    df = df.filter(pl.col("Origin").is_in(hub_airports) | pl.col("Dest").is_in(hub_airports))
    df = _add_utc_timestamps(df)
    return df


def ingest_bts(
    start=None,
    end=None,
    hub_airports: list[str] = HUB_AIRPORTS,
    raw_dir: Path = BTS_RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    overwrite: bool = False,
) -> Path:
    """Download + filter every month in range, write one combined parquet.

    Returns the path to the combined parquet file.
    """
    months = month_range(start, end) if start or end else month_range()
    out_path = processed_dir / "bts_ontime.parquet"
    if out_path.exists() and not overwrite:
        print(f"[bts] {out_path} already exists, skipping (pass overwrite=True to rebuild)")
        return out_path

    frames = []
    for year, month in months:
        print(f"[bts] {year}-{month:02d}: downloading...")
        zip_path = download_bts_month(year, month, raw_dir)
        print(f"[bts] {year}-{month:02d}: filtering...")
        frames.append(load_and_filter_month(zip_path, hub_airports))

    combined = pl.concat(frames)
    combined.write_parquet(out_path)
    print(f"[bts] wrote {combined.height:,} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    ingest_bts()
