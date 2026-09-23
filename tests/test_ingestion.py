"""Sanity tests for step-1 ingestion: URL building, hub scoping, and the
BTS-vs-NOAA timezone distinction (DST-aware local time vs. fixed local
standard time). Full leakage tests belong to the feature pipeline (step 2);
these just guard the ingestion layer's own correctness.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl
import pytest

from data.config import month_range
from data.ingest_bts import _add_utc_timestamps, bts_zip_url
from data.ingest_noaa import _parse_ceiling_ft, _strip_flag_suffix, station_year_url


def test_month_range_inclusive_across_year_boundary():
    months = month_range((2023, 11), (2024, 2))
    assert months == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]


def test_month_range_single_month():
    assert month_range((2024, 6), (2024, 6)) == [(2024, 6)]


def test_bts_zip_url_matches_known_live_pattern():
    # Verified against the real PREZIP directory listing on 2026-09-22.
    url = bts_zip_url(2024, 1)
    assert url == (
        "https://transtats.bts.gov/PREZIP/"
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip"
    )


def test_noaa_station_year_url():
    url = station_year_url("72219013874", 2024)
    assert url == "https://www.ncei.noaa.gov/data/local-climatological-data/access/2024/72219013874.csv"


@pytest.mark.parametrize(
    "sky, expected",
    [
        (None, None),
        ("", None),
        ("CLR:00", None),
        ("FEW:02 170 SCT:04 200", None),  # no BKN/OVC/VV -> unlimited ceiling
        ("SCT:04 65 BKN:07 180 OVC:08 250", 18000),  # lowest of BKN/OVC layers
        ("BKN:07 11 BKN:07 23 OVC:08 35", 1100),
        ("VV:09 002", 200),  # obscured sky (fog) counts as a ceiling
    ],
)
def test_parse_ceiling_ft(sky, expected):
    assert _parse_ceiling_ft(sky) == expected


def test_strip_flag_suffix_handles_sensor_and_variable_flags():
    df = pl.DataFrame({"v": ["0.50s", "2.50V", "10.00", "", "T"]})
    out = df.select(_strip_flag_suffix(pl.col("v")).alias("v"))["v"].to_list()
    assert out == [0.50, 2.50, 10.00, None, 0.0]


def test_bts_and_noaa_disagree_on_summer_utc_offset_for_atlanta():
    """This is the exact trap the ingestion code exists to avoid: BTS local
    times observe DST, NOAA LCD local times do not. For the same nominal
    summer clock reading at ATL, the two sources are 1 hour apart in real
    (UTC) time -- naively treating both as "local time" and comparing them
    directly would misalign a weather-to-flight join by an hour for roughly
    eight months of the year.
    """
    from data.airports import AIRPORTS

    atl = AIRPORTS["ATL"]
    # BTS: DST-aware conversion via IANA zone (America/New_York -> EDT in July).
    naive = pl.DataFrame({"t": ["2024-07-15 12:00:00"]}).select(
        pl.col("t").str.to_datetime()
    )["t"][0]
    bts_utc = (
        pl.Series([naive])
        .dt.replace_time_zone(atl.iana_tz)
        .dt.convert_time_zone("UTC")[0]
    )
    # NOAA: fixed standard offset, no DST, even in July.
    noaa_utc = naive - timedelta(hours=atl.std_utc_offset_hours)

    assert bts_utc.replace(tzinfo=None) != noaa_utc
    assert abs((bts_utc.replace(tzinfo=None) - noaa_utc).total_seconds()) == 3600
