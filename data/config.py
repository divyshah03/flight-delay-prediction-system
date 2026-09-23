"""Central configuration for data ingestion: scope (airports, date range) and paths.

Canonical HUB_AIRPORTS/date-range values here are the ones confirmed directly
with the user on 2026-09-22. A concurrent session independently scaffolded
features/models/evaluation against a different hub list (LGA/BOS instead of
MCO/IAH) and a different constant naming scheme (START_DATE/END_DATE,
CUTOFF_HOURS_BEFORE_DEPARTURE, DELAY_THRESHOLD_MINUTES). Both naming schemes
are exported below so that scaffold's imports keep working unmodified, rather
than touching six downstream feature/model files to rename things.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# --- Scope: confirmed with user 2026-09-22 ---
# Top 15 US hub airports by traffic, chosen for volume + geographic/weather
# diversity (coastal, mountain, snow-belt, desert, hot/humid).
HUB_AIRPORTS: list[str] = [
    "ATL", "DFW", "DEN", "ORD", "LAX",
    "JFK", "LAS", "MCO", "MIA", "CLT",
    "SEA", "PHX", "EWR", "SFO", "IAH",
]

START_YEAR_MONTH: tuple[int, int] = (2023, 1)
END_YEAR_MONTH: tuple[int, int] = (2024, 12)

# Aliases for the feature/model/eval scaffold's expected names (see module
# docstring). Keep in sync with START_YEAR_MONTH/END_YEAR_MONTH above.
START_DATE: date = date(START_YEAR_MONTH[0], START_YEAR_MONTH[1], 1)
END_DATE: date = date(END_YEAR_MONTH[0], END_YEAR_MONTH[1], 28)  # exact day unused by callers
CUTOFF_HOURS_BEFORE_DEPARTURE: int = 4
DELAY_THRESHOLD_MINUTES: int = 15

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
BTS_RAW_DIR = RAW_DIR / "bts"
NOAA_RAW_DIR = RAW_DIR / "noaa"
PROCESSED_DIR = DATA_DIR / "processed"

for _dir in (BTS_RAW_DIR, NOAA_RAW_DIR, PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def month_range(
    start: tuple[int, int] = START_YEAR_MONTH,
    end: tuple[int, int] = END_YEAR_MONTH,
) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) tuples from start to end."""
    start_y, start_m = start
    end_y, end_m = end
    months = []
    y, m = start_y, start_m
    while (y, m) <= (end_y, end_m):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months
