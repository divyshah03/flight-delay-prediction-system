"""Shared data-scope constants for BTS + NOAA ingestion."""

from __future__ import annotations

from datetime import date

# Top US hubs — keep this list tight; expanding it is a scope change.
HUB_AIRPORTS: tuple[str, ...] = (
    "ATL",
    "ORD",
    "DFW",
    "DEN",
    "LAX",
    "CLT",
    "LAS",
    "PHX",
    "MIA",
    "SEA",
    "SFO",
    "EWR",
    "JFK",
    "LGA",
    "BOS",
)

# 1–2 years of history is enough for a resume-scale pipeline.
START_DATE: date = date(2023, 1, 1)
END_DATE: date = date(2024, 12, 31)

# Prediction cutoff: only information available at or before T-4h may be used.
CUTOFF_HOURS_BEFORE_DEPARTURE: int = 4

# Binary delay target used throughout classification.
DELAY_THRESHOLD_MINUTES: int = 15
