"""Data ingestion package: BTS flights + NOAA weather, scoped to hub airports."""

from data.config import (
    CUTOFF_HOURS_BEFORE_DEPARTURE,
    DELAY_THRESHOLD_MINUTES,
    END_DATE,
    HUB_AIRPORTS,
    START_DATE,
)

__all__ = [
    "CUTOFF_HOURS_BEFORE_DEPARTURE",
    "DELAY_THRESHOLD_MINUTES",
    "END_DATE",
    "HUB_AIRPORTS",
    "START_DATE",
]
