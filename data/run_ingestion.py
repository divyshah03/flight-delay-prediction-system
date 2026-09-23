"""CLI entrypoint: run BTS + NOAA ingestion for the configured scope.

Usage:
    python -m data.run_ingestion [--overwrite]
"""

from __future__ import annotations

import argparse

from .config import HUB_AIRPORTS, START_YEAR_MONTH, END_YEAR_MONTH
from .ingest_bts import ingest_bts
from .ingest_noaa import ingest_noaa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="rebuild parquet even if cached")
    args = parser.parse_args()

    print(f"Hub airports: {HUB_AIRPORTS}")
    print(f"Date range: {START_YEAR_MONTH} to {END_YEAR_MONTH}")

    bts_path = ingest_bts(overwrite=args.overwrite)
    noaa_path = ingest_noaa(overwrite=args.overwrite)

    print(f"\nDone.\n  BTS  -> {bts_path}\n  NOAA -> {noaa_path}")


if __name__ == "__main__":
    main()
