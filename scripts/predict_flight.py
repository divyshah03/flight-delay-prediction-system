"""CLI convenience wrapper: look up a real flight from the already-ingested
2024 dataset by carrier/flight number/route, and run it through the trained
models -- so testing the project doesn't require hand-assembling a feature
vector.

This does NOT compute features live from raw booking info (that would need
an online feature store -- see api/main.py's scope note). It looks up a row
that the existing point-in-time pipeline already computed, in
data/processed/features.parquet, and feeds it to the same trained XGBoost
artifacts the API serves.

Usage:
    python -m scripts.predict_flight OH 5167 CLT ATL
    python -m scripts.predict_flight OH 5167 CLT ATL --date 2024-01-18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import xgboost as xgb

from evaluation.metrics import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
ARTIFACT_DIR = ROOT / "artifacts"


def find_flight(
    carrier: str, flight_number: str, origin: str, dest: str, date: str | None
) -> pl.DataFrame:
    df = pl.read_parquet(FEATURES_PATH)
    matches = df.filter(
        (pl.col("Reporting_Airline") == carrier.upper())
        & (pl.col("Flight_Number_Reporting_Airline").cast(pl.Utf8) == str(int(flight_number)))
        & (pl.col("Origin") == origin.upper())
        & (pl.col("Dest") == dest.upper())
    )
    if date:
        matches = matches.filter(pl.col("FlightDate").cast(pl.Utf8).str.starts_with(date))
    return matches.sort("FlightDate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict delay for a flight in the 2024 dataset")
    parser.add_argument("carrier", help="e.g. OH")
    parser.add_argument("flight_number", help="e.g. 5167")
    parser.add_argument("origin", help="e.g. CLT")
    parser.add_argument("dest", help="e.g. ATL")
    parser.add_argument("--date", help="YYYY-MM-DD, optional -- narrows to one specific flight")
    args = parser.parse_args()

    if not FEATURES_PATH.exists():
        sys.exit(f"No processed features found at {FEATURES_PATH}. Run the feature pipeline first.")

    matches = find_flight(args.carrier, args.flight_number, args.origin, args.dest, args.date)
    if matches.height == 0:
        sys.exit(
            f"No flight found for {args.carrier}{args.flight_number} "
            f"{args.origin}->{args.dest}"
            + (f" on {args.date}" if args.date else "")
            + " in the 2024 dataset. Check the carrier code, flight number, and airport codes."
        )

    if matches.height > 1 and not args.date:
        dates = matches["FlightDate"].to_list()
        print(
            f"Found {matches.height} matching flights on different dates; "
            f"using the most recent ({dates[-1]}). Pass --date to pick another.",
            file=sys.stderr,
        )
    row = matches[-1]

    clf = xgb.XGBClassifier()
    clf.load_model(ARTIFACT_DIR / "xgboost_classifier.json")
    reg = xgb.XGBRegressor()
    reg.load_model(ARTIFACT_DIR / "xgboost_regressor.json")

    x = row.select(FEATURE_COLUMNS).to_numpy()
    proba = float(clf.predict_proba(x)[0, 1])
    expected_minutes = max(float(reg.predict(x)[0]), 0.0)

    print(f"{args.carrier.upper()}{args.flight_number} {args.origin.upper()} -> {args.dest.upper()} "
          f"on {row['FlightDate'][0]}")
    print(f"  Predicted: {proba:.0%} chance of delay, expected ~{expected_minutes:.0f} min late if delayed")

    actual_minutes = row["DepDelayMinutes"][0]
    actual_delayed = row["is_delayed"][0]
    if actual_minutes is not None:
        outcome = "delayed" if actual_delayed else "on time"
        print(f"  Actual:    {outcome} ({actual_minutes:.0f} min departure delay)")
    else:
        print("  Actual:    unknown (cancelled or missing departure time)")


if __name__ == "__main__":
    main()
