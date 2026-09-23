"""Tail-number delay propagation feature -- the headline feature in this
project.

For a flight departing hub H, the aircraft's "most recent completed flight"
is, by definition, whichever prior flight (same Tail_Number) most recently
LANDED at H -- an aircraft can only begin its next leg from wherever it last
arrived. The candidate is identified by matching `prior.Dest == current.Origin`
and gating on the PRIOR flight's actual ARRIVAL time (not departure time):

- Arrival guarantees the leg has fully landed, so its DepDelayMinutes is a
  known, final quantity -- not still in progress.
- Critically, for real ingested data, arrival-side UTC timestamps are always
  available for these candidates (Dest is always one of the scoped hub
  airports here), whereas departure-side UTC is only available when the
  candidate's own Origin also happens to be a hub. Gating on departure time
  instead would silently drop every inbound leg from a non-hub spoke airport
  -- which is most real-world arrivals at a hub-and-spoke hub. Requiring the
  Dest match also fixes a real bug in an earlier version of this function,
  which joined on Tail_Number alone with no airport check at all, and could
  therefore credit a flight with a "prior delay" from an aircraft that most
  recently landed somewhere else entirely.

Implementation note: this is a `join_asof` keyed on (Tail_Number, airport)
with the airport role flipped between sides (history's Dest vs. current's
Origin) via `by_left`/`by_right`, rather than a join on Tail_Number alone
followed by a filter. A Tail_Number-only self-join is O(k^2) per aircraft --
over a 2-year window a single aircraft can have several hundred flights, and
summed across thousands of aircraft that's billions of intermediate rows and
does not finish. The as-of join is O(n log n) and, as a bonus, can never
self-match (a flight's own Origin/Dest are never equal), so no explicit
"exclude self" filter is needed either.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from data.config import DELAY_THRESHOLD_MINUTES

# See features/hub_backlog.py's module docstring for why this epsilon exists:
# join_asof's "backward" strategy is inclusive of an exact match, so we
# subtract a microsecond to get strict "before cutoff" semantics.
_EPSILON = timedelta(microseconds=1)


def add_tail_propagation(flights: pl.DataFrame) -> pl.DataFrame:
    """Add prior-tail delay flag and minutes for each flight.

    Cutoff rule: only a prior leg with actual_arr_dt < this flight's
    cutoff_dt, and whose Dest matches this flight's Origin, may contribute.
    """
    required = {
        "flight_id",
        "Tail_Number",
        "Origin",
        "Dest",
        "cutoff_dt",
        "actual_arr_dt",
        "DepDelayMinutes",
        "Diverted",
    }
    missing = required - set(flights.columns)
    if missing:
        raise ValueError(f"flights missing columns for tail propagation: {sorted(missing)}")

    # Diverted=1 rows are excluded from history: BTS keeps `Dest` as the
    # originally-scheduled destination even when the aircraft actually landed
    # at a different (diversion) airport, which this dataset doesn't capture.
    # Matching on Dest for such a row would seat the aircraft at an airport it
    # never reached, corrupting the next leg's prior-flight lookup.
    history = (
        flights.filter(pl.col("actual_arr_dt").is_not_null() & (pl.col("Diverted") == 0))
        .select(
            pl.col("Tail_Number"),
            pl.col("Dest").alias("_airport"),
            pl.col("actual_arr_dt"),
            pl.col("DepDelayMinutes").alias("prior_delay_minutes"),
        )
        .sort(["Tail_Number", "_airport", "actual_arr_dt"])
    )

    current = (
        flights.select(
            "flight_id",
            "Tail_Number",
            pl.col("Origin").alias("_airport"),
            (pl.col("cutoff_dt") - _EPSILON).alias("_query_dt"),
        )
        .sort(["Tail_Number", "_airport", "_query_dt"])
    )

    joined = current.join_asof(
        history,
        left_on="_query_dt",
        right_on="actual_arr_dt",
        by=["Tail_Number", "_airport"],
        strategy="backward",
    ).select("flight_id", "prior_delay_minutes")

    return (
        flights.join(joined, on="flight_id", how="left")
        .with_columns(
            pl.col("prior_delay_minutes").fill_null(0.0).alias("tail_prior_delay_minutes"),
            (pl.col("prior_delay_minutes").fill_null(0.0) >= DELAY_THRESHOLD_MINUTES)
            .cast(pl.Int8)
            .alias("tail_prior_delayed"),
        )
        .drop("prior_delay_minutes")
    )
