"""Static metadata for the hub airports: IATA/ICAO codes, timezone, NOAA station ID.

Station IDs (USAF+WBAN, 11 digits) were looked up against NOAA's ISD station
history (https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv) by matching
ICAO code, and confirmed to have LCD coverage through 2025.

`std_utc_offset_hours` is the STANDARD (winter, non-DST) UTC offset for the
airport's zone. It exists separately from `iana_tz` because NOAA's Local
Climatological Data timestamps are in local standard time year-round (no DST
shift, verified empirically against each station's SOD/Sunrise records —
NOAA's LCD documentation does not state this clearly), while BTS flight times
are local wall-clock time and do observe DST. Use `iana_tz` (with a proper
zoneinfo-aware conversion) for BTS times, and the fixed `std_utc_offset_hours`
for NOAA LCD times. Mixing these up silently shifts weather-to-flight joins
by an hour for about eight months of the year.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AirportMeta:
    iata: str
    icao: str
    name: str
    noaa_station_id: str  # USAF+WBAN, 11 digits, matches LCD file name
    iana_tz: str  # for BTS local wall-clock times (DST-aware)
    std_utc_offset_hours: int  # for NOAA LCD times (fixed, no DST)


AIRPORTS: dict[str, AirportMeta] = {
    a.iata: a
    for a in [
        AirportMeta("ATL", "KATL", "Hartsfield-Jackson Atlanta Intl", "72219013874", "America/New_York", -5),
        AirportMeta("DFW", "KDFW", "Dallas/Fort Worth Intl", "72259003927", "America/Chicago", -6),
        AirportMeta("DEN", "KDEN", "Denver Intl", "72565003017", "America/Denver", -7),
        AirportMeta("ORD", "KORD", "Chicago O'Hare Intl", "72530094846", "America/Chicago", -6),
        AirportMeta("LAX", "KLAX", "Los Angeles Intl", "72295023174", "America/Los_Angeles", -8),
        AirportMeta("JFK", "KJFK", "John F Kennedy Intl", "74486094789", "America/New_York", -5),
        AirportMeta("LAS", "KLAS", "Harry Reid (McCarran) Intl", "72386023169", "America/Los_Angeles", -8),
        AirportMeta("MCO", "KMCO", "Orlando Intl", "72205012815", "America/New_York", -5),
        AirportMeta("MIA", "KMIA", "Miami Intl", "72202012839", "America/New_York", -5),
        AirportMeta("CLT", "KCLT", "Charlotte/Douglas Intl", "72314013881", "America/New_York", -5),
        AirportMeta("SEA", "KSEA", "Seattle-Tacoma Intl", "72793024233", "America/Los_Angeles", -8),
        AirportMeta("PHX", "KPHX", "Phoenix Sky Harbor Intl", "72278023183", "America/Phoenix", -7),
        AirportMeta("EWR", "KEWR", "Newark Liberty Intl", "72502014734", "America/New_York", -5),
        AirportMeta("SFO", "KSFO", "San Francisco Intl", "72494023234", "America/Los_Angeles", -8),
        AirportMeta("IAH", "KIAH", "George Bush Intercontinental Houston", "72243012960", "America/Chicago", -6),
    ]
}

HUB_IATA_CODES = list(AIRPORTS.keys())
