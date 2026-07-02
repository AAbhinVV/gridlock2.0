"""
Dispatch engine: route each detected event to the nearest police station with
a complete, ML-backed action order.

Flow (used by the Live Operations dashboard on every clock tick):
    live feed detects a new event
      -> Forecaster predicts duration / closure / major-event probability
      -> recommend() sizes manpower, barricades and the diversion plan
      -> nearest police station is looked up by haversine distance
      -> a DispatchOrder is issued to that station

The station directory is learned from the data itself: each station's location
is the median coordinate of the events it historically handled (8k+ records),
so no external gazetteer is needed.

Usage (smoke test):
    python -m src.dispatch
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from src.data_prep import EventInput
from src.live_feed import to_ist

EARTH_RADIUS_KM = 6371.0
MIN_STATION_EVENTS = 5  # ignore stations seen too rarely to trust their location


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance; lat2/lon2 may be numpy arrays."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


class PoliceStationDirectory:
    """Station name -> representative location + jurisdiction stats, built
    from the historical event records."""

    def __init__(self, df: pd.DataFrame, min_events: int = MIN_STATION_EVENTS):
        known = df[df["police_station"] != "Unknown"]
        g = known.groupby("police_station").agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            events_handled=("police_station", "size"),
            zone=("zone", lambda s: s.mode().iat[0] if len(s.mode()) else "Unknown"),
        )
        self.stations = g[g["events_handled"] >= min_events].reset_index()

    def nearest(self, lat: float, lon: float) -> dict:
        d = haversine_km(lat, lon,
                         self.stations["latitude"].to_numpy(),
                         self.stations["longitude"].to_numpy())
        i = int(np.argmin(d))
        row = self.stations.iloc[i]
        return {
            "police_station": row["police_station"],
            "station_zone": row["zone"],
            "station_latitude": float(row["latitude"]),
            "station_longitude": float(row["longitude"]),
            "distance_km": round(float(d[i]), 2),
        }


def event_input_from_row(row: pd.Series) -> EventInput:
    """Engineered live-feed row -> EventInput, using only fields known at
    reporting time."""
    start = row["start_datetime"]
    start_ist = to_ist(start).isoformat() if pd.notna(start) else None
    return EventInput(
        event_type=row["event_type"],
        event_cause=row["event_cause"],
        veh_type=row["veh_type"],
        corridor=row["corridor"],
        zone=row["zone"],
        police_station=row["police_station"],
        junction=row["junction"],
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        description=str(row.get("description_text", "") or ""),
        start_datetime=start_ist,
        requires_road_closure=int(row["requires_road_closure"]),
    )


@dataclass
class DispatchOrder:
    """One actionable alert sent to a police station."""
    order_id: str
    issued_at_ist: str
    # where to act
    police_station: str
    station_zone: str
    distance_km: float
    # what happened
    event_id: str
    event_cause: str
    corridor: str
    zone: str
    latitude: float
    longitude: float
    description: str
    # forecast (ML)
    impact_score: float
    severity_band: str
    expected_duration_min: float
    duration_p90_min: float
    closure_probability: float
    major_event_probability: float
    # action plan
    manpower: int
    barricades: int
    need_diversion: bool
    diversion_plan: str
    summary: str
    drivers: list = field(default_factory=list)
    status: str = "SENT"

    def as_dict(self) -> dict:
        return asdict(self)


def build_dispatch_order(row: pd.Series, forecast: dict,
                         directory: PoliceStationDirectory,
                         issued_at: pd.Timestamp) -> DispatchOrder:
    """Combine an event row, its ML forecast/recommendation and the nearest
    station into a dispatch order. `issued_at` is the (simulated) UTC time."""
    station = directory.nearest(float(row["latitude"]), float(row["longitude"]))
    event_id = str(row.get("id", "EVT-NA"))
    return DispatchOrder(
        order_id=f"DSP-{event_id}",
        issued_at_ist=str(to_ist(issued_at)),
        police_station=station["police_station"],
        station_zone=station["station_zone"],
        distance_km=station["distance_km"],
        event_id=event_id,
        event_cause=row["event_cause"],
        corridor=row["corridor"],
        zone=row["zone"],
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        description=str(row.get("description_text", "") or "")[:300],
        impact_score=forecast["impact_score"],
        severity_band=forecast["severity_band"],
        expected_duration_min=forecast["expected_duration_min"],
        duration_p90_min=forecast["duration_p90_min"],
        closure_probability=forecast["closure_probability"],
        major_event_probability=forecast["major_event_probability"],
        manpower=forecast["manpower"],
        barricades=forecast["barricades"],
        need_diversion=forecast["need_diversion"],
        diversion_plan=forecast["diversion_plan"],
        summary=forecast["summary"],
        drivers=forecast["drivers"],
    )


# ---------------------------------------------------------------------- smoke
if __name__ == "__main__":
    from src.data_prep import engineer_features, load_raw
    from src.inference import Forecaster
    from src.live_feed import LiveFeedSimulator

    df = engineer_features(load_raw("."))
    directory = PoliceStationDirectory(df)
    print(f"Police station directory: {len(directory.stations)} stations")
    print(directory.stations.sort_values("events_handled", ascending=False)
          .head(5).to_string(index=False))

    feed = LiveFeedSimulator(df)
    forecaster = Forecaster()

    day = feed.busiest_days(5).sort_values("events").iloc[-1]["date"]
    t0 = feed.day_start(day, hour=9)
    new = feed.new_events(t0, t0 + pd.Timedelta(hours=2)).head(3)
    print(f"\nDispatching {len(new)} events reported 09:00-11:00 IST on {day}:\n")
    for _, row in new.iterrows():
        forecast = forecaster.predict(event_input_from_row(row))
        order = build_dispatch_order(row, forecast, directory, row["start_datetime"])
        print(f"  {order.order_id}: {order.event_cause} on {order.corridor}")
        print(f"    -> {order.police_station} station ({order.distance_km} km away)")
        print(f"    -> {order.summary}\n")
