"""
Live traffic feed simulator.

Replays the historical Astram event stream against a simulated clock so the
dashboard can demonstrate the full real-time loop — *detect congestion ->
forecast impact -> dispatch to the nearest police station* — without needing a
live data connection. In production this module is the swap point for a real
feed (API / webhook / control-room CAD system): the downstream contract is
simply "a DataFrame of event rows".

Usage (smoke test):
    python -m src.live_feed
"""

from __future__ import annotations

import pandas as pd

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)

# Events with an unknown duration still need to expire from the "active" set,
# otherwise the live map only ever grows. Unknowns get their cause's median
# duration; everything is capped so the demo shows realistic turnover.
DEFAULT_DURATION_MIN = 60.0
MAX_SIM_DURATION_MIN = 60 * 12  # cap live lifetime at 12 h (long works stay visible all day)


def to_utc(ist_naive: pd.Timestamp) -> pd.Timestamp:
    """Naive IST timestamp -> tz-aware UTC (the tz used by start_datetime)."""
    return (pd.Timestamp(ist_naive) - IST_OFFSET).tz_localize("UTC")


def to_ist(utc_aware: pd.Timestamp) -> pd.Timestamp:
    """tz-aware UTC -> naive IST for display."""
    return (pd.Timestamp(utc_aware) + IST_OFFSET).tz_localize(None)


class LiveFeedSimulator:
    """Replays engineered event rows (see data_prep.engineer_features) in
    simulated time. All query timestamps are tz-aware UTC."""

    def __init__(self, df: pd.DataFrame,
                 default_duration_min: float = DEFAULT_DURATION_MIN):
        ev = df[df["start_datetime"].notna()].copy()

        cause_median = ev.groupby("event_cause")["duration_min"].median()
        ev["sim_duration_min"] = (
            ev["duration_min"]
            .fillna(ev["event_cause"].map(cause_median))
            .fillna(default_duration_min)
            .clip(5.0, MAX_SIM_DURATION_MIN)
        )
        ev["sim_end"] = ev["start_datetime"] + pd.to_timedelta(
            ev["sim_duration_min"], unit="m"
        )
        self.events = ev.sort_values("start_datetime").reset_index(drop=True)

    # ------------------------------------------------------------------ query
    def new_events(self, since: pd.Timestamp, until: pd.Timestamp) -> pd.DataFrame:
        """Events *reported* in the window (since, until] — i.e. what a control
        room would see arrive between two clock ticks."""
        m = (self.events["start_datetime"] > since) & (
            self.events["start_datetime"] <= until
        )
        return self.events[m]

    def active_events(self, at: pd.Timestamp) -> pd.DataFrame:
        """Events currently affecting traffic at simulated time `at`."""
        m = (self.events["start_datetime"] <= at) & (self.events["sim_end"] > at)
        return self.events[m]

    # ------------------------------------------------------------- demo picks
    def busiest_days(self, n: int = 10) -> pd.DataFrame:
        """IST dates with the most reported events — good demo days."""
        ist_date = (self.events["start_datetime"] + IST_OFFSET).dt.date
        top = ist_date.value_counts().head(n).rename_axis("date").reset_index(name="events")
        return top.sort_values("date").reset_index(drop=True)

    @staticmethod
    def day_start(ist_date, hour: int = 7) -> pd.Timestamp:
        """UTC timestamp for `hour`:00 IST on the given date (default 07:00,
        just before the morning peak — a natural demo starting point)."""
        return to_utc(pd.Timestamp(ist_date) + pd.Timedelta(hours=hour))


# ---------------------------------------------------------------------- smoke
if __name__ == "__main__":
    from src.data_prep import engineer_features, load_raw

    feed = LiveFeedSimulator(engineer_features(load_raw(".")))
    print(f"Loaded {len(feed.events)} replayable events")
    print("\nBusiest days (IST):")
    print(feed.busiest_days(5).to_string(index=False))

    day = feed.busiest_days(5).sort_values("events").iloc[-1]["date"]
    t = feed.day_start(day)
    print(f"\nReplaying {day} from {to_ist(t)} IST in 30-min ticks:")
    for _ in range(6):
        t2 = t + pd.Timedelta(minutes=30)
        new = feed.new_events(t, t2)
        active = feed.active_events(t2)
        print(f"  {to_ist(t2).time()} IST  +{len(new):2d} new  "
              f"{len(active):3d} active  "
              f"({active['corridor'].nunique()} corridors affected)")
        t = t2
