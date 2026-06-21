"""
Command-line forecaster for a single event.

Examples
--------
# A bus breakdown on a major corridor during evening peak:
python predict.py --cause vehicle_breakdown --veh bmtc_bus \
    --corridor "Hosur Road" --time "2024-09-10 18:30"

# A planned political rally:
python predict.py --type planned --cause public_event \
    --corridor "CBD 2" --closure 1 --time "2024-09-15 17:00"

Run with no arguments to score a default sample event.
"""

from __future__ import annotations

import argparse
import json

from src.data_prep import EventInput
from src.inference import Forecaster


def parse_args():
    p = argparse.ArgumentParser(description="Event-driven congestion forecaster")
    p.add_argument("--type", default="unplanned", dest="event_type")
    p.add_argument("--cause", default="vehicle_breakdown", dest="event_cause")
    p.add_argument("--veh", default="none", dest="veh_type")
    p.add_argument("--corridor", default="Non-corridor")
    p.add_argument("--zone", default="Unknown")
    p.add_argument("--station", default="Unknown", dest="police_station")
    p.add_argument("--junction", default="Unknown")
    p.add_argument("--desc", default="", dest="description",
                   help="free-text incident description (English/Kannada)")
    p.add_argument("--lat", type=float, default=12.9716, dest="latitude")
    p.add_argument("--lon", type=float, default=77.5946, dest="longitude")
    p.add_argument("--time", default=None, dest="start_datetime",
                   help="local (IST) time, e.g. '2024-09-10 18:30'")
    p.add_argument("--closure", type=int, default=0, dest="requires_road_closure")
    p.add_argument("--json", action="store_true", help="print raw JSON only")
    return p.parse_args()


def pretty(result: dict):
    inp = result["input"]
    print("=" * 64)
    print("EVENT-DRIVEN CONGESTION FORECAST")
    print("=" * 64)
    print(f"  Event      : {inp['event_type']} / {inp['event_cause']}"
          f" ({inp['veh_type']})")
    print(f"  Location   : {inp['corridor']} | {inp['zone']}"
          f"  ({inp['latitude']:.4f}, {inp['longitude']:.4f})")
    print(f"  Start      : {inp['start_datetime'] or 'now'}")
    print("-" * 64)
    print(f"  Impact score        : {result['impact_score']}/100"
          f"  [{result['severity_band']}]")
    print(f"  Expected duration   : {result['expected_duration_min']:.0f} min"
          f"  (worst-case {result['duration_p90_min']:.0f} min)")
    print(f"  Closure probability : {result['closure_probability']*100:.1f}%")
    print(f"  Major-event prob.   : {result['major_event_probability']*100:.1f}%")
    print("-" * 64)
    print("  RECOMMENDED DEPLOYMENT")
    print(f"    Manpower    : {result['manpower']} officer(s)")
    print(f"    Barricades  : {result['barricades']}"
          f"  ({'required' if result['need_barricading'] else 'not required'})")
    print(f"    Diversion   : {'YES' if result['need_diversion'] else 'no'}")
    print(f"    Plan        : {result['diversion_plan']}")
    print("-" * 64)
    print(f"  Key drivers : {', '.join(result['drivers'])}")
    print(f"  Summary     : {result['summary']}")
    print("=" * 64)


def main():
    args = parse_args()
    event = EventInput(
        event_type=args.event_type,
        event_cause=args.event_cause,
        veh_type=args.veh_type,
        corridor=args.corridor,
        zone=args.zone,
        police_station=args.police_station,
        junction=args.junction,
        description=args.description,
        latitude=args.latitude,
        longitude=args.longitude,
        start_datetime=args.start_datetime,
        requires_road_closure=args.requires_road_closure,
    )
    forecaster = Forecaster()
    result = forecaster.predict(event)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        pretty(result)


if __name__ == "__main__":
    main()
