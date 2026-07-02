"""
Corridor-level congestion detection.

Turns the set of *currently active* events from the live feed into a
per-corridor congestion index and a traffic status (Light / Moderate / Heavy /
Severe). Detection is deliberately rule-based and instantaneous — it runs on
every clock tick over every active event — while the (heavier) ML models are
reserved for forecasting each event's impact and sizing the response.

Weighting per active event:
    base 1.0
    +1.5 if a road closure is declared      (a blocked lane ≫ a slow lane)
    +0.5 if it is ongoing during peak hours
    ×1.25 if on a named arterial corridor   (impact ripples much further)
"""

from __future__ import annotations

import pandas as pd

NON_ARTERIAL = {"Non-corridor", "Unknown"}
ARTERIAL_MULTIPLIER = 1.25

# (min index, status). Calibrated on the busiest replay days so a normal peak
# shows a mix of levels rather than all-red.
LEVELS = [
    (6.0, "Severe"),
    (3.5, "Heavy"),
    (1.5, "Moderate"),
    (0.0, "Light"),
]

STATUS_COLORS = {
    "Light": "#2ecc71",
    "Moderate": "#f1c40f",
    "Heavy": "#e67e22",
    "Severe": "#e74c3c",
}


def congestion_status(index: float) -> str:
    for threshold, status in LEVELS:
        if index >= threshold:
            return status
    return "Light"


def corridor_congestion(active: pd.DataFrame) -> pd.DataFrame:
    """Aggregate active events into a per-corridor congestion table, sorted
    most-congested first. Expects engineered rows (requires_road_closure,
    is_peak columns present)."""
    cols = ["corridor", "congestion_index", "status", "active_events", "closures"]
    if active.empty:
        return pd.DataFrame(columns=cols)

    a = active.copy()
    weight = (
        1.0
        + 1.5 * a["requires_road_closure"].astype(float)
        + 0.5 * a["is_peak"].astype(float)
    )
    weight *= [
        1.0 if c in NON_ARTERIAL else ARTERIAL_MULTIPLIER for c in a["corridor"]
    ]
    a["weight"] = weight

    g = a.groupby("corridor").agg(
        congestion_index=("weight", "sum"),
        active_events=("weight", "size"),
        closures=("requires_road_closure", "sum"),
    ).reset_index()
    g["congestion_index"] = g["congestion_index"].round(2)
    g["status"] = g["congestion_index"].map(congestion_status)
    g["closures"] = g["closures"].astype(int)
    return g.sort_values("congestion_index", ascending=False).reset_index(drop=True)[cols]


def congested_corridors(active: pd.DataFrame, min_status: str = "Moderate") -> pd.DataFrame:
    """Corridors at or above `min_status` — the ones worth an operator's attention."""
    order = {s: i for i, (_, s) in enumerate(LEVELS)}  # Severe=0 ... Light=3
    table = corridor_congestion(active)
    return table[table["status"].map(order) <= order[min_status]]
