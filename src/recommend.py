"""
Recommendation engine.

Takes the raw model predictions (impact duration, closure probability,
priority probability) for an event and turns them into an operational plan:

    * an Impact Score (0-100) that quantifies expected congestion severity
    * recommended manpower (number of officers)
    * barricading plan (need + count)
    * diversion plan (need + guidance)

The mapping is transparent and rule-based on top of the ML predictions, so it
is explainable to traffic-control decision makers (important for adoption),
while the *inputs* to the rules are learned from history.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# Corridors are arterial roads; an incident there ripples much further than on
# an interior ("Non-corridor") street, so they get a load multiplier.
CORRIDOR_IMPORTANCE = {
    "Non-corridor": 0.8,
    "Unknown": 1.0,
}
DEFAULT_CORRIDOR_IMPORTANCE = 1.25  # any named arterial corridor


@dataclass
class Recommendation:
    impact_score: float            # 0-100
    severity_band: str             # Low / Moderate / High / Severe
    expected_duration_min: float   # P50
    duration_p90_min: float        # worst-case (P90)
    closure_probability: float
    major_event_probability: float
    manpower: int
    barricades: int
    need_barricading: bool
    need_diversion: bool
    diversion_plan: str
    summary: str
    drivers: list

    def as_dict(self) -> dict:
        return asdict(self)


def _duration_component(duration_min: float) -> float:
    """Map minutes -> 0..1 using a log curve (saturates for very long events)."""
    import math
    # 30 min -> ~0.27, 2h -> ~0.5, 8h -> ~0.75, 24h+ -> ~0.9
    return min(1.0, math.log1p(max(duration_min, 0)) / math.log1p(60 * 24))


def compute_impact_score(
    duration_min: float,
    closure_prob: float,
    major_prob: float,
    corridor: str,
    is_peak: int,
) -> tuple[float, list]:
    """Weighted blend of the three model outputs + context, scaled to 0-100."""
    dur_c = _duration_component(duration_min)
    base = 0.45 * dur_c + 0.30 * closure_prob + 0.25 * major_prob

    importance = CORRIDOR_IMPORTANCE.get(corridor, DEFAULT_CORRIDOR_IMPORTANCE)
    peak_mult = 1.15 if is_peak else 1.0

    score = min(100.0, base * 100.0 * importance * peak_mult)

    drivers = []
    if dur_c > 0.6:
        drivers.append("long expected duration")
    if closure_prob > 0.4:
        drivers.append("likely road closure")
    if major_prob > 0.6:
        drivers.append("high-impact event profile")
    if importance >= DEFAULT_CORRIDOR_IMPORTANCE:
        drivers.append(f"arterial corridor ({corridor})")
    if is_peak:
        drivers.append("occurs during peak hours")
    if not drivers:
        drivers.append("routine, low-impact event")
    return round(score, 1), drivers


def _severity_band(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Severe"


def _manpower(score: float, closure_prob: float, is_peak: int,
              duration_min: float, duration_p90: float) -> int:
    """Officer count scales with impact; closures, peak and a wide P50->P90
    uncertainty band add a surge (risk-aware staffing)."""
    if score < 25:
        base = 2
    elif score < 50:
        base = 4
    elif score < 75:
        base = 7
    else:
        base = 11
    if closure_prob > 0.5:
        base += 3
    if is_peak:
        base += 2
    # Risk surge: if the worst case is much longer than expected, staff for it.
    if duration_p90 > max(duration_min, 1) * 2.5 and duration_p90 > 180:
        base += 2
    return int(base)


def _barricades(score: float, closure_prob: float, need_diversion: bool,
                closure_threshold: float) -> tuple[bool, int]:
    need = closure_prob >= closure_threshold or score >= 60
    if not need:
        return False, 0
    count = 4
    if closure_prob > 0.6:
        count += 4
    if need_diversion:
        count += 4
    if score >= 75:
        count += 4
    return True, int(count)


def _diversion_plan(
    need_diversion: bool, corridor: str, duration_min: float, closure_prob: float
) -> str:
    if not need_diversion:
        return "No diversion required. Manage flow with on-site officers and signage."
    arterial = corridor not in ("Non-corridor", "Unknown")
    if arterial:
        return (
            f"Activate a signed diversion off {corridor}. Pre-position officers at the "
            "two nearest upstream junctions to divert through-traffic onto parallel "
            "service roads, and alert the adjacent corridor control room. "
            "Publish the diversion on advisory channels before peak build-up."
        )
    return (
        "Set up a local diversion around the affected stretch using the nearest "
        "cross streets. Place advance-warning signage ~200 m upstream in both "
        "directions and station an officer to wave traffic through."
    )


def recommend(
    duration_min: float,
    closure_prob: float,
    major_prob: float,
    corridor: str = "Non-corridor",
    is_peak: int = 0,
    duration_p90: float | None = None,
    closure_threshold: float = 0.5,
) -> Recommendation:
    if duration_p90 is None:
        duration_p90 = duration_min
    score, drivers = compute_impact_score(
        duration_min, closure_prob, major_prob, corridor, is_peak
    )
    band = _severity_band(score)

    need_diversion = closure_prob >= closure_threshold or (score >= 65 and duration_min > 90)
    need_barricading, barricades = _barricades(score, closure_prob, need_diversion,
                                               closure_threshold)
    manpower = _manpower(score, closure_prob, is_peak, duration_min, duration_p90)
    diversion_plan = _diversion_plan(need_diversion, corridor, duration_min, closure_prob)

    if duration_p90 > max(duration_min, 1) * 2.5 and duration_p90 > 180:
        drivers.append("high duration uncertainty")

    def _fmt(m):
        return f"{m:.0f} min" if m < 90 else f"{m / 60.0:.1f} h"
    summary = (
        f"{band} impact (score {score}). Expected ~{_fmt(duration_min)} "
        f"(worst-case ~{_fmt(duration_p90)}). "
        f"Deploy {manpower} officer(s)"
        + (f", {barricades} barricades" if need_barricading else ", no barricades")
        + ("; set up a diversion." if need_diversion else "; no diversion needed.")
    )

    return Recommendation(
        impact_score=score,
        severity_band=band,
        expected_duration_min=round(duration_min, 1),
        duration_p90_min=round(duration_p90, 1),
        closure_probability=round(closure_prob, 3),
        major_event_probability=round(major_prob, 3),
        manpower=manpower,
        barricades=barricades,
        need_barricading=need_barricading,
        need_diversion=need_diversion,
        diversion_plan=diversion_plan,
        summary=summary,
        drivers=drivers,
    )
