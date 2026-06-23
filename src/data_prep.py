"""
Shared data loading, cleaning and feature engineering for the
Event-Driven Congestion forecasting system.

This module is the single source of truth for how a raw "Astram" event row
is turned into model features, so that training (train.py), batch scoring and
the live dashboard (app.py) all behave identically.

Feature groups produced for the models:
  * low-cardinality categoricals  -> one-hot encoded
  * high-cardinality categoricals -> out-of-fold target encoded
  * free-text `description`        -> char n-gram TF-IDF -> TruncatedSVD
  * numeric / engineered           -> passthrough (time, holiday, spatial,
                                       density, text keyword flags)

Note: end-coordinates are deliberately NOT used - they are only populated once
a road-closure stretch is recorded, so they leak the closure target and are
unknown at forecast time.

Some features need to be *fitted* on the training data (KMeans spatial clusters
and historical density counts). Those live in :class:`FeatureBuilder`, which is
fitted during training and reused at inference; everything else is row-wise and
deterministic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

CSV_NAME = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"

IST_OFFSET = pd.Timedelta(hours=5, minutes=30)  # file timestamps are UTC

# Bengaluru city centre (Majestic / KSR station) — used for dist_to_center.
CENTER_LAT, CENTER_LON = 12.9716, 77.5946
DEG_TO_KM = 111.0  # rough degrees-to-km at this latitude

LAT_MIN, LAT_MAX = 12.6, 13.3
LON_MIN, LON_MAX = 77.3, 77.9

MAX_DURATION_MIN = 60 * 24 * 3  # 3 days (cap implausible never-closed records)

RANDOM_STATE = 42
N_CLUSTERS = 40

LONG_LIVED_CAUSES = {
    "construction", "water_logging", "pot_holes", "road_conditions", "tree_fall",
}

# --- feature schema ---------------------------------------------------------
LOWCARD_CAT = [
    "event_type", "event_cause", "veh_type", "zone",
    "is_corridor", "long_lived_cause",
]
HIGHCARD_CAT = ["corridor", "police_station", "junction", "cluster_id"]

KEYWORD_FLAGS = [
    "kw_accident", "kw_block", "kw_divert", "kw_lane", "kw_bothside",
    "kw_bus", "kw_tree", "kw_water", "kw_work", "kw_vip", "kw_rally",
    "kw_signal", "kw_heavy", "kw_breakdown", "kw_fire", "kw_metro",
    "kw_flyover", "kw_school", "kw_hospital", "kw_overturn", "kw_pothole",
]
BASE_NUMERIC = [
    "latitude", "longitude", "hour", "dow", "month", "is_weekend", "is_peak",
    "is_holiday", "days_to_holiday",
    "corridor_density", "junction_density", "cluster_density", "zone_density",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "peak_corridor", "weekend_peak",
    "dist_to_center",
    "desc_length", "desc_word_count", "kw_total",
]
CLOSURE_FLAG = "requires_road_closure"
TEXT_COL = "description_text"

# All columns the model frame must contain (closure flag included; individual
# preprocessors decide whether to use it).
MODEL_COLUMNS = (
    LOWCARD_CAT + HIGHCARD_CAT + KEYWORD_FLAGS + BASE_NUMERIC
    + [CLOSURE_FLAG, TEXT_COL]
)

# Bengaluru / Karnataka public holidays & major festivals covering the data
# window (Nov 2023 - Apr 2024) plus common recurring dates, kept fully offline.
HOLIDAYS = {
    "2023-11-12", "2023-11-13", "2023-11-14",  # Deepavali
    "2023-11-27",  # Guru Nanak Jayanti / Kartika Purnima
    "2023-12-25",  # Christmas
    "2024-01-01", "2024-01-15", "2024-01-26",  # New Year, Sankranti, Republic Day
    "2024-03-08",  # Maha Shivaratri
    "2024-03-25",  # Holi
    "2024-03-29",  # Good Friday
    "2024-04-09",  # Ugadi
    "2024-04-11",  # Ramzan / Eid
    "2024-04-14",  # Ambedkar Jayanti
    "2024-04-17",  # Ram Navami
    # a few forward dates so future inference still has nearby anchors
    "2024-08-15", "2024-10-02", "2024-10-31", "2024-11-01", "2024-12-25",
    "2025-01-01", "2025-01-26", "2025-08-15", "2025-10-02", "2025-12-25",
}
_HOLIDAY_DATES = sorted(pd.to_datetime(list(HOLIDAYS)))

# Keyword -> substrings (English + transliteration + Kannada script).
_KEYWORD_PATTERNS = {
    "kw_accident": ["accident", "collision", "ಅಪಘಾತ"],
    "kw_block": ["block", "blocked", "jam", "stuck", "ಬ್ಲಾಕ್", "ಜಾಮ್", "ನಿಂತಿ"],
    "kw_divert": ["divert", "diversion", "ಡೈವರ್", "ತಿರುಗಿ"],
    "kw_lane": ["lane", "ಲೇನ್"],
    "kw_bothside": ["both", "two side", "ಎರಡೂ", "ಎರಡು ಕಡೆ"],
    "kw_bus": ["bus", "bmtc", "ksrtc", "ಬಸ್"],
    "kw_tree": ["tree", "branch", "ಮರ", "ಗಿಡ"],
    "kw_water": ["water", "logging", "flood", "rain", "ನೀರು", "ಮಳೆ"],
    "kw_work": ["work", "construction", "dig", "pipe", "cement", "ಕಾಮಗಾರಿ",
                "ವರ್ಕ್", "ಅಗೆ"],
    "kw_vip": ["vip", "minister", "convoy", "ವಿಐಪಿ", "ಸಚಿವ"],
    "kw_rally": ["rally", "protest", "procession", "dharna", "ಪ್ರತಿಭಟನೆ",
                 "ಮೆರವಣಿಗೆ", "ಧರಣಿ"],
    "kw_signal": ["signal", "ಸಿಗ್ನಲ್"],
    "kw_heavy": ["heavy", "lorry", "truck", "container", "ಲಾರಿ", "ಟ್ರಕ್"],
    "kw_breakdown": ["breakdown", "break down", "problem", "puncture", "engine",
                     "gear", "ಸಮಸ್ಯೆ", "ಕೆಟ್ಟು", "ಪಂಚರ್", "ಬ್ರೇಕ್"],
    "kw_fire": ["fire", "ಬೆಂಕಿ"],
    "kw_metro": ["metro", "ಮೆಟ್ರೋ"],
    "kw_flyover": ["flyover", "overpass", "ಫ್ಲೈಓವರ್"],
    "kw_school": ["school", "college", "ಶಾಲೆ", "ಕಾಲೇಜು"],
    "kw_hospital": ["hospital", "ambulance", "ಆಸ್ಪತ್ರೆ", "ಆಂಬ್ಯುಲೆನ್ಸ್"],
    "kw_overturn": ["overturn", "topple", "ಪಲ್ಟಿ"],
    "kw_pothole": ["pothole", "ಗುಂಡಿ"],
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _csv_path(data_dir: str = ".") -> str:
    return os.path.join(data_dir, CSV_NAME)


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _add_time_features(df: pd.DataFrame, dt_col: str = "start_datetime") -> pd.DataFrame:
    local = df[dt_col] + IST_OFFSET
    df["hour"] = local.dt.hour
    df["dow"] = local.dt.dayofweek
    df["month"] = local.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_peak"] = (
        local.dt.hour.between(8, 11) | local.dt.hour.between(17, 21)
    ).astype(int)
    for c, default in [("hour", 12), ("dow", 0), ("month", 1),
                       ("is_weekend", 0), ("is_peak", 0)]:
        df[c] = df[c].fillna(default).astype(int)
        
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    return df


def _add_holiday_features(df: pd.DataFrame, dt_col: str = "start_datetime") -> pd.DataFrame:
    local_date = (df[dt_col] + IST_OFFSET).dt.normalize().dt.tz_localize(None)
    hol = np.array([d.value for d in _HOLIDAY_DATES], dtype="float64")

    def nearest_days(ts):
        if pd.isna(ts):
            return 30.0, 0
        d = min(abs((ts.value - hol)) / 8.64e13)  # ns -> days
        return float(min(d, 30.0)), int(d <= 0.5)

    res = local_date.apply(nearest_days)
    df["days_to_holiday"] = [r[0] for r in res]
    df["is_holiday"] = [r[1] for r in res]
    return df


def _add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df.get("description")
    if text is None:
        text = pd.Series([""] * len(df), index=df.index)
    text = text.fillna("").astype(str)
    df["description_text"] = text
    df["desc_length"] = text.str.len().fillna(0).astype(int)
    df["desc_word_count"] = text.str.split().str.len().fillna(0).astype(int)
    low = text.str.lower()
    for flag, subs in _KEYWORD_PATTERNS.items():
        pat = "|".join(re.escape(s) for s in subs)
        df[flag] = low.str.contains(pat, regex=True, na=False).astype(int)
    df["kw_total"] = df[KEYWORD_FLAGS].sum(axis=1)
    return df


def _clean_coords(df: pd.DataFrame) -> pd.DataFrame:
    bad = (
        ~df["latitude"].between(LAT_MIN, LAT_MAX)
        | ~df["longitude"].between(LON_MIN, LON_MAX)
    )
    df.loc[bad, ["latitude", "longitude"]] = np.nan
    df["latitude"] = df["latitude"].fillna(df["latitude"].median())
    df["longitude"] = df["longitude"].fillna(df["longitude"].median())
    return df


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def load_raw(data_dir: str = ".") -> pd.DataFrame:
    return pd.read_csv(_csv_path(data_dir), low_memory=False)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise cleaning + feature engineering (no fitted state).

    Adds engineered columns and the regression target ``duration_min`` (NaN
    where it cannot be computed). Spatial cluster + density features are added
    separately by a fitted :class:`FeatureBuilder`.
    """
    df = df.copy()

    for c in ["start_datetime", "end_datetime", "closed_datetime",
              "resolved_datetime"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])

    # regression target: impact duration (minutes)
    impact_end = df.get("resolved_datetime")
    if impact_end is None:
        impact_end = pd.Series(pd.NaT, index=df.index)
    impact_end = impact_end.fillna(df.get("closed_datetime"))
    impact_end = impact_end.fillna(df.get("end_datetime"))
    df["duration_min"] = (impact_end - df["start_datetime"]).dt.total_seconds() / 60.0
    df.loc[~df["duration_min"].between(0.5, MAX_DURATION_MIN), "duration_min"] = np.nan

    # categorical cleaning
    df["event_type"] = df["event_type"].fillna("unplanned").astype(str).str.lower()
    df["event_cause"] = df["event_cause"].fillna("others").astype(str).str.strip().str.lower()
    df["veh_type"] = df["veh_type"].fillna("none").astype(str).str.lower()
    df["corridor"] = df["corridor"].fillna("Unknown").astype(str)
    df["zone"] = df["zone"].fillna("Unknown").astype(str)
    df["police_station"] = df["police_station"].fillna("Unknown").astype(str)
    df["junction"] = df.get("junction", pd.Series(index=df.index)).fillna("Unknown").astype(str)

    df["is_corridor"] = (df["corridor"] != "Non-corridor").astype(int).astype(str)
    df["long_lived_cause"] = df["event_cause"].isin(LONG_LIVED_CAUSES).astype(int).astype(str)

    rrc = df.get("requires_road_closure")
    if rrc is not None:
        df["requires_road_closure"] = (
            rrc.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
        )
    else:
        df["requires_road_closure"] = 0

    if "priority" in df.columns:
        df["is_high_priority"] = (df["priority"] == "High").astype(int)

    df = _clean_coords(df)
    df = _add_time_features(df)
    df = _add_holiday_features(df)
    df = _add_text_features(df)

    # Spatial: distance from Bengaluru city centre (approx km)
    df["dist_to_center"] = np.sqrt(
        (df["latitude"] - CENTER_LAT) ** 2
        + (df["longitude"] - CENTER_LON) ** 2
    ) * DEG_TO_KM
    
    # Interaction features
    df["peak_corridor"] = df["is_peak"] * df["is_corridor"].astype(int)
    df["weekend_peak"] = df["is_weekend"] * df["is_peak"]
    return df


class FeatureBuilder:
    """Fits + applies the spatial/density features that need training data:
    KMeans location clusters and historical event-density counts."""

    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans: Optional[KMeans] = None
        self.cluster_counts: dict = {}
        self.corridor_counts: dict = {}
        self.junction_counts: dict = {}
        self.zone_counts: dict = {}

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        coords = df[["latitude", "longitude"]].to_numpy()
        self.kmeans = KMeans(self.n_clusters, n_init=10,
                             random_state=RANDOM_STATE).fit(coords)
        clusters = self.kmeans.predict(coords)
        self.cluster_counts = pd.Series(clusters).value_counts().to_dict()
        self.corridor_counts = df["corridor"].value_counts().to_dict()
        self.junction_counts = df["junction"].value_counts().to_dict()
        self.zone_counts = df["zone"].value_counts().to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        coords = df[["latitude", "longitude"]].to_numpy()
        cl = self.kmeans.predict(coords)
        df["cluster_id"] = cl.astype(str)
        df["cluster_density"] = [self.cluster_counts.get(c, 0) for c in cl]
        df["corridor_density"] = df["corridor"].map(self.corridor_counts).fillna(0)
        df["junction_density"] = df["junction"].map(self.junction_counts).fillna(0)
        df["zone_density"] = df["zone"].map(self.zone_counts).fillna(0)
        return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-input columns with stable dtypes. Requires that the
    spatial/density columns (from FeatureBuilder.transform) are already present."""
    X = pd.DataFrame(index=df.index)
    for c in LOWCARD_CAT + HIGHCARD_CAT:
        X[c] = df[c].astype(str)
    for c in BASE_NUMERIC + KEYWORD_FLAGS + [CLOSURE_FLAG]:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    return X[MODEL_COLUMNS]


def make_preprocessor(include_closure_flag: bool) -> ColumnTransformer:
    """ColumnTransformer: one-hot (low-card) + target-encode (high-card) +
    TF-IDF/SVD (text) + passthrough numerics. Used inside every model pipeline
    so encoders are fit (and cross-fit for target encoding) per training fold."""
    numeric = BASE_NUMERIC + KEYWORD_FLAGS + ([CLOSURE_FLAG] if include_closure_flag else [])
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                                  min_df=8, max_features=3000)),
        ("svd", TruncatedSVD(n_components=20, random_state=RANDOM_STATE)),
    ])
    return ColumnTransformer(
        transformers=[
            ("low", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                  sparse_output=False), LOWCARD_CAT),
            ("high", TargetEncoder(target_type="auto", random_state=RANDOM_STATE,
                                   cv=3), HIGHCARD_CAT),
            ("txt", text_pipe, TEXT_COL),
            ("num", "passthrough", numeric),
        ]
    )


# ----------------------------------------------------------------------------
# Single-event input (used by CLI + dashboard)
# ----------------------------------------------------------------------------

@dataclass
class EventInput:
    """A single (future or live) event as known at reporting time."""
    event_type: str = "unplanned"
    event_cause: str = "vehicle_breakdown"
    veh_type: str = "none"
    corridor: str = "Non-corridor"
    zone: str = "Unknown"
    police_station: str = "Unknown"
    junction: str = "Unknown"
    latitude: float = 12.9716
    longitude: float = 77.5946
    description: str = ""
    start_datetime: Optional[str] = None  # ISO string in *local* (IST) time
    requires_road_closure: int = 0

    def to_raw_row(self) -> pd.DataFrame:
        if self.start_datetime:
            local = pd.to_datetime(self.start_datetime, errors="coerce")
            utc = local - IST_OFFSET
        else:
            utc = pd.Timestamp.utcnow().tz_localize(None)
        row = {
            "event_type": self.event_type,
            "event_cause": self.event_cause,
            "veh_type": self.veh_type,
            "corridor": self.corridor,
            "zone": self.zone,
            "police_station": self.police_station,
            "junction": self.junction,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "description": self.description,
            "requires_road_closure": bool(self.requires_road_closure),
            "start_datetime": pd.Timestamp(utc).tz_localize("UTC").isoformat(),
            "end_datetime": None,
            "closed_datetime": None,
            "resolved_datetime": None,
            "priority": None,
        }
        return pd.DataFrame([row])

    def features(self, builder: FeatureBuilder) -> pd.DataFrame:
        eng = engineer_features(self.to_raw_row())
        eng = builder.transform(eng)
        return build_feature_frame(eng)

    def as_dict(self) -> dict:
        return asdict(self)
