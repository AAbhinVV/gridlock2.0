"""
Inference helpers: load the trained models + fitted feature builder once and
turn an EventInput into a full forecast + operational recommendation.
Shared by the CLI and the dashboard.
"""

from __future__ import annotations

import os

import joblib
import numpy as np

from src.data_prep import EventInput
from src.recommend import Recommendation, recommend

MODELS_DIR = "models"


class Forecaster:
    def __init__(self, models_dir: str = MODELS_DIR):
        L = lambda n: joblib.load(os.path.join(models_dir, f"{n}.joblib"))
        self.builder = L("feature_builder")
        
        self.dur_hgb = L("duration_model_hgb")
        self.dur_lgbm = L("duration_model_lgbm")
        self.dur_cb = L("duration_model_cb")
        
        self.dur90_hgb = L("duration_model_p90_hgb")
        self.dur90_lgbm = L("duration_model_p90_lgbm")
        self.dur90_cb = L("duration_model_p90_cb")
        
        self.clo_hgb = L("closure_model_hgb")
        self.clo_lgbm = L("closure_model_lgbm")
        self.clo_cb = L("closure_model_cb")
        
        self.maj_hgb = L("major_model_hgb")
        self.maj_lgbm = L("major_model_lgbm")
        self.maj_cb = L("major_model_cb")
        
        self.reference = L("reference_tables")
        self.thresholds = self.reference.get("thresholds", {"closure": 0.5})

    def _dur(self, models, X):
        pls = [m.predict(X)[0] for m in models]
        pl = np.mean(pls)
        return float(np.clip(np.expm1(pl), 1.0, 60 * 24 * 3))

    def predict(self, event: EventInput) -> dict:
        X = event.features(self.builder)

        p50 = self._dur([self.dur_hgb, self.dur_lgbm, self.dur_cb], X)
        p90 = max(self._dur([self.dur90_hgb, self.dur90_lgbm, self.dur90_cb], X), p50)

        closure_prob = float(np.mean([
            self.clo_hgb.predict_proba(X)[0, 1],
            self.clo_lgbm.predict_proba(X)[0, 1],
            self.clo_cb.predict_proba(X)[0, 1]
        ]))
        
        major_prob = float(np.mean([
            self.maj_hgb.predict_proba(X)[0, 1],
            self.maj_lgbm.predict_proba(X)[0, 1],
            self.maj_cb.predict_proba(X)[0, 1]
        ]))

        rec: Recommendation = recommend(
            duration_min=p50,
            closure_prob=closure_prob,
            major_prob=major_prob,
            corridor=event.corridor,
            is_peak=int(X["is_peak"].iloc[0]),
            duration_p90=p90,
            closure_threshold=float(self.thresholds.get("closure", 0.5)),
        )
        out = rec.as_dict()
        out["input"] = event.as_dict()
        return out
