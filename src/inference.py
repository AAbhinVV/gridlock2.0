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
import src.train  # Required to unpickle FrozenPreprocessor

MODELS_DIR = "models"


class Forecaster:
    def __init__(self, models_dir: str = MODELS_DIR):
        # Fix unpickling for objects saved when src.train was run as __main__
        import sys
        from src.train import FrozenPreprocessor
        sys.modules["__main__"].FrozenPreprocessor = FrozenPreprocessor

        L = lambda n: joblib.load(os.path.join(models_dir, f"{n}.joblib"))
        self.builder = L("feature_builder")
        
        self.duration_model = L("duration_model_stack")
        self.duration_model_p90 = L("duration_model_p90_stack")
        self.closure_model = L("closure_model_stack")
        self.major_model = L("major_model_stack")
        
        self.reference = L("reference_tables")
        self.thresholds = self.reference.get("thresholds", {"closure": 0.5})

    def _dur(self, model, X):
        return float(np.clip(np.expm1(model.predict(X)[0]), 1.0, 60 * 24 * 3))

    def predict(self, event: EventInput) -> dict:
        X = event.features(self.builder)

        p50 = self._dur(self.duration_model, X)
        p90 = max(self._dur(self.duration_model_p90, X), p50)

        closure_prob = float(self.closure_model.predict_proba(X)[0, 1])
        major_prob = float(self.major_model.predict_proba(X)[0, 1])

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
