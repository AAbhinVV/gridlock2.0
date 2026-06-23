"""
Train on a train split, then TEST on a held-out split and write every
test-set event's forecast + recommended deployment to ``predictions.csv``.

Honest evaluation: the feature builder (spatial clusters + density priors) and
all models are fit on the TRAIN split only, then applied to unseen test rows.

Run:  python make_predictions.py            (default 20% test split)
      python make_predictions.py --test-size 0.25
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_prep import (
    FeatureBuilder,
    build_feature_frame,
    engineer_features,
    load_raw,
)
from src.recommend import recommend
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from src.train import (
    MAJOR_DURATION_MIN, 
    make_hgb_classifier, make_lgbm_classifier, make_cb_classifier,
    make_hgb_regressor, make_lgbm_regressor, make_cb_regressor
)

OUT_CSV = "predictions.csv"
RANDOM_STATE = 42


def _dur(model, X):
    return np.clip(np.expm1(model.predict(X)), 1.0, 60 * 24 * 3)


def main():
    parser = argparse.ArgumentParser(description="Generate held-out test predictions")
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    print("Loading and engineering features ...")
    df = engineer_features(load_raw(".")).reset_index(drop=True)
    df["major_actual"] = np.where(
        df["duration_min"].notna(),
        ((df["duration_min"] >= MAJOR_DURATION_MIN) | (df["requires_road_closure"] == 1)).astype("Int64"),
        pd.NA,
    )

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=RANDOM_STATE,
        stratify=df["requires_road_closure"],
    )
    print(f"  train={len(train_df)}  test={len(test_df)}")

    # Feature builder fit on TRAIN only, then applied to both splits.
    builder = FeatureBuilder().fit(train_df)
    train_t = builder.transform(train_df)
    test_t = builder.transform(test_df).reset_index(drop=True)
    X_test = build_feature_frame(test_t)

    # 1. duration P50 + P90 (rows with a known duration)
    print("Training duration regressors on train split ...")
    dtr = train_t[train_t["duration_min"].notna()]
    Xdtr, ydtr = build_feature_frame(dtr), np.log1p(dtr["duration_min"].values)
    
    def mk_stack(quantile=0.5):
        return StackingRegressor(
            estimators=[
                ("hgb", make_hgb_regressor(quantile)),
                ("lgbm", make_lgbm_regressor(quantile)),
                ("cb", make_cb_regressor(quantile))
            ],
            final_estimator=RidgeCV(),
            n_jobs=1
        )

    p50_stack = mk_stack(0.5).fit(Xdtr, ydtr)
    pred_p50 = _dur(p50_stack, X_test)
    
    p90_stack = mk_stack(0.9).fit(Xdtr, ydtr)
    pred_p90 = np.maximum(_dur(p90_stack, X_test), pred_p50)

    # 2. closure classifier (all rows)
    print("Training closure classifiers on train split ...")
    X_clo, y_clo = build_feature_frame(train_t), train_t["requires_road_closure"].astype(int).values
    
    def mk_stack_clf():
        return StackingClassifier(
            estimators=[
                ("hgb", make_hgb_classifier()),
                ("lgbm", make_lgbm_classifier()),
                ("cb", make_cb_classifier())
            ],
            final_estimator=LogisticRegression(),
            n_jobs=1
        )
        
    clo_stack = mk_stack_clf().fit(X_clo, y_clo)
    closure_prob = clo_stack.predict_proba(X_test)[:, 1]

    # 3. major-disruption classifier (rows with known duration)
    print("Training major-disruption classifiers on train split ...")
    maj_y = ((dtr["duration_min"] >= MAJOR_DURATION_MIN)
             | (dtr["requires_road_closure"] == 1)).astype(int).values
    maj_stack = mk_stack_clf().fit(Xdtr, maj_y)
    major_prob = maj_stack.predict_proba(X_test)[:, 1]

    # tuned closure threshold from the trained reference tables, if available
    closure_thr = 0.5
    ref_path = os.path.join("models", "reference_tables.joblib")
    if os.path.exists(ref_path):
        closure_thr = joblib.load(ref_path).get("thresholds", {}).get("closure", 0.5)

    print("Building recommendations ...")
    rows = []
    for i in range(len(test_t)):
        ev = test_t.iloc[i]
        rec = recommend(
            duration_min=float(pred_p50[i]), closure_prob=float(closure_prob[i]),
            major_prob=float(major_prob[i]), corridor=str(ev["corridor"]),
            is_peak=int(ev["is_peak"]), duration_p90=float(pred_p90[i]),
            closure_threshold=float(closure_thr),
        )
        rows.append({
            "id": ev["id"], "event_type": ev["event_type"],
            "event_cause": ev["event_cause"], "veh_type": ev["veh_type"],
            "corridor": ev["corridor"], "zone": ev["zone"],
            "is_peak": int(ev["is_peak"]),
            "actual_duration_min": (round(float(ev["duration_min"]), 1)
                                    if pd.notna(ev["duration_min"]) else ""),
            "actual_requires_closure": int(ev["requires_road_closure"]),
            "actual_major": ("" if pd.isna(ev["major_actual"]) else int(ev["major_actual"])),
            "pred_duration_p50_min": round(float(pred_p50[i]), 1),
            "pred_duration_p90_min": round(float(pred_p90[i]), 1),
            "pred_closure_prob": round(float(closure_prob[i]), 3),
            "pred_major_prob": round(float(major_prob[i]), 3),
            "impact_score": rec.impact_score, "severity_band": rec.severity_band,
            "rec_manpower": rec.manpower, "rec_barricades": rec.barricades,
            "need_barricading": rec.need_barricading, "need_diversion": rec.need_diversion,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} test-set predictions -> {OUT_CSV}")

    # held-out scorecard
    mask = test_t["duration_min"].notna().values
    if mask.sum():
        mae = mean_absolute_error(test_t["duration_min"].values[mask], pred_p50[mask])
        medae = np.median(np.abs(test_t["duration_min"].values[mask] - pred_p50[mask]))
        print(f"  duration : MAE={mae:.1f} min  medAE={medae:.1f} min (n={int(mask.sum())})")
    print(f"  closure  : ROC-AUC="
          f"{roc_auc_score(test_t['requires_road_closure'].astype(int), closure_prob):.3f}")
    mm = test_t["major_actual"].notna().values
    if mm.sum():
        print(f"  major    : ROC-AUC="
              f"{roc_auc_score(test_t['major_actual'].astype('Int64')[mm].astype(int), major_prob[mm]):.3f}")


if __name__ == "__main__":
    main()
