"""
Train the models that power the Event-Driven Congestion system.

  1. duration  - quantile regressor -> P50 (expected) and P90 (worst-case)
                 impact duration in minutes
  2. closure   - calibrated classifier -> probability the event needs a road
                 closure (barricading + diversion)
  3. major     - calibrated classifier -> probability the event becomes a
                 high-impact incident (long duration OR road closure)

Pipeline (shared, see data_prep.make_preprocessor):
  one-hot (low-card cats) + target-encode (high-card cats) + TF-IDF/SVD on the
  free-text description + passthrough numerics (time, holiday, spatial, density,
  stretch length, keyword flags).

For each model we:
  * tune hyper-parameters with RandomizedSearchCV,
  * evaluate with THREE schemes - random k-fold, a temporal holdout, and a
    GroupKFold by location cluster - for an honest, leakage-aware estimate,
  * calibrate classifier probabilities and tune the closure threshold,
  * compute permutation feature importances,
  * refit the final model on all data and save it.

Run:  python -m src.train               (default 5 folds)
      python -m src.train --folds 5 --n-iter 20
      python -m src.train --fast         (small search, for quick iteration)
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from src.data_prep import (
    FeatureBuilder,
    MAX_DURATION_MIN,
    build_feature_frame,
    engineer_features,
    load_raw,
    make_preprocessor,
)

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
REPORTS_DIR = "reports"
RANDOM_STATE = 42
MAJOR_DURATION_MIN = 180  # 3 hours

# Prior (v1) baseline, kept for the before/after improvement narrative.
BASELINE_V1 = {
    "duration": {"median_ae_minutes": 35.0, "r2_log": 0.20},
    "closure": {"roc_auc": 0.794},
    "major": {"roc_auc": 0.883},
}

HGB_REG_PARAM_DIST = {
    "reg__learning_rate": [0.05, 0.08, 0.12],
    "reg__max_leaf_nodes": [15, 31, 63],
    "reg__max_iter": [200, 300, 400],
    "reg__min_samples_leaf": [20, 40, 80],
    "reg__l2_regularization": [0.0, 1.0, 5.0],
}
LGBM_REG_PARAM_DIST = {
    "reg__learning_rate": [0.05, 0.08, 0.12],
    "reg__num_leaves": [15, 31, 63],
    "reg__n_estimators": [200, 300, 400],
    "reg__min_child_samples": [20, 40, 80],
    "reg__reg_lambda": [0.0, 1.0, 5.0],
}
CB_REG_PARAM_DIST = {
    "reg__learning_rate": [0.05, 0.08, 0.12],
    "reg__depth": [4, 6, 8],
    "reg__iterations": [200, 300, 400],
    "reg__l2_leaf_reg": [1, 3, 5],
}

HGB_CLF_PARAM_DIST = {
    "clf__learning_rate": [0.05, 0.08, 0.12],
    "clf__max_leaf_nodes": [15, 31, 63],
    "clf__max_iter": [200, 300, 400],
    "clf__min_samples_leaf": [20, 40, 80],
    "clf__l2_regularization": [0.0, 1.0, 5.0],
}
LGBM_CLF_PARAM_DIST = {
    "clf__learning_rate": [0.05, 0.08, 0.12],
    "clf__num_leaves": [15, 31, 63],
    "clf__n_estimators": [200, 300, 400],
    "clf__min_child_samples": [20, 40, 80],
    "clf__reg_lambda": [0.0, 1.0, 5.0],
}
CB_CLF_PARAM_DIST = {
    "clf__learning_rate": [0.05, 0.08, 0.12],
    "clf__depth": [4, 6, 8],
    "clf__iterations": [200, 300, 400],
    "clf__l2_leaf_reg": [1, 3, 5],
}

# ----------------------------------------------------------------------------
# Pipeline factories
# ----------------------------------------------------------------------------

def make_hgb_regressor(quantile: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(include_closure_flag=True)),
        ("reg", HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile,
            learning_rate=0.08, max_iter=300, l2_regularization=1.0,
            random_state=RANDOM_STATE)),
    ])

def make_lgbm_regressor(quantile: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(include_closure_flag=True)),
        ("reg", LGBMRegressor(
            objective="quantile", alpha=quantile,
            learning_rate=0.08, n_estimators=300, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbose=-1)),
    ])

def make_cb_regressor(quantile: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(include_closure_flag=True)),
        ("reg", CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile}",
            learning_rate=0.08, iterations=300, l2_leaf_reg=3,
            random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)),
    ])

def make_hgb_classifier() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(include_closure_flag=False)),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=300, l2_regularization=1.0,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])

def make_lgbm_classifier() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(include_closure_flag=False)),
        ("clf", LGBMClassifier(
            learning_rate=0.08, n_estimators=300, reg_lambda=1.0,
            class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)),
    ])

def make_cb_classifier() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor(include_closure_flag=False)),
        ("clf", CatBoostClassifier(
            learning_rate=0.08, iterations=300, l2_leaf_reg=3,
            auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)),
    ])

def _save(obj, name: str):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(obj, path)
    print(f"  saved -> {path}")

def _summary(values) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "folds": [round(float(v), 4) for v in arr]}

# ----------------------------------------------------------------------------
# Fold generators (positional indices)
# ----------------------------------------------------------------------------

def folds_random(n, y=None, stratify=False, n_splits=5):
    if stratify:
        sk = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(sk.split(np.zeros(n), y))
    kf = KFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(kf.split(np.zeros(n)))

def folds_group(groups, n_splits=5):
    gk = GroupKFold(n_splits)
    return list(gk.split(np.zeros(len(groups)), groups=np.asarray(groups)))

def fold_temporal(start: pd.Series, frac=0.8):
    order = np.argsort(start.fillna(start.max()).values)
    cut = int(len(order) * frac)
    return [(order[:cut], order[cut:])]

# ----------------------------------------------------------------------------
# Evaluators
# ----------------------------------------------------------------------------

def eval_regression(make_pipes, X, y_log, folds) -> dict:
    mae, medae, rmsle, within2 = [], [], [], []
    for tr, te in folds:
        ms = [mk() for mk in make_pipes]
        for m in ms:
            m.fit(X.iloc[tr], y_log[tr])
        
        pls = [m.predict(X.iloc[te]) for m in ms]
        pl = np.mean(pls, axis=0)
        
        pred, true = np.expm1(pl), np.expm1(y_log[te])
        err = np.abs(pred - true)
        mae.append(err.mean())
        medae.append(np.median(err))
        rmsle.append(np.sqrt(np.mean((pl - y_log[te]) ** 2)))
        ratio = np.maximum(pred, true) / np.maximum(np.minimum(pred, true), 1e-9)
        within2.append(float(np.mean(ratio <= 2.0)))
    return {
        "mae_minutes": _summary(mae),
        "median_ae_minutes": _summary(medae),
        "rmsle": _summary(rmsle),
        "within_2x": _summary(within2),
    }

def eval_classification(make_pipes, X, y, folds):
    roc, pr, acc = [], [], []
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        ms = [mk() for mk in make_pipes]
        for m in ms:
            m.fit(X.iloc[tr], y[tr])
            
        probas = [m.predict_proba(X.iloc[te])[:, 1] for m in ms]
        proba = np.mean(probas, axis=0)
        oof[te] = proba
        roc.append(roc_auc_score(y[te], proba))
        pr.append(average_precision_score(y[te], proba))
        acc.append(((proba >= 0.5).astype(int) == y[te]).mean())
    return {"roc_auc": _summary(roc), "pr_auc": _summary(pr),
            "accuracy": _summary(acc)}, oof

def tune(base_pipe, param_dist, X, y, scoring, n_iter, cv_folds):
    search = RandomizedSearchCV(
        base_pipe, param_dist, n_iter=n_iter, scoring=scoring,
        cv=cv_folds, random_state=RANDOM_STATE, n_jobs=1, refit=False,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_params_, search.best_score_

def importance(models, X, y, scoring, names, n_repeats=4) -> list:
    class EnsembleWrapper(BaseEstimator):
        def __init__(self, models):
            self.models = models
        def fit(self, X, y=None):
            return self
        def predict(self, X):
            return np.mean([m.predict(X) for m in self.models], axis=0)
        def predict_proba(self, X):
            return np.mean([m.predict_proba(X) for m in self.models], axis=0)
        def __sklearn_tags__(self):
            return self.models[0].__sklearn_tags__() if hasattr(self.models[0], "__sklearn_tags__") else super().__sklearn_tags__()
        def _get_tags(self):
            return self.models[0]._get_tags() if hasattr(self.models[0], "_get_tags") else super()._get_tags()
    
    wrapper = EnsembleWrapper(models)
    wrapper._estimator_type = getattr(models[0], "_estimator_type", getattr(models[0].steps[-1][1], "_estimator_type", "regressor"))
    if hasattr(models[0], 'classes_'):
        wrapper.classes_ = models[0].classes_
        
    r = permutation_importance(wrapper, X, y, scoring=scoring,
                               n_repeats=n_repeats, random_state=RANDOM_STATE,
                               n_jobs=1)
    order = np.argsort(r.importances_mean)[::-1]
    return [{"feature": names[i], "importance": round(float(r.importances_mean[i]), 5)}
            for i in order[:15]]

# ----------------------------------------------------------------------------
# Model trainers
# ----------------------------------------------------------------------------

def train_duration(d: pd.DataFrame, n_splits, n_iter) -> dict:
    print(f"\n[1/3] Impact-duration quantile regressors (tune n_iter={n_iter})")
    X = build_feature_frame(d)
    y = np.log1p(d["duration_min"].values)

    best_hgb, _ = tune(make_hgb_regressor(0.5), HGB_REG_PARAM_DIST, X, y,
                       "neg_mean_absolute_error", n_iter, cv_folds=3)
    best_lgbm, _ = tune(make_lgbm_regressor(0.5), LGBM_REG_PARAM_DIST, X, y,
                        "neg_mean_absolute_error", n_iter, cv_folds=3)
    best_cb, _ = tune(make_cb_regressor(0.5), CB_REG_PARAM_DIST, X, y,
                      "neg_mean_absolute_error", n_iter, cv_folds=3)

    def mk_hgb(): return clone(make_hgb_regressor(0.5)).set_params(**best_hgb)
    def mk_lgbm(): return clone(make_lgbm_regressor(0.5)).set_params(**best_lgbm)
    def mk_cb(): return clone(make_cb_regressor(0.5)).set_params(**best_cb)
    mk_pipes = [mk_hgb, mk_lgbm, mk_cb]

    rnd = eval_regression(mk_pipes, X, y, folds_random(len(X)))
    tmp = eval_regression(mk_pipes, X, y, fold_temporal(d["start_datetime"]))
    grp = eval_regression(mk_pipes, X, y, folds_group(d["cluster_id"]))
    print(f"  random  : medAE={rnd['median_ae_minutes']['mean']:.1f} min  "
          f"RMSLE={rnd['rmsle']['mean']:.3f}  within2x={rnd['within_2x']['mean']:.3f}")
    print(f"  temporal: medAE={tmp['median_ae_minutes']['mean']:.1f} min  "
          f"within2x={tmp['within_2x']['mean']:.3f}")
    print(f"  group   : medAE={grp['median_ae_minutes']['mean']:.1f} min  "
          f"within2x={grp['within_2x']['mean']:.3f}")

    # final P50 + P90 models on all data
    hgb_p50 = mk_hgb().fit(X, y)
    lgbm_p50 = mk_lgbm().fit(X, y)
    cb_p50 = mk_cb().fit(X, y)
    
    def mk_hgb90(): return clone(make_hgb_regressor(0.9)).set_params(**best_hgb)
    def mk_lgbm90(): return clone(make_lgbm_regressor(0.9)).set_params(**best_lgbm)
    def mk_cb90(): return clone(make_cb_regressor(0.9)).set_params(**best_cb)
    
    hgb_p90 = mk_hgb90().fit(X, y)
    lgbm_p90 = mk_lgbm90().fit(X, y)
    cb_p90 = mk_cb90().fit(X, y)
    
    _save(hgb_p50, "duration_model_hgb")
    _save(lgbm_p50, "duration_model_lgbm")
    _save(cb_p50, "duration_model_cb")
    
    _save(hgb_p90, "duration_model_p90_hgb")
    _save(lgbm_p90, "duration_model_p90_lgbm")
    _save(cb_p90, "duration_model_p90_cb")

    imp = importance([hgb_p50, lgbm_p50, cb_p50], X, y, "neg_mean_absolute_error", list(X.columns))
    return {
        "n": int(len(d)), "n_splits": n_splits, "best_params": best_hgb,
        "median_ae_minutes": rnd["median_ae_minutes"]["mean"],
        "rmsle": rnd["rmsle"]["mean"],
        "within_2x": rnd["within_2x"]["mean"],
        "cv_random": rnd, "cv_temporal": tmp, "cv_group": grp,
        "feature_importance": imp,
    }

def _train_classifier(name, idx, X, y, start, groups, n_splits, n_iter) -> tuple:
    best_hgb, _ = tune(make_hgb_classifier(), HGB_CLF_PARAM_DIST, X, y,
                       "roc_auc", n_iter, cv_folds=3)
    best_lgbm, _ = tune(make_lgbm_classifier(), LGBM_CLF_PARAM_DIST, X, y,
                        "roc_auc", n_iter, cv_folds=3)
    best_cb, _ = tune(make_cb_classifier(), CB_CLF_PARAM_DIST, X, y,
                      "roc_auc", n_iter, cv_folds=3)

    def mk_hgb(): return clone(make_hgb_classifier()).set_params(**best_hgb)
    def mk_lgbm(): return clone(make_lgbm_classifier()).set_params(**best_lgbm)
    def mk_cb(): return clone(make_cb_classifier()).set_params(**best_cb)
    mk_pipes = [mk_hgb, mk_lgbm, mk_cb]

    rnd, oof = eval_classification(mk_pipes, X, y, folds_random(len(X), y, stratify=True))
    tmp, _ = eval_classification(mk_pipes, X, y, fold_temporal(start))
    grp, _ = eval_classification(mk_pipes, X, y, folds_group(groups))
    print(f"  random  : ROC-AUC={rnd['roc_auc']['mean']:.3f}+/-{rnd['roc_auc']['std']:.3f}  "
          f"PR-AUC={rnd['pr_auc']['mean']:.3f}")
    print(f"  temporal: ROC-AUC={tmp['roc_auc']['mean']:.3f}  group: ROC-AUC={grp['roc_auc']['mean']:.3f}")

    # tuned operating threshold (max F1) from out-of-fold probabilities
    thresholds = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    print(f"  tuned threshold (max F1) = {best_t:.2f}  (F1={max(f1s):.3f})")

    # final calibrated models on all data
    calibrated_hgb = CalibratedClassifierCV(mk_hgb(), method="isotonic", cv=3)
    calibrated_hgb.fit(X, y)
    
    calibrated_lgbm = CalibratedClassifierCV(mk_lgbm(), method="isotonic", cv=3)
    calibrated_lgbm.fit(X, y)
    
    calibrated_cb = CalibratedClassifierCV(mk_cb(), method="isotonic", cv=3)
    calibrated_cb.fit(X, y)
    
    _save(calibrated_hgb, f"{name}_model_hgb")
    _save(calibrated_lgbm, f"{name}_model_lgbm")
    _save(calibrated_cb, f"{name}_model_cb")

    models = [mk().fit(X, y) for mk in mk_pipes]
    imp = importance(models, X, y, "roc_auc", list(X.columns))
    
    metrics = {
        "n": int(len(X)), "n_splits": n_splits, "best_params": best_hgb,
        "positive_rate": float(y.mean()),
        "roc_auc": rnd["roc_auc"]["mean"], "pr_auc": rnd["pr_auc"]["mean"],
        "threshold": best_t,
        "cv_random": rnd, "cv_temporal": tmp, "cv_group": grp,
        "feature_importance": imp,
    }
    return metrics, oof, best_t

def train_closure(df: pd.DataFrame, n_splits, n_iter) -> tuple:
    print(f"\n[2/3] Road-closure (barricading/diversion) classifiers (tune n_iter={n_iter})")
    d = df.reset_index(drop=True)
    X = build_feature_frame(d)
    y = d["requires_road_closure"].astype(int).values
    m, _, t = _train_classifier("closure", d.index, X, y,
                                d["start_datetime"], d["cluster_id"],
                                n_splits, n_iter)
    return m, t

def train_major(df: pd.DataFrame, n_splits, n_iter) -> tuple:
    print(f"\n[3/3] Major-disruption classifiers (tune n_iter={n_iter})")
    d = df[df["duration_min"].notna()].reset_index(drop=True)
    y = ((d["duration_min"] >= MAJOR_DURATION_MIN)
         | (d["requires_road_closure"] == 1)).astype(int).values
    X = build_feature_frame(d)
    m, oof, t = _train_classifier("major", d.index, X, y,
                                  d["start_datetime"], d["cluster_id"],
                                  n_splits, n_iter)
    print(classification_report(y, (oof >= 0.5).astype(int),
                                target_names=["Routine", "Major"], digits=3))
    return m, t

# ----------------------------------------------------------------------------
# Reference tables / priors
# ----------------------------------------------------------------------------

def build_reference_tables(df: pd.DataFrame, thresholds: dict) -> dict:
    by_cause = (df.dropna(subset=["duration_min"])
                .groupby("event_cause")["duration_min"].median().round(1).to_dict())
    closure_by_cause = (df.groupby("event_cause")["requires_road_closure"]
                        .mean().round(3).to_dict())
    return {
        "median_duration_by_cause": by_cause,
        "closure_rate_by_cause": closure_by_cause,
        "corridor_event_counts": {k: int(v) for k, v in df["corridor"].value_counts().items()},
        "overall_median_duration": float(df["duration_min"].dropna().median()),
        "thresholds": thresholds,
    }

def main():
    parser = argparse.ArgumentParser(description="Train event-congestion models")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20,
                        help="RandomizedSearchCV iterations per model")
    parser.add_argument("--fast", action="store_true",
                        help="small search for quick iteration")
    args = parser.parse_args()
    n_splits = max(2, args.folds)
    n_iter = 4 if args.fast else args.n_iter

    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Loading and engineering features ...")
    df = engineer_features(load_raw("."))
    builder = FeatureBuilder().fit(df)
    df = builder.transform(df)
    _save(builder, "feature_builder")
    print(f"  rows={len(df)}  usable durations={df['duration_min'].notna().sum()}  "
          f"features={len(build_feature_frame(df).columns)}")

    dur = train_duration(df[df["duration_min"].notna()].reset_index(drop=True),
                         n_splits, n_iter)
    clo, clo_t = train_closure(df, n_splits, n_iter)
    maj, maj_t = train_major(df, n_splits, n_iter)

    thresholds = {"closure": clo_t, "major": maj_t}
    _save(build_reference_tables(df, thresholds), "reference_tables")

    report = {
        "dataset_rows": int(len(df)), "cv_folds": n_splits,
        "baseline_v1": BASELINE_V1,
        "duration": dur, "closure": clo, "major": maj,
        "improvement": {
            "duration_median_ae_minutes": round(
                BASELINE_V1["duration"]["median_ae_minutes"] - dur["median_ae_minutes"], 2),
            "closure_roc_auc_gain": round(clo["roc_auc"] - BASELINE_V1["closure"]["roc_auc"], 3),
            "major_roc_auc_gain": round(maj["roc_auc"] - BASELINE_V1["major"]["roc_auc"], 3),
        },
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(REPORTS_DIR, "feature_importance.json"), "w") as f:
        json.dump({"duration": dur["feature_importance"],
                   "closure": clo["feature_importance"],
                   "major": maj["feature_importance"]}, f, indent=2)
    print(f"\nMetrics -> {os.path.join(REPORTS_DIR, 'metrics.json')}")
    print("Done.")

if __name__ == "__main__":
    main()
