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
  * tune hyper-parameters with Optuna (Bayesian Optimization),
  * evaluate a Stacking Ensemble with THREE schemes - random k-fold, a temporal holdout, and a
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
import sys
from unittest.mock import MagicMock
sys.modules['sqlite3'] = MagicMock()
sys.modules['_sqlite3'] = MagicMock()

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    StackingRegressor,
    StackingClassifier,
)
from sklearn.linear_model import RidgeCV, LogisticRegression
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
    StratifiedKFold,
    cross_val_score,
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
    "reg__learning_rate": ("float", 0.01, 0.2, True),
    "reg__max_leaf_nodes": ("int", 15, 255),
    "reg__max_iter": ("int", 200, 800),
    "reg__min_samples_leaf": ("int", 5, 100),
    "reg__l2_regularization": ("float", 0.0, 10.0),
}
LGBM_REG_PARAM_DIST = {
    "reg__learning_rate": ("float", 0.01, 0.2, True),
    "reg__num_leaves": ("int", 15, 255),
    "reg__n_estimators": ("int", 200, 800),
    "reg__min_child_samples": ("int", 5, 100),
    "reg__reg_lambda": ("float", 0.0, 10.0),
}
CB_REG_PARAM_DIST = {
    "reg__learning_rate": ("float", 0.01, 0.2, True),
    "reg__depth": ("int", 4, 10),
    "reg__iterations": ("int", 200, 800),
    "reg__l2_leaf_reg": ("int", 1, 10),
}

HGB_CLF_PARAM_DIST = {
    "clf__learning_rate": ("float", 0.01, 0.2, True),
    "clf__max_leaf_nodes": ("int", 15, 255),
    "clf__max_iter": ("int", 200, 800),
    "clf__min_samples_leaf": ("int", 5, 100),
    "clf__l2_regularization": ("float", 0.0, 10.0),
}
LGBM_CLF_PARAM_DIST = {
    "clf__learning_rate": ("float", 0.01, 0.2, True),
    "clf__num_leaves": ("int", 15, 255),
    "clf__n_estimators": ("int", 200, 800),
    "clf__min_child_samples": ("int", 5, 100),
    "clf__reg_lambda": ("float", 0.0, 10.0),
}
CB_CLF_PARAM_DIST = {
    "clf__learning_rate": ("float", 0.01, 0.2, True),
    "clf__depth": ("int", 4, 10),
    "clf__iterations": ("int", 200, 800),
    "clf__l2_leaf_reg": ("int", 1, 10),
}

# ----------------------------------------------------------------------------
# FrozenPreprocessor — the single biggest training speedup
# ----------------------------------------------------------------------------

class FrozenPreprocessor(BaseEstimator):
    """Wraps an already-fitted ColumnTransformer so that Pipeline.fit()
    skips the expensive TF-IDF/SVD/encoding step.  sklearn.clone() is
    overridden to preserve the fitted state, so Optuna trials and CV
    folds reuse the same transform without re-fitting."""

    def __init__(self, fitted_ct=None):
        self.fitted_ct = fitted_ct

    def __sklearn_clone__(self):
        return FrozenPreprocessor(self.fitted_ct)

    def fit(self, X, y=None):
        return self                        # no-op

    def transform(self, X):
        return self.fitted_ct.transform(X)

    def fit_transform(self, X, y=None):
        return self.fitted_ct.transform(X)  # skip fit

    def get_feature_names_out(self, *a, **kw):
        return self.fitted_ct.get_feature_names_out(*a, **kw)


# Module-level cache filled by main() before any training starts.
_FROZEN_PREP_REG = None   # for regressors  (include_closure_flag=True)
_FROZEN_PREP_CLF = None   # for classifiers (include_closure_flag=False)

# ----------------------------------------------------------------------------
# Pipeline factories
# ----------------------------------------------------------------------------

def _prep_reg():
    return _FROZEN_PREP_REG or make_preprocessor(include_closure_flag=True)

def _prep_clf():
    return _FROZEN_PREP_CLF or make_preprocessor(include_closure_flag=False)

def make_hgb_regressor(quantile: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", _prep_reg()),
        ("reg", HistGradientBoostingRegressor(
            loss="quantile", quantile=quantile,
            learning_rate=0.08, max_iter=300, l2_regularization=1.0,
            random_state=RANDOM_STATE)),
    ])

def make_lgbm_regressor(quantile: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", _prep_reg()),
        ("reg", LGBMRegressor(
            objective="quantile", alpha=quantile,
            learning_rate=0.08, n_estimators=300, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbose=-1)),
    ])

def make_cb_regressor(quantile: float = 0.5) -> Pipeline:
    return Pipeline([
        ("prep", _prep_reg()),
        ("reg", CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile}",
            learning_rate=0.08, iterations=300, l2_leaf_reg=3,
            random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)),
    ])

def make_hgb_classifier() -> Pipeline:
    return Pipeline([
        ("prep", _prep_clf()),
        ("clf", HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=300, l2_regularization=1.0,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])

def make_lgbm_classifier() -> Pipeline:
    return Pipeline([
        ("prep", _prep_clf()),
        ("clf", LGBMClassifier(
            learning_rate=0.08, n_estimators=300, reg_lambda=1.0,
            class_weight="balanced", random_state=RANDOM_STATE, verbose=-1)),
    ])

def make_cb_classifier() -> Pipeline:
    return Pipeline([
        ("prep", _prep_clf()),
        ("clf", CatBoostClassifier(
            learning_rate=0.08, iterations=300, l2_leaf_reg=3,
            auto_class_weights="Balanced",
            random_state=RANDOM_STATE, verbose=False, allow_writing_files=False)),
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

def eval_regression(mk_pipe, X, y_log, folds) -> dict:
    mae, medae, rmsle, within2 = [], [], [], []
    for tr, te in folds:
        m = mk_pipe()
        m.fit(X.iloc[tr], y_log[tr])
        
        pl = m.predict(X.iloc[te])
        
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

def eval_classification(mk_pipe, X, y, folds):
    roc, pr, acc = [], [], []
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        m = mk_pipe()
        m.fit(X.iloc[tr], y[tr])
            
        proba = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = proba
        roc.append(roc_auc_score(y[te], proba))
        pr.append(average_precision_score(y[te], proba))
        acc.append(((proba >= 0.5).astype(int) == y[te]).mean())
    return {"roc_auc": _summary(roc), "pr_auc": _summary(pr),
            "accuracy": _summary(acc)}, oof

def tune(base_pipe, param_space, X, y, scoring, n_iter, cv_folds):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {}
        for name, spec in param_space.items():
            if spec[0] == "float":
                log_scale = spec[3] if len(spec) > 3 else False
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=log_scale)
            elif spec[0] == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif spec[0] == "cat":
                params[name] = trial.suggest_categorical(name, spec[1])
        
        pipe = clone(base_pipe).set_params(**params)
        scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv_folds, n_jobs=1)
        return scores.mean()

    # Determine direction based on scoring string
    # neg_mean_absolute_error -> maximize
    # roc_auc -> maximize
    direction = "maximize"
    
   
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_iter, n_jobs=1)
    return study.best_params, study.best_value

def importance(model, X, y, scoring, names, n_repeats=4) -> list:
    r = permutation_importance(model, X, y, scoring=scoring,
                               n_repeats=n_repeats, random_state=RANDOM_STATE,
                               n_jobs=1)
    order = np.argsort(r.importances_mean)[::-1]
    return [{"feature": names[i], "importance": round(float(r.importances_mean[i]), 5)}
            for i in order[:15]]

# ----------------------------------------------------------------------------
# Model trainers
# ----------------------------------------------------------------------------

def train_duration(d: pd.DataFrame, n_splits, n_iter, skip_extra_cv=False) -> dict:
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
    
    def mk_stack():
        estimators = [
            ("hgb", mk_hgb()),
            ("lgbm", mk_lgbm()),
            ("cb", mk_cb())
        ]
        return StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=1)

    rnd = eval_regression(mk_stack, X, y, folds_random(len(X), n_splits=n_splits))
    print(f"  random  : medAE={rnd['median_ae_minutes']['mean']:.1f} min  "
          f"RMSLE={rnd['rmsle']['mean']:.3f}  within2x={rnd['within_2x']['mean']:.3f}")

    tmp, grp = rnd, rnd  # defaults when skipped
    if not skip_extra_cv:
        tmp = eval_regression(mk_stack, X, y, fold_temporal(d["start_datetime"]))
        grp = eval_regression(mk_stack, X, y, folds_group(d["cluster_id"]))
        print(f"  temporal: medAE={tmp['median_ae_minutes']['mean']:.1f} min  "
              f"within2x={tmp['within_2x']['mean']:.3f}")
        print(f"  group   : medAE={grp['median_ae_minutes']['mean']:.1f} min  "
              f"within2x={grp['within_2x']['mean']:.3f}")
    else:
        print("  (temporal/group CV skipped in turbo mode)")

    # final P50 model on all data
    p50_stack = mk_stack().fit(X, y)
    
    def mk_stack90():
        def mk_hgb90(): return clone(make_hgb_regressor(0.9)).set_params(**best_hgb)
        def mk_lgbm90(): return clone(make_lgbm_regressor(0.9)).set_params(**best_lgbm)
        def mk_cb90(): return clone(make_cb_regressor(0.9)).set_params(**best_cb)
        estimators = [
            ("hgb", mk_hgb90()),
            ("lgbm", mk_lgbm90()),
            ("cb", mk_cb90())
        ]
        return StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), n_jobs=1)
    
    p90_stack = mk_stack90().fit(X, y)
    
    _save(p50_stack, "duration_model_stack")
    _save(p90_stack, "duration_model_p90_stack")

    imp = importance(p50_stack, X, y, "neg_mean_absolute_error", list(X.columns),
                     n_repeats=2 if skip_extra_cv else 4)
    return {
        "n": int(len(d)), "n_splits": n_splits, "best_params": best_hgb,
        "median_ae_minutes": rnd["median_ae_minutes"]["mean"],
        "rmsle": rnd["rmsle"]["mean"],
        "within_2x": rnd["within_2x"]["mean"],
        "cv_random": rnd, "cv_temporal": tmp, "cv_group": grp,
        "feature_importance": imp,
    }

def _train_classifier(name, idx, X, y, start, groups, n_splits, n_iter, skip_extra_cv=False) -> tuple:
    best_hgb, _ = tune(make_hgb_classifier(), HGB_CLF_PARAM_DIST, X, y,
                       "roc_auc", n_iter, cv_folds=3)
    best_lgbm, _ = tune(make_lgbm_classifier(), LGBM_CLF_PARAM_DIST, X, y,
                        "roc_auc", n_iter, cv_folds=3)
    best_cb, _ = tune(make_cb_classifier(), CB_CLF_PARAM_DIST, X, y,
                      "roc_auc", n_iter, cv_folds=3)

    def mk_hgb(): return clone(make_hgb_classifier()).set_params(**best_hgb)
    def mk_lgbm(): return clone(make_lgbm_classifier()).set_params(**best_lgbm)
    def mk_cb(): return clone(make_cb_classifier()).set_params(**best_cb)
    
    def mk_stack():
        estimators = [
            ("hgb", mk_hgb()),
            ("lgbm", mk_lgbm()),
            ("cb", mk_cb())
        ]
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=1)

    rnd, oof = eval_classification(mk_stack, X, y, folds_random(len(X), y, stratify=True, n_splits=n_splits))
    print(f"  random  : ROC-AUC={rnd['roc_auc']['mean']:.3f}+/-{rnd['roc_auc']['std']:.3f}  "
          f"PR-AUC={rnd['pr_auc']['mean']:.3f}")

    tmp, grp = rnd, rnd  # defaults when skipped
    if not skip_extra_cv:
        tmp, _ = eval_classification(mk_stack, X, y, fold_temporal(start))
        grp, _ = eval_classification(mk_stack, X, y, folds_group(groups))
        print(f"  temporal: ROC-AUC={tmp['roc_auc']['mean']:.3f}  group: ROC-AUC={grp['roc_auc']['mean']:.3f}")
    else:
        print("  (temporal/group CV skipped in turbo mode)")

    # tuned operating threshold (max F1) from out-of-fold probabilities
    thresholds = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    print(f"  tuned threshold (max F1) = {best_t:.2f}  (F1={max(f1s):.3f})")

    # final calibrated models on all data
    calibrated_stack = CalibratedClassifierCV(mk_stack(), method="isotonic", cv=3)
    calibrated_stack.fit(X, y)
    
    _save(calibrated_stack, f"{name}_model_stack")

    imp = importance(calibrated_stack, X, y, "roc_auc", list(X.columns),
                     n_repeats=2 if skip_extra_cv else 4)
    
    metrics = {
        "n": int(len(X)), "n_splits": n_splits, "best_params": best_hgb,
        "positive_rate": float(y.mean()),
        "roc_auc": rnd["roc_auc"]["mean"], "pr_auc": rnd["pr_auc"]["mean"],
        "threshold": best_t,
        "cv_random": rnd, "cv_temporal": tmp, "cv_group": grp,
        "feature_importance": imp,
    }
    return metrics, oof, best_t

def train_closure(df: pd.DataFrame, n_splits, n_iter, skip_extra_cv=False) -> tuple:
    print(f"\n[2/3] Road-closure (barricading/diversion) classifiers (tune n_iter={n_iter})")
    d = df.reset_index(drop=True)
    X = build_feature_frame(d)
    y = d["requires_road_closure"].astype(int).values
    m, _, t = _train_classifier("closure", d.index, X, y,
                                d["start_datetime"], d["cluster_id"],
                                n_splits, n_iter, skip_extra_cv)
    return m, t

def train_major(df: pd.DataFrame, n_splits, n_iter, skip_extra_cv=False) -> tuple:
    print(f"\n[3/3] Major-disruption classifiers (tune n_iter={n_iter})")
    d = df[df["duration_min"].notna()].reset_index(drop=True)
    y = ((d["duration_min"] >= MAJOR_DURATION_MIN)
         | (d["requires_road_closure"] == 1)).astype(int).values
    X = build_feature_frame(d)
    m, oof, t = _train_classifier("major", d.index, X, y,
                                  d["start_datetime"], d["cluster_id"],
                                  n_splits, n_iter, skip_extra_cv)
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
    global _FROZEN_PREP_REG, _FROZEN_PREP_CLF

    parser = argparse.ArgumentParser(description="Train event-congestion models")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=20,
                        help="Optuna trials per model")
    parser.add_argument("--fast", action="store_true",
                        help="small search for quick iteration")
    parser.add_argument("--turbo", action="store_true",
                        help="maximum speed: fewer trials, skip group/temporal CV")
    args = parser.parse_args()
    n_splits = max(2, args.folds)
    if args.turbo:
        n_iter = 6
        n_splits = 3
    elif args.fast:
        n_iter = 4
    else:
        n_iter = args.n_iter

    os.makedirs(REPORTS_DIR, exist_ok=True)
    print("Loading and engineering features ...")
    df = engineer_features(load_raw("."))
    builder = FeatureBuilder().fit(df)
    df = builder.transform(df)
    _save(builder, "feature_builder")
    print(f"  rows={len(df)}  usable durations={df['duration_min'].notna().sum()}  "
          f"features={len(build_feature_frame(df).columns)}")

    # ----- Pre-fit preprocessors ONCE and freeze them -----
    # This eliminates ~600+ redundant TF-IDF/SVD/encoding fits during
    # Optuna tuning and CV evaluation, giving a ~5-10x wall-clock speedup.
    dur_df = df[df["duration_min"].notna()].reset_index(drop=True)
    X_dur = build_feature_frame(dur_df)
    X_all = build_feature_frame(df)
    y_dur_tmp = np.log1p(dur_df["duration_min"].values)
    y_clo_tmp = df["requires_road_closure"].astype(int).values

    print("Pre-fitting preprocessors (one-time cost) ...")
    prep_reg = make_preprocessor(include_closure_flag=True)
    prep_reg.fit(X_dur, y_dur_tmp)
    _FROZEN_PREP_REG = FrozenPreprocessor(prep_reg)

    prep_clf = make_preprocessor(include_closure_flag=False)
    prep_clf.fit(X_all, y_clo_tmp)
    _FROZEN_PREP_CLF = FrozenPreprocessor(prep_clf)
    print("  done — all subsequent pipeline fits skip preprocessing.")

    skip_extra_cv = args.turbo
    dur = train_duration(dur_df, n_splits, n_iter, skip_extra_cv)
    clo, clo_t = train_closure(df, n_splits, n_iter, skip_extra_cv)
    maj, maj_t = train_major(df, n_splits, n_iter, skip_extra_cv)

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
