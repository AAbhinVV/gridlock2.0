# Knowledge Transfer: Event-Driven Congestion Forecasting & Deployment Planner

This document captures the **architecture, code organization, and key design decisions** for the project so a new maintainer can take over without re-discovering context from scratch.

---

## 1. What this project is

**Problem (hackathon brief):** Political rallies, festivals, sports events, construction, breakdowns, and sudden gatherings cause localized traffic breakdowns in Bengaluru. Today, impact is not quantified in advance, deployment is experience-driven, and there is no post-event learning loop.

**Solution:** An end-to-end system that:

1. **Predicts** impact duration, road-closure probability, and major-disruption probability from event attributes available at reporting time.
2. **Recommends** manpower, barricades, and diversion plans via a transparent rule engine.
3. **Surfaces** forecasts through a CLI, batch evaluation script, and Streamlit command-center dashboard.

**Data:** 8,173 anonymized Bengaluru traffic-event records from the *Astram* dataset (`Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv`).

---

## 2. High-level architecture

The system follows a **layered pipeline**: shared feature engineering → trained models → rule-based recommendations → multiple interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│  Interfaces                                                     │
│  app.py (Streamlit)  │  predict.py (CLI)  │  make_predictions.py│
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  src/inference.py — Forecaster                                  │
│  Loads models + FeatureBuilder; EventInput → full recommendation │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
┌──────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
│ src/recommend.py │ │  ML models      │ │ src/data_prep.py     │
│ Impact score +   │ │  (joblib in     │ │ engineer_features +  │
│ deployment rules │ │   models/)      │ │ FeatureBuilder       │
└──────────────────┘ └─────────────────┘ └──────────────────────┘
                               ▲
                               │
                    ┌──────────┴──────────┐
                    │  src/train.py       │
                    │  tune / validate /  │
                    │  calibrate / save   │
                    └─────────────────────┘
```

### Decision: Single source of truth for features (`src/data_prep.py`)

**What:** All training, batch scoring, CLI, and dashboard paths call the same `engineer_features`, `FeatureBuilder`, and `build_feature_frame` functions.

**Why:** Prevents train/serve skew — a common failure mode where the dashboard behaves differently from training because features were duplicated or diverged. The module docstring explicitly states this intent.

### Decision: Thin interfaces, fat shared core

**What:** `app.py`, `predict.py`, and `make_predictions.py` are entry points only. Business logic lives in `src/`.

**Why:** Keeps UI/CLI concerns separate from ML and recommendation logic. `Forecaster` in `src/inference.py` is the single inference path used by both CLI and dashboard.

### Decision: Joblib artifacts in `models/`, metrics in `reports/`

**What:** Trained pipelines, `FeatureBuilder`, and reference tables are serialized to `models/*.joblib`. Evaluation output goes to `reports/metrics.json` and `reports/feature_importance.json`.

**Why:** Simple, reproducible deployment for a hackathon/demo scope — no database or model registry required. Artifacts are versioned alongside code (retrain to refresh).

---

## 3. Data and target definitions

### Decision: Impact duration from resolution timestamps (not `end_datetime` alone)

**What:** Target `duration_min` = minutes from `start_datetime` to the first available of `resolved_datetime` → `closed_datetime` → `end_datetime`.

**Why:** `end_datetime` may reflect closure-stretch recording rather than when traffic impact actually ended. Using resolution/close times better approximates operational impact duration.

### Decision: Cap duration at 3 days (`MAX_DURATION_MIN = 4320`)

**What:** Rows outside `(0.5 min, 3 days]` get `duration_min = NaN` and are excluded from duration/major training.

**Why:** Some events are never formally closed in the data, producing implausible multi-day tails. Capping avoids training on bad labels while still allowing long events up to 3 days. Median error (not mean) is reported because the distribution is heavy-tailed.

### Decision: UTC → IST (+5:30) for all temporal features

**What:** Raw CSV timestamps are parsed as UTC; hour, day-of-week, peak-hour, and holiday features use IST.

**Why:** Operational decisions (peak hours, holidays) are local to Bengaluru. Peak windows are hard-coded as 08–11 and 17–21 IST.

### Decision: Coordinate cleaning with Bengaluru bounding box

**What:** Lat/lon outside `[12.6, 13.3] × [77.3, 77.9]` are nulled and imputed with dataset medians.

**Why:** Placeholder or out-of-city coordinates appear in raw data. Bounding-box filtering keeps spatial features meaningful without dropping rows.

---

## 4. Feature engineering decisions

Feature schema is centralized in constants at the top of `src/data_prep.py`.

| Feature group | Columns | Encoding | Rationale |
|---|---|---|---|
| Low-cardinality categoricals | `event_type`, `event_cause`, `veh_type`, `zone`, `is_corridor`, `long_lived_cause` | One-hot (`min_frequency=10`, `handle_unknown="ignore"`) | Stable cardinality; rare levels collapsed |
| High-cardinality categoricals | `corridor`, `police_station`, `junction`, `cluster_id` | Target encoding (`cv=3`) | Too many levels for one-hot; OOF encoding reduces leakage within folds |
| Free text | `description_text` | Char n-gram TF-IDF (3–4 grams) → TruncatedSVD (12 components) | Strongest signal across models; handles English + Kannada + transliteration |
| Keyword flags | 14 binary flags (`kw_accident`, `kw_block`, …) | Passthrough numeric | Interpretable text signals alongside dense TF-IDF |
| Temporal | `hour`, `dow`, `month`, `is_weekend`, `is_peak` | Passthrough | Captures time-of-day and weekly patterns |
| Holiday | `is_holiday`, `days_to_holiday` | Passthrough | Offline Karnataka/India holiday calendar — no external API at inference |
| Spatial | `latitude`, `longitude` | Passthrough | Raw coords + derived cluster |
| Density (fitted) | `corridor_density`, `junction_density`, `cluster_density` | Passthrough | Historical event frequency priors per location |
| Closure flag (conditional) | `requires_road_closure` | Included for duration model only | See below |

### Decision: `FeatureBuilder` for fitted spatial features only

**What:** `FeatureBuilder` fits KMeans (40 clusters) on lat/lon and counts events per cluster, corridor, and junction. Everything else in `engineer_features` is row-wise and deterministic.

**Why:** Cluster assignments and density counts must be learned from training data and frozen at inference. Splitting fitted vs. deterministic logic makes the train/test boundary explicit (`make_predictions.py` fits builder on train only).

### Decision: KMeans with 40 clusters (`N_CLUSTERS = 40`)

**What:** Unsupervised spatial buckets replace raw high-cardinality location IDs for `cluster_id`.

**Why:** Provides a geography signal that generalizes slightly better than junction names alone. Used as a **group** in GroupKFold validation to measure performance on unseen areas.

### Decision: Long-lived cause flag

**What:** `long_lived_cause = 1` for `{construction, water_logging, pot_holes, road_conditions, tree_fall}`.

**Why:** These causes structurally persist longer and often need different deployment posture than transient breakdowns.

### Decision: Text — dual representation (keywords + TF-IDF/SVD)

**What:** 14 hand-crafted keyword patterns (English + Kannada script) **plus** a char-level TF-IDF pipeline reduced to 12 SVD components.

**Why:** Keywords are explainable for stakeholders; TF-IDF/SVD captures phrasing not covered by rules. Together they drove the largest gain over the v1 baseline (see `reports/metrics.json` → `improvement`).

### Decision: Offline holiday calendar (`HOLIDAYS` set)

**What:** Hard-coded dates covering Nov 2023–Apr 2024 data window plus forward anchors through 2025.

**Why:** No network dependency at inference; `days_to_holiday` capped at 30 days gives a smooth proximity signal.

### Decision: Include `requires_road_closure` in duration model, exclude from classifiers

**What:** `make_preprocessor(include_closure_flag=True)` for regressors; `False` for closure/major classifiers.

**Why:** For **duration**, knowing a closure was declared is legitimate input at forecast time (operator may already know a rally will close a road). For **closure prediction**, including the target as a feature would be circular leakage.

---

## 5. Model architecture decisions

### Decision: Three models, not one multi-task model

| Artifact | Type | Target | Role |
|---|---|---|---|
| `duration_model` | Quantile regressor (P50) | `log1p(duration_min)` | Expected impact duration |
| `duration_model_p90` | Quantile regressor (P90) | same | Worst-case duration for risk-aware staffing |
| `closure_model` | Calibrated classifier | `requires_road_closure` | Barricade/diversion trigger |
| `major_model` | Calibrated classifier | `(duration ≥ 180 min) OR closure` | High-impact incident profile |

**Why:**

- Different targets have different label availability (only ~2,901 rows have usable duration).
- Quantile regression gives an uncertainty band without a separate probabilistic duration model.
- **Major disruption** replaces the raw `priority` field (see leakage section below).

### Decision: HistGradientBoosting (not XGBoost/LightGBM/neural)

**What:** `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` inside sklearn `Pipeline`.

**Why:** Strong tabular performance, native handling of mixed feature types after preprocessing, no extra dependencies, fast enough for hackathon iteration. `class_weight="balanced"` on classifiers addresses closure/major imbalance.

### Decision: Log-transform duration target

**What:** Train on `log1p(duration_min)`; invert with `expm1` at inference; clip to `[1, 4320]` minutes.

**Why:** Duration is right-skewed. Log space stabilizes regression and aligns with RMSLE as a secondary metric.

### Decision: Quantile loss for duration (P50 + P90)

**What:** Two separate models with `loss="quantile"`, `quantile=0.5` and `0.9`.

**Why:** Median prediction matches the evaluation metric (median AE). P90 supports conservative staffing when the P50→P90 gap is wide (`recommend.py` → `_manpower`).

### Decision: Isotonic calibration + tuned threshold for classifiers

**What:** After hyperparameter search, final classifiers wrapped in `CalibratedClassifierCV(method="isotonic", cv=3)`. Closure operating threshold tuned on OOF predictions to maximize F1; stored in `reference_tables.joblib` → `thresholds.closure`.

**Why:** Raw HGB probabilities may be miscalibrated. Isotonic calibration improves probability quality for the rule engine. F1-based threshold balances precision/recall for operational alerts (threshold is consumed by `recommend()` for barricade/diversion decisions).

### Decision: Single sklearn Pipeline per model

**What:** Each model = `ColumnTransformer` (prep) + estimator, saved as one joblib object.

**Why:** Preprocessing is refit automatically on each train call; no manual encoder state management beyond `FeatureBuilder`.

---

## 6. Validation and evaluation decisions

### Decision: Three validation schemes (not random CV alone)

| Scheme | Implementation | What it tests |
|---|---|---|
| Random k-fold | `KFold` / stratified for classifiers | Overall fit quality |
| Temporal holdout | Train first 80% by `start_datetime`, test last 20% | Generalization to future months |
| GroupKFold by `cluster_id` | Location clusters held out | Generalization to unseen geographic areas |

**Why:** Random CV can be optimistic for time-series and spatial data. Reporting all three (in `reports/metrics.json`) gives an honest picture. Group CV shows closure ROC-AUC dropping ~0.82 → ~0.77 on new areas — documented as a known limitation.

### Decision: `make_predictions.py` uses a separate honest holdout

**What:** 80/20 stratified split on `requires_road_closure`; `FeatureBuilder` and models fit on train only; writes `predictions.csv`.

**Why:** Full `src/train.py` refits on all data for production artifacts. `make_predictions.py` provides an unbiased test-set scorecard and export for stakeholders without nested-CV complexity.

**Note:** `make_predictions.py` uses default hyperparameters (no RandomizedSearchCV) for speed — metrics may differ slightly from fully tuned `src/train.py` CV numbers.

### Decision: Permutation feature importance (not tree gain alone)

**What:** Top-15 permutation importances saved per model to `reports/feature_importance.json`; displayed in dashboard Historical Insight tab.

**Why:** Permutation importance is model-agnostic and more trustworthy for high-dimensional encoded features. Supports explainability narrative for traffic-control stakeholders.

### Decision: Preserve v1 baseline in metrics (`BASELINE_V1`)

**What:** Hard-coded prior metrics in `train.py`; `improvement` block in `metrics.json`.

**Why:** Documents iteration story (duration medAE 35→32 min, closure AUC 0.79→0.82, major AUC 0.88→0.89) for presentations and regression checks.

---

## 7. Data integrity — leakage traps (critical)

These are **intentional exclusions** documented in README and code comments.

### Trap 1: `priority` field

**Observation:** `priority == "High"` iff corridor ≠ `"Non-corridor"` (>99% deterministic).

**Decision:** Do **not** predict `priority`. Instead, define **major disruption** as `(duration ≥ 3 h) OR requires_road_closure`.

**Why:** Predicting `priority` yields fake ROC-AUC ≈ 1.0 — it encodes the corridor rule, not learned impact. Major disruption is a substantively meaningful operational label.

### Trap 2: End coordinates / stretch length

**Observation:** `endlatitude` / `endlongitude` are populated only after a closure stretch is recorded.

**Decision:** End coordinates and derived stretch-length features are **never used**.

**Why:** They leak the closure target (~98% closure when present) and are unknown at forecast time. Including them inflated closure AUC to a fake ~0.98; honest score is ~0.82.

### Trap 3: Target encoding without CV (avoided)

**Decision:** Use sklearn `TargetEncoder(cv=3)` inside the pipeline.

**Why:** Naive target encoding on the full training fold would leak label information into features.

---

## 8. Recommendation engine decisions (`src/recommend.py`)

### Decision: ML predictions + transparent rules (hybrid)

**What:** Models produce duration, closure prob, major prob. `recommend()` maps them to manpower, barricades, diversion via fixed rules.

**Why:** Traffic command centers need **explainable** deployment plans. Pure ML-to-resources mapping would be a black box. Rules are tunable with SOPs; ML inputs update when models retrain (post-event learning loop).

### Decision: Impact Score formula

```
dur_component = log1p(duration) / log1p(1440)   # saturates ~24h
base = 0.45 * dur_c + 0.30 * closure_prob + 0.25 * major_prob
score = min(100, base * 100 * corridor_importance * peak_multiplier)
```

**Weights rationale:** Duration weighted highest (45%) because time-on-ground drives staffing cost; closure (30%) triggers physical infrastructure; major prob (25%) captures composite severity.

**Corridor importance:** Named arterial corridors ×1.25; `Non-corridor` ×0.8; `Unknown` ×1.0 — reflects ripple effect on network traffic.

**Peak multiplier:** ×1.15 during peak hours — more congestion amplification.

### Decision: Severity bands (0–100)

| Band | Range |
|---|---|
| Low | < 25 |
| Moderate | 25–49 |
| High | 50–74 |
| Severe | ≥ 75 |

**Why:** Gives operators a quick triage label alongside numeric score.

### Decision: Risk-aware manpower surge

**What:** Extra +2 officers if `duration_p90 > 2.5 × duration_p50` and `duration_p90 > 180` min.

**Why:** Wide P50–P90 gap signals high uncertainty; staff for worst case without always using P90 as the point estimate.

### Decision: Barricade and diversion rules use tuned closure threshold

**What:** `need_barricading` if `closure_prob ≥ closure_threshold` OR `score ≥ 60`. Diversion if `closure_prob ≥ threshold` OR (`score ≥ 65` AND `duration > 90` min).

**Why:** Threshold comes from data-driven F1 tuning; score-based fallback catches high-impact events below probability threshold.

### Decision: Corridor-aware diversion text (template, not routing API)

**What:** `_diversion_plan()` returns human-readable guidance differing for arterial vs. local streets.

**Why:** No live routing/graph data in scope. Templates are actionable placeholders until integrated with a real traffic management system.

---

## 9. Interface decisions

### Streamlit dashboard (`app.py`)

| Decision | Reason |
|---|---|
| Two tabs: Forecast & Deploy + Historical Insight | Separates operational workflow from exploratory analytics |
| `@st.cache_resource` for `Forecaster`, `@st.cache_data` for data/metrics | Avoids reloading joblib models and CSV on every widget interaction |
| Median lat/lon for dashboard forecasts when user doesn't pick a map point | Simplifies UX; corridor/zone/cause carry most signal. CLI allows explicit `--lat/--lon` |
| Plotly for gauges, bars, density map | Interactive charts without custom JS; OSM map for event hot-spots |
| Gate on `models_ready()` | Clear error if user skips training step |

### CLI (`predict.py`)

| Decision | Reason |
|---|---|
| `--json` flag for machine-readable output | Enables scripting and integration tests |
| Default sample event when run with no args | Quick smoke test after training |
| IST time string in `--time` | Matches operator mental model; converted to UTC internally in `EventInput` |

### Batch export (`make_predictions.py`)

| Decision | Reason |
|---|---|
| Output `predictions.csv` with actuals + preds + recommendations | Single artifact for demo, error analysis, and stakeholder review |
| Stratify split on `requires_road_closure` | Preserves class balance in test set |

---

## 10. Dependencies and runtime

**Stack:** Python 3.x, pandas, numpy, scikit-learn ≥1.4, joblib, plotly, streamlit.

**Decision:** Minimal dependency footprint (no XGBoost, no deep learning, no database).

**Why:** Reproducible setup for hackathon judges and quick `pip install -r requirements.txt` onboarding.

### Standard workflows

```bash
# Train production models (writes models/ and reports/)
python -m src.train

# Quick training iteration
python -m src.train --fast

# Single event forecast
python predict.py --cause vehicle_breakdown --corridor "Hosur Road" --time "2024-09-10 18:30"

# Held-out test predictions
python make_predictions.py

# Dashboard
streamlit run app.py
```

---

## 11. Known limitations (by design or data)

| Limitation | Implication | Possible next step |
|---|---|---|
| ~2,901 / 8,173 rows have usable duration | Duration/major models train on subset | Better close-out logging in source system |
| Group CV weaker on unseen locations | Closure AUC ~0.77 on new clusters | More geographic coverage; hierarchical spatial models |
| Heavy duration tail / censored events | 3-day cap + P90 heuristic | Survival modeling for uncensored times |
| Rule thresholds not calibrated to real SOPs | Defaults in `recommend.py` | Workshop with traffic police to tune weights |
| No real-time feeds | Event-level batch forecasting only | Weather, speed/volume, official event calendars |
| Dashboard uses median coordinates | Less precise spatial forecast | Map picker for lat/lon |
| Holiday calendar static | New holidays need manual update | Calendar API or annual refresh |

---

## 12. File map (what to read first)

| File | Responsibility |
|---|---|
| `src/data_prep.py` | **Start here.** Feature schema, cleaning, `FeatureBuilder`, `EventInput`, preprocessors |
| `src/train.py` | Training loops, CV, calibration, artifact saving |
| `src/recommend.py` | Impact score and deployment rules |
| `src/inference.py` | `Forecaster` — production inference path |
| `app.py` | Streamlit UI |
| `predict.py` | CLI wrapper |
| `make_predictions.py` | Honest holdout evaluation + CSV export |
| `models/` | Serialized models (not in git unless committed after train) |
| `reports/metrics.json` | Full CV numbers, best hyperparameters, baseline comparison |
| `reports/feature_importance.json` | Top features per model |
| `README.md` | User-facing overview and quick start |

---

## 13. Retraining and post-event learning

The intended learning loop:

1. New Astram events arrive (CSV append or replace).
2. Run `python -m src.train` — refits `FeatureBuilder`, retrains all models, updates thresholds and reference tables.
3. Deploy updated `models/` artifacts.
4. Historical Insight tab and reference tables (`median_duration_by_cause`, `closure_rate_by_cause`) refresh automatically from new data.

**Important:** Any new feature must be added in `data_prep.py` only, then flow through `build_feature_frame` and `make_preprocessor` so train and inference stay aligned.

---

## 14. Decision log (quick reference)

| # | Decision | Reason |
|---|---|---|
| 1 | Shared `data_prep.py` for train and inference | Prevent train/serve skew |
| 2 | `FeatureBuilder` for cluster + density only | Clear fit/transform boundary |
| 3 | Exclude end-coordinates | Label leakage + unavailable at forecast time |
| 4 | Replace `priority` with `major` label | Deterministic corridor rule, not predictive |
| 5 | HistGradientBoosting in sklearn Pipeline | Strong tabular baseline, minimal deps |
| 6 | Log duration target | Handle skew; use median AE |
| 7 | P50 + P90 quantile models | Point estimate + risk band |
| 8 | Isotonic calibration + F1 threshold | Reliable probabilities for rules |
| 9 | Triple validation (random / temporal / group) | Honest generalization metrics |
| 10 | TF-IDF + keyword flags on description | Best accuracy + some interpretability |
| 11 | Target encoding for high-cardinality cats | Practical alternative to huge one-hot |
| 12 | Hybrid ML + rule recommendation engine | Explainable ops plans for adoption |
| 13 | Corridor/peak multipliers in impact score | Network effects and congestion context |
| 14 | `Forecaster` class shared by CLI and dashboard | One inference code path |
| 15 | Joblib artifacts, no DB | Simplicity for hackathon / demo deployment |

---

*Last updated from codebase review — June 2025. For metric numbers after retraining, always prefer `reports/metrics.json` over static values in this document.*
