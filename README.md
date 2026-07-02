# 🚦 Event-Driven Congestion Forecasting & Deployment Planner

**Problem statement (hackathon):** Political rallies, festivals, sports events,
construction, breakdowns and sudden gatherings create localized traffic
breakdowns. Today their impact is *not quantified in advance*, resource
deployment is *experience-driven*, and there is *no post-event learning*.

> **How can historical and real-time data be used to forecast event-related
> traffic impact and recommend optimal manpower, barricading and diversion
> plans?**

This project answers that end-to-end with machine learning trained on **8,173
real Bengaluru traffic-event records** (the anonymized *Astram* dataset), plus
a transparent recommendation engine and an interactive command-center
dashboard.

---

## What it does

For any planned or unplanned event — described by its **cause, location
(corridor/zone), vehicle type, and time** — the system predicts:

| Question | Model | Output |
|---|---|---|
| **How long** will it affect traffic? | Impact-duration regressor | minutes |
| **Will it need a road closure** (barricades/diversion)? | Closure classifier | probability |
| **Will it become a high-impact incident?** | Major-disruption classifier | probability |

These three signals are blended into a **0–100 Impact Score**, which a
rule-based **recommendation engine** turns into an operational plan:

- 👮 **Manpower** — number of officers to deploy
- 🚧 **Barricading** — whether and how many barricades
- ↪️ **Diversion** — whether to divert, and a concrete corridor-aware plan

The recommendation logic is rule-based *on top of* learned predictions, so it
stays **explainable to traffic-control decision makers** while its inputs are
**learned from history** — directly enabling the "post-event learning system"
the brief asks for (retrain on new events → priors update).

### 🚨 Live Operations (the full loop, end to end)

The dashboard's first tab runs the whole pipeline in (simulated) real time:

1. **Track** — a live feed of traffic events plays against a clock
   (`src/live_feed.py` replays the historical stream; swap it for a real
   API/CAD feed in production).
2. **Detect** — active events are aggregated into a per-corridor congestion
   index with Light/Moderate/Heavy/Severe status (`src/congestion.py`).
3. **Forecast** — every newly detected event is scored by the ML models
   (duration P50/P90, closure probability, major-event probability).
4. **Dispatch** — a complete action order (manpower, barricades, diversion
   plan) is routed to the **nearest police station** by haversine distance
   over a 54-station directory learned from the data itself
   (`src/dispatch.py`), with a live map, alert cards and an exportable
   dispatch log.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (writes models/ and reports/metrics.json)
python -m src.train

# 3a. Score a single event from the command line
python predict.py --cause vehicle_breakdown --veh bmtc_bus \
    --corridor "Hosur Road" --time "2024-09-10 18:30"

python predict.py --type planned --cause public_event \
    --corridor "CBD 2" --closure 1 --time "2024-09-15 17:00"

# 3b. Held-out test: train on a train split, predict on unseen test rows,
#     and write every test event's forecast + recommendation to predictions.csv
python make_predictions.py

# 3c. …or launch the interactive dashboard (opens on the Live Operations tab)
streamlit run app.py

# Smoke-test the live pipeline from the CLI
python -m src.live_feed   # feed simulator + busiest replay days
python -m src.dispatch    # station directory + sample dispatch orders
```

> Note: the committed `models/` were pickled under **NumPy 2.x** — use
> `numpy>=2.0` (as pinned in `requirements.txt`) or retrain locally.

### `predictions.csv`
`make_predictions.py` performs an honest held-out evaluation (the feature
builder *and* the models are fit on the train split only, never seeing the test
rows) and writes one row per test event with: the **actual** outcome (duration,
closure, major), the **predicted** values (`pred_duration_p50_min`,
`pred_duration_p90_min`, `pred_closure_prob`, `pred_major_prob`), and the
**recommendation** (`impact_score`, `severity_band`, `rec_manpower`,
`rec_barricades`, `need_diversion`). It also prints a held-out scorecard
(duration medAE ≈ 32 min, closure ROC-AUC ≈ 0.82, major ROC-AUC ≈ 0.87).

---

## How it works

### 1. Data & feature engineering (`src/data_prep.py`)
- Parses timestamps (UTC → **IST**) and derives the **impact duration** target
  from `start_datetime` → first available of
  `resolved_datetime` / `closed_datetime` / `end_datetime`.
- Cleans placeholder/out-of-city coordinates to a Bengaluru bounding box.
- Rich, multi-source feature set:
  - **Low-cardinality categoricals** (one-hot): event type, cause, vehicle
    type, zone, on-corridor flag, long-lived-cause flag.
  - **High-cardinality categoricals** (out-of-fold **target encoding**):
    corridor, police station, junction, and a **KMeans location cluster**.
  - **Free-text `description`** (English + Kannada + transliteration): 14
    interpretable keyword flags + a char n-gram **TF-IDF → TruncatedSVD** block.
    This is the single strongest signal across all three models.
  - **Temporal:** hour, day-of-week, month, weekend, **peak-hour**, plus an
    offline **Karnataka/India holiday calendar** (`is_holiday`,
    `days_to_holiday`).
  - **Spatial / historical density:** per-corridor, per-junction and
    per-cluster event counts (a learned `FeatureBuilder` fits these on train).

### 2. Models (`src/train.py`)
`scikit-learn` **HistGradientBoosting** estimators in a single `Pipeline`
(one-hot + target-encode + TF-IDF/SVD + passthrough). For each model we
**tune hyper-parameters** (`RandomizedSearchCV`), **calibrate** classifier
probabilities (isotonic) and **tune the operating threshold**, then refit on
all data. Validation uses **three schemes** — random k-fold, a **temporal
holdout** (train earlier months, test later), and a **GroupKFold by location
cluster** — so generalisation to new times/places is measured honestly.

| Model | Target | Random 5-fold | Temporal | Group (new areas) |
|---|---|---|---|---|
| `duration_model` (P50, quantile) | impact duration | medAE ≈ **32 min** | ≈ 46 min | ≈ 36 min |
| `closure_model` (calibrated) | `requires_road_closure` | **ROC-AUC ≈ 0.82** | ≈ 0.85 | ≈ 0.77 |
| `major_model` (calibrated) | major = (dur ≥ 3 h) OR closure | **ROC-AUC ≈ 0.89** | ≈ 0.87 | ≈ 0.85 |

A second **P90 quantile** duration model (`duration_model_p90`) gives a
worst-case estimate for risk-aware staffing. Full per-fold numbers, tuned
params, and **permutation feature importances** are written to
`reports/metrics.json` and `reports/feature_importance.json`. Tune the run with
`python -m src.train --folds 5 --n-iter 12` (or `--fast` for quick iteration).

**Improvement over the v1 baseline** (recorded in `metrics.json`): duration
median error 35 → ~32 min, closure ROC-AUC 0.79 → 0.82, major ROC-AUC
0.88 → 0.89 — driven mostly by the text, spatial and density features.

### 3. Recommendation engine (`src/recommend.py`)
Blends the predictions with operational context (corridor importance, peak
hour) into an Impact Score, then maps it to manpower / barricades / diversion
via transparent, tunable rules. It is **risk-aware** — a wide P50→P90 duration
band adds a manpower surge — and uses the model's **tuned closure threshold**.
Every recommendation lists the **key drivers** behind it.

### 4. Live pipeline (`src/live_feed.py`, `src/congestion.py`, `src/dispatch.py`)
- **Feed simulator** replays the historical event stream against a simulated
  clock (`new_events` / `active_events` per tick) — the swap point for a real
  feed.
- **Congestion detector** turns the active-event set into a per-corridor
  index (closure/peak/arterial weighted) and traffic status, instantly on
  every tick.
- **Dispatch engine** builds a police-station directory from the data (median
  location of each station's 8k+ historical events), finds the nearest
  station by haversine distance, and issues a `DispatchOrder` combining the
  ML forecast with the sized action plan. A batch scorer (`forecast_events`)
  keeps each tick to a single model call.

### 5. Interfaces
- **`predict.py`** — CLI for a single event (human-readable or `--json`),
  including a P50–P90 duration interval and optional `--desc` free text.
- **`app.py`** — Streamlit "Command Center" with a *Live Operations* tab
  (simulated real-time detect → forecast → dispatch, live map, dispatch log),
  a *Forecast & Deploy* tab and a *Historical Insight* tab (duration/closure
  by cause, hourly pattern, busiest corridors, an event hot-spot map, and
  **permutation-importance** charts explaining each model).

---

## Data integrity (two leakage traps we caught)

1. **The `priority` field is a deterministic rule.** Every event on a named
   corridor is `High`, every `Non-corridor` event is `Low` (>99% of rows). A
   model "predicting" it scores a perfect ROC-AUC of 1.000 — leakage, not
   forecasting. We therefore model the genuinely informative **major-disruption**
   label instead.
2. **End-coordinates leak the closure target.** `endlatitude/endlongitude` are
   only filled once a closure *stretch* is recorded, so a derived "stretch
   length" feature implied closure 98% of the time (and is unknown at forecast
   time). Including it inflated closure ROC-AUC to a fake 0.98; we removed it,
   bringing the score back to an honest ~0.82.

Both are intentional, documented decisions — the models are trained only on
signals actually available *before* an event is resolved.

---

## Project structure

```
.
├── app.py                    # Streamlit command-center dashboard (3 tabs)
├── predict.py                # CLI single-event forecaster
├── make_predictions.py       # held-out test -> predictions.csv
├── predictions.csv           # generated test-set predictions + recommendations
├── requirements.txt
├── README.md
├── src/
│   ├── data_prep.py          # cleaning, feature engineering, FeatureBuilder (shared)
│   ├── train.py              # tunes/calibrates/validates + trains all models
│   ├── recommend.py          # impact score → manpower/barricade/diversion plan
│   ├── inference.py          # loads models + builder, EventInput → recommendation
│   ├── live_feed.py          # simulated real-time event feed (replay of history)
│   ├── congestion.py         # per-corridor congestion index + traffic status
│   └── dispatch.py           # nearest-police-station routing + dispatch orders
├── models/                   # saved .joblib models + feature_builder + reference tables
└── reports/
    ├── metrics.json          # CV/temporal/group metrics + before/after + best params
    └── feature_importance.json
```

## Limitations & next steps
- Duration has a heavy tail (some records are never formally "closed"); we cap
  at 3 days, report **median** error, and provide a P90 worst-case. More
  reliable close-out logging would sharpen this further.
- Group-CV shows performance dips on **unseen locations** (e.g. closure
  ROC-AUC 0.82 → 0.77) — more geographic coverage would help generalisation.
- Adding live feeds (weather, real-time speed/volume, official event calendars)
  would turn this from event-level into continuous corridor forecasting.
- Recommendation thresholds are sensible defaults meant to be calibrated with
  traffic-police SOPs.
