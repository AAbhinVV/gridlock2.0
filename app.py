"""
Event-Driven Congestion Command Center  (Streamlit dashboard)

Run:  streamlit run app.py

Two tabs:
  * Forecast & Deploy  - score a single planned/unplanned event and get an
                         operational plan (manpower / barricades / diversion).
  * Historical Insight - explore the patterns the models learned from.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_prep import EventInput, engineer_features, load_raw
from src.inference import Forecaster
from src.train import FrozenPreprocessor  # Required to unpickle models correctly in Streamlit

st.set_page_config(
    page_title="Event-Driven Congestion Command Center",
    page_icon="🚦",
    layout="wide",
)

MODELS_DIR = "models"


# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_forecaster() -> Forecaster:
    return Forecaster(MODELS_DIR)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return engineer_features(load_raw("."))


@st.cache_data(show_spinner=False)
def get_choices(df: pd.DataFrame) -> dict:
    def top(col, n=30):
        return sorted(df[col].dropna().astype(str).value_counts().head(n).index.tolist())
    return {
        "event_type": ["unplanned", "planned"],
        "event_cause": top("event_cause", 30),
        "veh_type": top("veh_type", 15),
        "corridor": top("corridor", 30),
        "zone": top("zone", 15),
        "police_station": top("police_station", 60),
    }


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    path = os.path.join("reports", "metrics.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            return json.load(f)
    return {}


def models_ready() -> bool:
    # Check for the stacking ensemble models (current train.py output)
    # or fall back to legacy single-model names.
    required_base = ["reference_tables", "feature_builder"]
    model_names = ["duration_model", "duration_model_p90", "closure_model", "major_model"]
    for m in model_names:
        stack = os.path.join(MODELS_DIR, f"{m}_stack.joblib")
        legacy = os.path.join(MODELS_DIR, f"{m}.joblib")
        if not (os.path.exists(stack) or os.path.exists(legacy)):
            return False
    return all(
        os.path.exists(os.path.join(MODELS_DIR, f"{m}.joblib"))
        for m in required_base
    )


@st.cache_data(show_spinner=False)
def get_importance() -> dict:
    path = os.path.join("reports", "feature_importance.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            return json.load(f)
    return {}


# ----------------------------------------------------------------------------
# Visual helpers
# ----------------------------------------------------------------------------

BAND_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Severe": "#e74c3c",
}


def gauge(score: float, band: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " /100"},
            title={"text": f"Impact Score — {band}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": BAND_COLORS.get(band, "#3498db")},
                "steps": [
                    {"range": [0, 25], "color": "#eafaf1"},
                    {"range": [25, 50], "color": "#fef9e7"},
                    {"range": [50, 75], "color": "#fdebd0"},
                    {"range": [75, 100], "color": "#fadbd8"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

st.title("🚦 Event-Driven Congestion Command Center")
st.caption(
    "Forecast the traffic impact of planned & unplanned events and get "
    "data-driven manpower, barricading and diversion recommendations."
)

if not models_ready():
    st.error(
        "Models not found. Train them first:\n\n```\npython -m src.train\n```"
    )
    st.stop()

df = get_data()
choices = get_choices(df)
forecaster = get_forecaster()
metrics = get_metrics()

tab_forecast, tab_history = st.tabs(["🔮 Forecast & Deploy", "📊 Historical Insight"])

# ============================================================================
# TAB 1 — Forecast & Deploy
# ============================================================================
with tab_forecast:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Event details")
        event_type = st.selectbox("Event type", choices["event_type"])
        event_cause = st.selectbox(
            "Cause", choices["event_cause"],
            index=choices["event_cause"].index("vehicle_breakdown")
            if "vehicle_breakdown" in choices["event_cause"] else 0,
        )
        veh_type = st.selectbox("Vehicle type (if any)", choices["veh_type"])
        corridor = st.selectbox("Corridor / road", choices["corridor"])
        zone = st.selectbox("Zone", choices["zone"])
        col_d, col_t = st.columns(2)
        with col_d:
            date = st.date_input("Date")
        with col_t:
            time = st.time_input("Start time")
        description = st.text_area(
            "Incident description (optional, English/Kannada)",
            placeholder="e.g. BMTC bus broken down blocking one lane",
            height=70,
        )
        requires_closure = st.checkbox("Road closure already declared?", value=False)
        go_btn = st.button("Forecast impact & recommend", type="primary",
                           use_container_width=True)

    with right:
        if go_btn:
            start_dt = f"{date} {time}"
            event = EventInput(
                event_type=event_type,
                event_cause=event_cause,
                veh_type=veh_type,
                corridor=corridor,
                zone=zone,
                description=description,
                latitude=float(df["latitude"].median()),
                longitude=float(df["longitude"].median()),
                start_datetime=start_dt,
                requires_road_closure=int(requires_closure),
            )
            res = forecaster.predict(event)

            st.plotly_chart(gauge(res["impact_score"], res["severity_band"]),
                            use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Expected duration",
                      f"{res['expected_duration_min']:.0f} min",
                      help=f"Worst-case (P90): {res['duration_p90_min']:.0f} min")
            m2.metric("Road-closure probability",
                      f"{res['closure_probability']*100:.0f}%")
            m3.metric("Major-event probability",
                      f"{res['major_event_probability']*100:.0f}%")

            st.markdown("### 🧰 Recommended deployment")
            d1, d2, d3 = st.columns(3)
            d1.metric("👮 Manpower", f"{res['manpower']} officers")
            d2.metric("🚧 Barricades",
                      res["barricades"] if res["need_barricading"] else "None")
            d3.metric("↪️ Diversion", "Required" if res["need_diversion"] else "No")

            st.info(f"**Diversion plan:** {res['diversion_plan']}")
            st.success(f"**Summary:** {res['summary']}")
            st.caption("Key drivers: " + ", ".join(res["drivers"]))
        else:
            st.info("Set the event details on the left and click "
                    "**Forecast impact & recommend**.")
            if metrics:
                st.markdown("#### Model performance (held-out test set)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Duration MAE",
                          f"{metrics['duration']['median_ae_minutes']:.0f} min (median)")
                c2.metric("Closure model ROC-AUC",
                          f"{metrics['closure']['roc_auc']:.2f}")
                c3.metric("Major-event ROC-AUC",
                          f"{metrics['major']['roc_auc']:.2f}")

# ============================================================================
# TAB 2 — Historical Insight
# ============================================================================
with tab_history:
    st.subheader("What the models learned from history")
    valid = df[df["duration_min"].notna()]

    c1, c2 = st.columns(2)
    with c1:
        cause_dur = (
            valid.groupby("event_cause")["duration_min"].median()
            .sort_values(ascending=False).reset_index()
        )
        fig = px.bar(cause_dur, x="duration_min", y="event_cause",
                     orientation="h", title="Median impact duration by cause (min)",
                     labels={"duration_min": "minutes", "event_cause": ""})
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        clo = (
            df.groupby("event_cause")["requires_road_closure"].mean()
            .sort_values(ascending=False).reset_index()
        )
        clo["requires_road_closure"] *= 100
        fig = px.bar(clo, x="requires_road_closure", y="event_cause",
                     orientation="h", title="Road-closure rate by cause (%)",
                     labels={"requires_road_closure": "% needing closure",
                             "event_cause": ""})
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        local_hour = ((df["start_datetime"] + pd.Timedelta(hours=5, minutes=30))
                      .dt.hour)
        hourly = local_hour.value_counts().sort_index().reset_index()
        hourly.columns = ["hour", "events"]
        fig = px.bar(hourly, x="hour", y="events",
                     title="Events by hour of day (IST)")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        top_corr = df["corridor"].value_counts().head(12).reset_index()
        top_corr.columns = ["corridor", "events"]
        fig = px.bar(top_corr, x="events", y="corridor", orientation="h",
                     title="Busiest corridors (event count)")
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Event hot-spots")
    sample = df.sample(min(3000, len(df)), random_state=1)
    fig = px.density_mapbox(
        sample, lat="latitude", lon="longitude", radius=8,
        center=dict(lat=12.9716, lon=77.5946), zoom=9.5,
        mapbox_style="open-street-map", height=520,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    imp = get_importance()
    if imp:
        st.markdown("#### What drives each model (permutation importance)")
        cols = st.columns(3)
        titles = {"duration": "Duration", "closure": "Road closure",
                  "major": "Major disruption"}
        for col, (key, title) in zip(cols, titles.items()):
            if key in imp and imp[key]:
                d = pd.DataFrame(imp[key]).head(8)
                fig = px.bar(d, x="importance", y="feature", orientation="h",
                             title=title)
                fig.update_layout(height=340, yaxis={"categoryorder": "total ascending"},
                                  margin=dict(l=0, r=0, t=40, b=0))
                col.plotly_chart(fig, use_container_width=True)
