"""
Event-Driven Congestion Command Center  (Streamlit dashboard)

Run:  streamlit run app.py

Three tabs:
  * Live Operations    - replay the event feed in simulated real time: detect
                         congestion, forecast each event's impact with the ML
                         models, and dispatch an action order (manpower /
                         barricades / diversion) to the nearest police station.
  * Forecast & Deploy  - score a single planned/unplanned event and get an
                         operational plan (manpower / barricades / diversion).
  * Historical Insight - explore the patterns the models learned from.
"""

from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.congestion import STATUS_COLORS, corridor_congestion
from src.data_prep import EventInput, engineer_features, load_raw
from src.dispatch import PoliceStationDirectory, build_dispatch_order, forecast_events
from src.inference import Forecaster
from src.live_feed import LiveFeedSimulator, to_ist
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


@st.cache_resource(show_spinner=False)
def get_feed() -> LiveFeedSimulator:
    return LiveFeedSimulator(get_data())


@st.cache_resource(show_spinner=False)
def get_directory() -> PoliceStationDirectory:
    return PoliceStationDirectory(get_data())


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

tab_live, tab_forecast, tab_history = st.tabs(
    ["🚨 Live Operations", "🔮 Forecast & Deploy", "📊 Historical Insight"]
)

# ============================================================================
# TAB 0 — Live Operations (simulated real-time feed -> detect -> dispatch)
# ============================================================================

STATUS_ICONS = {"Severe": "🔴", "Heavy": "🟠", "Moderate": "🟡", "Light": "🟢"}
MONITOR_COLOR = "#95a5a6"  # active events detected before the shift started
MAX_FORECASTS_PER_TICK = 15

with tab_live:
    feed = get_feed()
    directory = get_directory()

    days = feed.busiest_days(8)
    day_options = list(days["date"])
    day_events = dict(zip(days["date"], days["events"]))
    default_day = int(days["events"].idxmax())

    def live_reset(day):
        st.session_state.live_day = day
        st.session_state.live_time = feed.day_start(day, hour=7)
        st.session_state.live_orders = []
        st.session_state.live_order_ids = set()

    def live_advance(minutes: int):
        t0 = st.session_state.live_time
        t1 = t0 + pd.Timedelta(minutes=minutes)
        new = feed.new_events(t0, t1)
        new = new[~new["id"].astype(str).isin(st.session_state.live_order_ids)]
        if len(new) > MAX_FORECASTS_PER_TICK:
            new = new.tail(MAX_FORECASTS_PER_TICK)
        forecasts = forecast_events(get_forecaster(), new)
        for (_, row), fc in zip(new.iterrows(), forecasts):
            order = build_dispatch_order(row, fc, directory, row["start_datetime"])
            st.session_state.live_orders.append(order.as_dict())
            st.session_state.live_order_ids.add(order.event_id)
        st.session_state.live_time = t1

    c_day, c_tick, c_step, c_auto, c_reset = st.columns([2.1, 1.2, 1.1, 1.4, 0.9])
    with c_day:
        sel_day = st.selectbox(
            "Replay day (historical feed)", day_options, index=default_day,
            format_func=lambda d: f"{d} — {day_events[d]} events",
        )
    if "live_day" not in st.session_state or st.session_state.live_day != sel_day:
        live_reset(sel_day)
    with c_tick:
        tick = st.selectbox("Clock tick", [15, 30, 60], index=1,
                            format_func=lambda m: f"{m} min")
    with c_step:
        st.write("")
        if st.button("⏭ Advance", width="stretch"):
            live_advance(tick)
    with c_auto:
        st.write("")
        auto = st.toggle("▶ Auto-play", help="Advance one tick every 2 seconds")
    with c_reset:
        st.write("")
        if st.button("↺ Reset", width="stretch"):
            live_reset(sel_day)

    sim_t = st.session_state.live_time
    active = feed.active_events(sim_t)
    orders = st.session_state.live_orders
    odf = pd.DataFrame(orders)
    congestion = corridor_congestion(active)

    active_ids = set(active["id"].astype(str))
    live_orders_active = [o for o in orders if o["event_id"] in active_ids]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🕒 Simulated time (IST)", to_ist(sim_t).strftime("%H:%M"),
              help=f"Replaying {st.session_state.live_day}")
    k2.metric("🚗 Active events", len(active))
    k3.metric("🛣️ Congested corridors",
              int((congestion["status"] != "Light").sum()),
              help="Corridors at Moderate/Heavy/Severe congestion")
    k4.metric("📨 Orders dispatched", len(orders))
    k5.metric("👮 Officers deployed",
              int(sum(o["manpower"] for o in live_orders_active)),
              help="Sum of recommended manpower on currently active events")

    map_col, cong_col = st.columns([2.1, 1])

    with map_col:
        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lat=directory.stations["latitude"], lon=directory.stations["longitude"],
            mode="markers", name="Police stations",
            marker=dict(size=8, color="#2c3e50", opacity=0.45),
            text=[f"🚓 {s} PS" for s in directory.stations["police_station"]],
            hoverinfo="text",
        ))
        if not active.empty:
            sev_by_id, score_by_id = {}, {}
            if not odf.empty:
                sev_by_id = dict(zip(odf["event_id"], odf["severity_band"]))
                score_by_id = dict(zip(odf["event_id"], odf["impact_score"]))
            amap = active.copy()
            amap["sev"] = amap["id"].astype(str).map(sev_by_id).fillna("Monitoring")
            amap["score"] = amap["id"].astype(str).map(score_by_id).fillna(20.0)
            for band, color in [("Monitoring", MONITOR_COLOR), *BAND_COLORS.items()]:
                sub = amap[amap["sev"] == band]
                if sub.empty:
                    continue
                fig.add_trace(go.Scattermapbox(
                    lat=sub["latitude"], lon=sub["longitude"],
                    mode="markers", name=band,
                    marker=dict(size=8 + sub["score"] / 8, color=color, opacity=0.85),
                    text=[f"{r.event_cause} · {r.corridor}" for r in sub.itertuples()],
                    hoverinfo="text",
                ))
        fig.update_layout(
            mapbox=dict(style="open-street-map",
                        center=dict(lat=12.9716, lon=77.5946), zoom=10.2),
            height=470, margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.01, x=0.01,
                        bgcolor="rgba(255,255,255,0.75)"),
        )
        st.plotly_chart(fig, width="stretch")

    with cong_col:
        st.markdown("##### 🛣️ Corridor congestion")
        if congestion.empty:
            st.info("No active events — roads are clear.")
        else:
            view = congestion.head(10).copy()
            view["status"] = [f"{STATUS_ICONS[s]} {s}" for s in view["status"]]
            view.columns = ["Corridor", "Index", "Status", "Events", "Closures"]
            st.dataframe(view, width="stretch", hide_index=True,
                         height=390)

    st.markdown("#### 📨 Dispatch orders — sent to the nearest police station")
    if not orders:
        st.info("No orders yet. Press **⏭ Advance** (or turn on **▶ Auto-play**) to "
                "run the live feed; every newly detected event is forecast by the "
                "ML models and dispatched automatically.")
    else:
        for i, o in enumerate(reversed(orders[-4:])):
            icon = STATUS_ICONS.get(
                {"Low": "Light", "Moderate": "Moderate", "High": "Heavy",
                 "Severe": "Severe"}[o["severity_band"]], "🟢")
            title = (f"{icon} {o['order_id']} → **{o['police_station']} PS** "
                     f"({o['distance_km']} km) — {o['severity_band']} impact, "
                     f"score {o['impact_score']}")
            with st.expander(title, expanded=(i == 0)):
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("👮 Manpower", f"{o['manpower']} officers")
                e2.metric("🚧 Barricades", o["barricades"] or "None")
                e3.metric("⏱ Expected duration",
                          f"{o['expected_duration_min']:.0f} min",
                          help=f"Worst-case (P90): {o['duration_p90_min']:.0f} min")
                e4.metric("↪️ Diversion",
                          "Required" if o["need_diversion"] else "No")
                st.markdown(
                    f"**Event:** {o['event_cause'].replace('_', ' ')} on "
                    f"**{o['corridor']}** ({o['zone']} zone) · reported "
                    f"{o['issued_at_ist']} IST · closure probability "
                    f"{o['closure_probability']*100:.0f}%"
                )
                if o["description"]:
                    st.caption(f"Operator note: “{o['description']}”")
                st.info(f"**Diversion plan:** {o['diversion_plan']}")
                st.caption("Key drivers: " + ", ".join(o["drivers"]))

        with st.expander(f"📜 Full dispatch log ({len(orders)} orders)"):
            log_cols = ["order_id", "issued_at_ist", "police_station", "distance_km",
                        "event_cause", "corridor", "severity_band", "impact_score",
                        "manpower", "barricades", "need_diversion",
                        "expected_duration_min", "status"]
            st.dataframe(odf[log_cols], width="stretch", hide_index=True)
            st.download_button(
                "⬇️ Download dispatch log (CSV)",
                odf[log_cols].to_csv(index=False).encode(),
                file_name=f"dispatch_log_{st.session_state.live_day}.csv",
                mime="text/csv",
            )

    if auto:
        time.sleep(2)
        live_advance(tick)
        st.rerun()

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
                           width="stretch")

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
                            width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        local_hour = ((df["start_datetime"] + pd.Timedelta(hours=5, minutes=30))
                      .dt.hour)
        hourly = local_hour.value_counts().sort_index().reset_index()
        hourly.columns = ["hour", "events"]
        fig = px.bar(hourly, x="hour", y="events",
                     title="Events by hour of day (IST)")
        st.plotly_chart(fig, width="stretch")

    with c4:
        top_corr = df["corridor"].value_counts().head(12).reset_index()
        top_corr.columns = ["corridor", "events"]
        fig = px.bar(top_corr, x="events", y="corridor", orientation="h",
                     title="Busiest corridors (event count)")
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Event hot-spots")
    sample = df.sample(min(3000, len(df)), random_state=1)
    fig = px.density_mapbox(
        sample, lat="latitude", lon="longitude", radius=8,
        center=dict(lat=12.9716, lon=77.5946), zoom=9.5,
        mapbox_style="open-street-map", height=520,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, width="stretch")

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
                col.plotly_chart(fig, width="stretch")
