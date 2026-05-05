"""
VIDYUT — Page 1: Demand Forecasting
Multi-horizon ensemble prediction per 11kV feeder with CSV upload,
zone risk heatmap, load-shedding alerts, and Prophet decomposition.
"""

import sys
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.components.shared import (
    inject_css, render_sidebar_brand, render_sidebar_status,
    run_forecast, check_load_shedding, render_load_alert,
    get_sample_demand_csv, PLOTLY_LAYOUT, TEAL, AMBER, RED, GREEN, BLUE,
)
from src.config.feature_config import (
    TRANSFORMER_CAPACITY_MAP, DEFAULT_TRANSFORMER_KVA,
    LOAD_SHEDDING_RULES, BESCOM_ZONES,
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()
    st.markdown("---")

    st.markdown("**⚙️ CONFIGURATION**")
    transformer_kva = st.selectbox(
        "Transformer Rating",
        options=list(TRANSFORMER_CAPACITY_MAP.values()),
        index=2,                           # 100 kVA default
        format_func=lambda v: f"{v} kVA (~{int(v*0.85)} kW)",
    )
    zone_filter = st.selectbox("Zone Filter", ["All Zones"] + list(BESCOM_ZONES.keys()))
    st.markdown("---")
    render_sidebar_status()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>Ministry of Power | Load Forecasting Module</div>", unsafe_allow_html=True)
st.title("⚡ Demand Forecasting — 11kV Feeder Intelligence")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_forecast, tab_zone, tab_compare, tab_explain = st.tabs(
    ["📈 Forecast", "🗺️ Zone Heatmap", "🔄 Feeder Compare", "📖 Feature Guide"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:

    # CSV Upload Section
    with st.expander("📂 UPLOAD FEEDER DATA (CSV)", expanded=False):
        st.markdown("""<div class='feature-card'>
            <h4>Data-Driven Mode</h4>
            <p>Upload your feeder CSV. Required columns: <b>feeder_id, timestamp, demand_kw</b>.
            The forecast will use your historical data for accurate lag-based predictions.</p>
        </div>""", unsafe_allow_html=True)

        col_up, col_dl = st.columns([4, 1])
        uploaded = col_up.file_uploader("Upload Feeder Dataset (CSV)", type=["csv"], key="demand_csv")
        col_dl.markdown("<div style='margin-top:27px;'></div>", unsafe_allow_html=True)
        col_dl.download_button("⬇ Sample CSV", data=get_sample_demand_csv(),
                               file_name="bescom_feeder_sample.csv", mime="text/csv",
                               use_container_width=True)

        if uploaded:
            try:
                df_up   = pd.read_csv(uploaded)
                required = {"feeder_id", "timestamp", "demand_kw"}
                if not required.issubset(df_up.columns):
                    missing = required - set(df_up.columns)
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                    st.session_state.pop("uploaded_df", None)
                else:
                    df_up["timestamp"] = pd.to_datetime(df_up["timestamp"])
                    df_up["demand_kw"] = pd.to_numeric(df_up["demand_kw"], errors="coerce").fillna(0.0)
                    st.session_state["uploaded_df"] = df_up
                    feeders_in_csv = sorted(df_up["feeder_id"].unique().tolist())
                    st.session_state["csv_feeders"] = feeders_in_csv
                    st.success(f"✅ Loaded {len(df_up):,} rows · {len(feeders_in_csv)} feeder(s) detected")
                    st.dataframe(df_up.head(8), use_container_width=True)
            except Exception as e:
                st.error(f"Parse error: {e}")
                st.session_state.pop("uploaded_df", None)
        else:
            st.session_state.pop("uploaded_df", None)
            st.session_state.pop("csv_feeders", None)

    # Forecast Config
    with st.expander("⚙️ FORECAST PARAMETERS", expanded=True):
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        # Dynamic feeder dropdown if CSV uploaded
        csv_feeders = st.session_state.get("csv_feeders", [])
        if csv_feeders:
            feeder_id = c1.selectbox("Feeder ID (from CSV)", options=csv_feeders)
        else:
            feeder_id = c1.text_input("Feeder ID", value="FEEDER_001",
                                       help="Or upload a CSV to auto-populate")

        HORIZON_OPTIONS = {
            "12h": 12, "24h": 24, "48h": 48, "72h": 72,
            "7d": 168, "30d": 720,
        }
        horizon_label = c2.selectbox("Forecast Horizon", list(HORIZON_OPTIONS.keys()), index=1)
        horizon_hours = HORIZON_OPTIONS[horizon_label]

        gran_map = {"15 min": 15, "30 min": 30, "1 hr": 60}
        gran_label = c3.selectbox("Granularity", list(gran_map.keys()))
        gran_min   = gran_map[gran_label]

        run_btn = c4.button("▶ RUN FORECAST", type="primary", use_container_width=True)

    # Run Forecast
    if run_btn:
        hist_data = st.session_state.get("uploaded_df")
        if hist_data is not None and feeder_id in hist_data["feeder_id"].values:
            hist_for_feeder = hist_data[hist_data["feeder_id"] == feeder_id].copy()
        elif hist_data is not None:
            hist_for_feeder = hist_data.copy()
        else:
            hist_for_feeder = None

        with st.spinner("Running Vidyut Ensemble Forecast..."):
            summary, points, warn = run_forecast(
                feeder_id, horizon_hours, gran_min, hist_df=hist_for_feeder
            )
        st.session_state["forecast_result"]  = (summary, points, warn, feeder_id, horizon_hours, transformer_kva)

    # Show results if available
    if "forecast_result" in st.session_state:
        summary, points, warn, f_id, h_hrs, t_kva = st.session_state["forecast_result"]

        if warn:
            st.warning(warn)

        # Load-shedding alert
        alert_lv, load_pct, max_kw = check_load_shedding(summary["max_kw"], t_kva)
        if alert_lv != "NORMAL":
            render_load_alert(alert_lv, load_pct, max_kw)

        # Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Peak Demand",   f"{summary['max_kw']:.1f} kW")
        m2.metric("Mean Load",     f"{summary['mean_kw']:.1f} kW")
        m3.metric("Min Load",      f"{summary['min_kw']:.1f} kW")
        m4.metric("Total Energy",  f"{summary['total_kwh']:.0f} kWh")
        m5.metric("Capacity Used", f"{load_pct*100:.1f}%",
                  delta=f"{'⚠️ HIGH' if load_pct>0.8 else 'Normal'}")
        m6.metric("Transformer",   f"{t_kva} kVA")

        df = pd.DataFrame(points)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Main forecast chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat_ensemble"], name="Ensemble",
            line=dict(color=TEAL, width=3), fill="tozeroy",
            fillcolor="rgba(0,212,170,0.05)",
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat_prophet"], name="Prophet (40%)",
            line=dict(color=BLUE, width=1.5, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat_lgbm"], name="LightGBM (60%)",
            line=dict(color=AMBER, width=1.5, dash="dot"),
        ))

        # Capacity line
        cap_kw = t_kva * 0.85
        fig.add_hline(y=cap_kw, line_dash="dot", line_color=RED, opacity=0.6,
                      annotation_text=f"Transformer capacity ({cap_kw:.0f} kW)",
                      annotation_font_color=RED, annotation_font_size=10)

        # Peak marker
        peak_idx = df["yhat_ensemble"].idxmax()
        fig.add_trace(go.Scatter(
            x=[df.loc[peak_idx, "timestamp"]], y=[df.loc[peak_idx, "yhat_ensemble"]],
            mode="markers+text", name="Peak",
            marker=dict(color=RED, size=14, symbol="triangle-up",
                        line=dict(color="white", width=2)),
            text=[f"  ▲ {df.loc[peak_idx,'yhat_ensemble']:.0f} kW"],
            textposition="middle right", textfont=dict(color=RED, size=11),
        ))

        layout = {**PLOTLY_LAYOUT,
                  "title": f"Load Profile — {f_id} | {h_hrs}h Horizon",
                  "yaxis": dict(title="Demand (kW)", gridcolor="#1A2D45"),
                  "height": 420}
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

        # Peak hours table
        df_hourly = df.groupby(df["timestamp"].dt.hour)["yhat_ensemble"].mean().reset_index()
        df_hourly.columns = ["Hour", "Avg Demand (kW)"]
        top3 = df_hourly.nlargest(3, "Avg Demand (kW)")
        st.markdown(f"**🕐 Top-3 Peak Hours for {f_id}:**")
        c_pk1, c_pk2, c_pk3 = st.columns(3)
        for col_m, (_, row) in zip([c_pk1, c_pk2, c_pk3], top3.iterrows()):
            col_m.metric(f"{int(row['Hour']):02d}:00 IST", f"{row['Avg Demand (kW)']:.1f} kW")

        lf = summary["mean_kw"] / summary["max_kw"] * 100
        st.info(f"📊 Peak at **{df.loc[peak_idx,'timestamp'].strftime('%H:%M')}** · {summary['max_kw']:.1f} kW · Load Factor: **{lf:.1f}%**")

        # Export
        csv_export = df[["timestamp", "yhat_ensemble", "yhat_prophet", "yhat_lgbm"]].to_csv(index=False)
        st.download_button("⬇ Export Forecast CSV", csv_export,
                           f"forecast_{f_id}_{h_hrs}h.csv", "text/csv")

        # Feature explanation (collapsible)
        with st.expander("📖 HOW IT WORKS — Feature Reference", expanded=False):
            t1, t2, t3, t4 = st.tabs(["📅 Calendar", "📈 Lag & Rolling", "🌡️ Weather", "🤖 Architecture"])
            with t1:
                st.markdown("""
| Feature | Description | Why It Matters |
|---|---|---|
| `hour` | Hour of day (0–23) | Strong intra-day demand cycles |
| `dayofweek` | Mon=0…Sun=6 | Industrial vs residential pattern shift |
| `is_peak_hour` | 1 if 07:00–10:00 or 18:00–22:00 IST | BESCOM regulated tariff windows |
| `season` | 0=Winter 1=Spring 2=Summer 3=Monsoon | Bangalore 4-season AC load driver |
| `hour_sin/cos` | Cyclic encoding | Prevents midnight discontinuity |
                """)
            with t2:
                st.markdown("""
**Lag Features** — past demand at fixed 15-min period offsets:

| Feature | Time Back | Signal |
|---|---|---|
| `lag_4` | 1 hour | Recent level |
| `lag_96` | 24 hours | Same time yesterday (strong autocorrelation) |
| `lag_672` | 7 days | Same time last week |

**Rolling Stats** — `roll_mean_4` to `roll_mean_672`, `roll_std_4` to `roll_std_672`

> Upload your CSV so lags use real history, not the 500 kW baseline.
                """)
            with t3:
                st.markdown("""
| Feature | Source | Description |
|---|---|---|
| `T2M` | NASA POWER | 2m temperature (°C) — primary AC load driver |
| `RH2M` | NASA POWER | Relative humidity — raises AC compressor load |
| `ALLSKY_SFC_SW_DWN` | NASA POWER | Solar irradiance — rooftop solar proxy |
                """)
            with t4:
                st.code("""
Prophet (40%) + LightGBM (60%) → Weighted Ensemble
  ├─ Prophet: Fourier seasonality + Karnataka holidays
  ├─ LightGBM: 48 lag/weather features, quantile regression
  └─ Intervals: P10 / P90 quantile bounds
                """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ZONE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_zone:
    st.subheader("Zone-Level Risk Heatmap — 8 BESCOM Zones")
    st.caption("Simulated load factor and risk level per zone (real production: pulls from forecasting pipeline)")

    rng = np.random.default_rng(77)
    zone_data = []
    for zone, meta in BESCOM_ZONES.items():
        load_factor  = rng.uniform(0.55, 1.05)
        mean_demand  = rng.uniform(300, 900)
        max_demand   = mean_demand * rng.uniform(1.2, 1.6)
        risk         = ("HIGH" if load_factor > 0.90 else
                        "MEDIUM" if load_factor > 0.75 else "LOW")
        zone_data.append({
            "zone": zone, "lat": meta["lat"], "lon": meta["lon"],
            "load_factor": round(load_factor, 3),
            "mean_demand_kw": round(mean_demand, 1),
            "max_demand_kw": round(max_demand, 1),
            "risk": risk, "color": meta["color"],
        })
    df_zones = pd.DataFrame(zone_data)

    # Metrics row
    z1, z2, z3, z4 = st.columns(4)
    z1.metric("High-Risk Zones", len(df_zones[df_zones["risk"] == "HIGH"]))
    z2.metric("Avg Load Factor", f"{df_zones['load_factor'].mean():.1%}")
    z3.metric("Peak Zone Demand", f"{df_zones['max_demand_kw'].max():.0f} kW")
    z4.metric("Zones Monitored", len(df_zones))

    # Plotly bubble map
    fig_map = go.Figure()
    for _, row in df_zones.iterrows():
        risk_color = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}[row["risk"]]
        fig_map.add_trace(go.Scattermapbox(
            lat=[row["lat"]], lon=[row["lon"]],
            mode="markers+text",
            marker=dict(size=max(20, row["load_factor"] * 50),
                        color=risk_color, opacity=0.8),
            text=[f"  {row['zone']}"],
            textfont=dict(color="white", size=11),
            textposition="middle right",
            name=row["zone"],
            hovertemplate=(f"<b>{row['zone']}</b><br>"
                           f"Load Factor: {row['load_factor']:.1%}<br>"
                           f"Peak: {row['max_demand_kw']:.0f} kW<br>"
                           f"Risk: {row['risk']}<extra></extra>"),
        ))
    fig_map.update_layout(
        mapbox=dict(style="dark", center=dict(lat=12.97, lon=77.59), zoom=9.5),
        paper_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Risk table
    display = df_zones[["zone", "load_factor", "mean_demand_kw", "max_demand_kw", "risk"]].copy()
    display.columns = ["Zone", "Load Factor", "Mean kW", "Peak kW", "Risk"]
    display["Load Factor"] = display["Load Factor"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEEDER COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Side-by-Side Feeder Comparison")

    c_f1, c_f2, c_h = st.columns([2, 2, 1])
    fid1 = c_f1.text_input("Feeder A", value="FEEDER_001", key="cmp1")
    fid2 = c_f2.text_input("Feeder B", value="FEEDER_002", key="cmp2")
    cmp_horizon = c_h.selectbox("Horizon", [12, 24, 48], index=1, key="cmp_h")
    cmp_btn = st.button("🔄 COMPARE FEEDERS", type="primary")

    if cmp_btn:
        with st.spinner("Forecasting both feeders..."):
            s1, p1, _ = run_forecast(fid1, cmp_horizon)
            s2, p2, _ = run_forecast(fid2, cmp_horizon)

        df1 = pd.DataFrame(p1); df1["timestamp"] = pd.to_datetime(df1["timestamp"])
        df2 = pd.DataFrame(p2); df2["timestamp"] = pd.to_datetime(df2["timestamp"])

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=df1["timestamp"], y=df1["yhat_ensemble"],
                                     name=fid1, line=dict(color=TEAL, width=2.5)))
        fig_cmp.add_trace(go.Scatter(x=df2["timestamp"], y=df2["yhat_ensemble"],
                                     name=fid2, line=dict(color=AMBER, width=2.5)))
        layout_cmp = {**PLOTLY_LAYOUT, "title": f"Feeder Comparison — {cmp_horizon}h",
                      "height": 380}
        fig_cmp.update_layout(**layout_cmp)
        st.plotly_chart(fig_cmp, use_container_width=True)

        cols_m = st.columns(4)
        cols_m[0].metric(f"{fid1} Peak", f"{s1['max_kw']:.1f} kW")
        cols_m[1].metric(f"{fid1} Energy", f"{s1['total_kwh']:.0f} kWh")
        cols_m[2].metric(f"{fid2} Peak", f"{s2['max_kw']:.1f} kW")
        cols_m[3].metric(f"{fid2} Energy", f"{s2['total_kwh']:.0f} kWh")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE GUIDE (static)
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("Model Architecture & Feature Reference")
    st.markdown("""
**Vidyut Ensemble** = Prophet (40%) + LightGBM (60%) trained on BESCOM 15-minute interval data.

| Category | Features | Count |
|---|---|---|
| Calendar | hour, dayofweek, dayofmonth, dayofyear, weekofyear, month, quarter, is_weekend, is_holiday, is_peak_hour, season + cyclic encodings | 13 |
| Lag | lag_4, lag_8, lag_12, lag_24, lag_48, lag_96, lag_192, lag_672 | 8 |
| Rolling Mean | roll_mean_4/8/16/32/96/192/672 | 7 |
| Rolling Std | roll_std_4/8/16/32/96/192/672 | 7 |
| Weather | T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, PRECTOTCORR, ALLSKY_SFC_SW_DWN | 7 |
| **Total** | | **48** |

### Load-Shedding Alert Logic
| Alert | Load % | Action |
|---|---|---|
| 🟡 YELLOW | ≥ 80% | Warning — monitor closely |
| 🟠 ORANGE | ≥ 90% | High risk — prepare load shedding |
| 🔴 RED | ≥ 95% | Imminent — execute shedding in 15-30 min |
| ⚡ TRIP | ≥ 110% | Transformer auto-protection will activate |
    """)
