"""VIDYUT — Page 1: Demand Forecasting (Clean, no emojis, interactive graph)"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk
from datetime import datetime, timedelta, timezone

_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.components.shared import (
    inject_css, run_forecast, check_load_shedding, render_load_alert,
    get_sample_demand_csv, PLOTLY_LAYOUT, TEAL, AMBER, RED, GREEN, BLUE,
)
from src.config.feature_config import (
    TRANSFORMER_CAPACITY_MAP, LOAD_SHEDDING_RULES, BESCOM_ZONES,
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**CONFIGURATION**")
    transformer_kva = st.selectbox("Transformer Rating",
        options=list(TRANSFORMER_CAPACITY_MAP.values()), index=2,
        format_func=lambda v: f"{v} kVA (~{int(v*0.85)} kW)")
    zone_filter = st.selectbox("Zone Filter", ["All Zones"] + list(BESCOM_ZONES.keys()))
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#7B8FAB;'><span style='color:#00C48C;'>&#9679;</span> ENGINE ONLINE<br><span style='color:#00C48C;'>&#9679;</span> ML MODELS READY<br><span style='color:#00C48C;'>&#9679;</span> WEATHER: LIVE</div>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>MINISTRY OF POWER | LOAD FORECASTING MODULE</div>", unsafe_allow_html=True)
st.title("Demand Forecasting — 11kV Feeder Intelligence")

tab_forecast, tab_zone, tab_compare, tab_guide = st.tabs(
    ["Forecast", "Zone Heatmap", "Feeder Compare", "Feature Guide"])

# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    # CSV Upload
    with st.expander("UPLOAD FEEDER DATA (CSV)", expanded=False):
        st.markdown(
            "<div class='feature-card'>"
            "<div class='feature-card-title'>DATA-DRIVEN MODE</div>"
            "<p>Upload your feeder CSV. Required columns: "
            "<code>feeder_id</code>, <code>timestamp</code>, <code>demand_kw</code>. "
            "The forecast uses your historical data for lag-based predictions.</p>"
            "</div>", unsafe_allow_html=True)

        col_up, col_dl = st.columns([4,1])
        uploaded = col_up.file_uploader(
            "Upload Feeder Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"], key="demand_csv")
        col_dl.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        col_dl.download_button(
            "Download Sample CSV", data=get_sample_demand_csv(),
            file_name="bescom_feeder_sample.csv", mime="text/csv",
            use_container_width=True)

        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = pd.read_excel(uploaded)
                required = {"feeder_id","timestamp","demand_kw"}
                missing  = required - set(df_up.columns)
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                    st.info("Expected: feeder_id, timestamp, demand_kw")
                else:
                    df_up["timestamp"]  = pd.to_datetime(df_up["timestamp"])
                    df_up["demand_kw"]  = pd.to_numeric(df_up["demand_kw"], errors="coerce").fillna(0.0)
                    st.session_state["uploaded_df"]  = df_up
                    st.session_state["csv_feeders"]  = sorted(df_up["feeder_id"].unique().tolist())
                    # Auto-run forecast for first feeder if not already done
                    if "forecast_result" not in st.session_state and len(st.session_state["csv_feeders"]) > 0:
                        with st.spinner("Running forecast with uploaded data..."):
                            hist_for_feeder = df_up[df_up["feeder_id"]==st.session_state["csv_feeders"][0]].copy()
                            summary, points, warn = run_forecast(st.session_state["csv_feeders"][0], 24, 15, hist_df=hist_for_feeder)
                            st.session_state["forecast_result"] = (summary, points, st.session_state["csv_feeders"][0], 24, 100)
            except Exception as e:
                st.error(f"File parse error: {e}")

        if "uploaded_df" in st.session_state:
            df_up = st.session_state["uploaded_df"]
            st.success(f"Loaded {len(df_up):,} rows across {df_up['feeder_id'].nunique()} feeder(s).")
            with st.expander("Preview Uploaded Data", expanded=True):
                st.dataframe(df_up.head(5), use_container_width=True)
        else:
            st.info("Upload a feeder dataset (CSV/Excel) to use custom data for forecasting.")

    # Forecast Config
    st.markdown("**FORECAST PARAMETERS**")
    with st.container():
        c1, c2, c3, c4 = st.columns([3,2,2,2])
        csv_feeders = st.session_state.get("csv_feeders", [])
        if csv_feeders:
            feeder_id = c1.selectbox("Feeder ID (from CSV)", options=csv_feeders)
        else:
            feeder_id = c1.text_input("Feeder ID", value="FEEDER_001")

        HORIZON_OPTIONS = {"12h":12,"24h":24,"48h":48,"72h":72,"7d":168,"30d":720}
        horizon_label = c2.selectbox("Forecast Horizon", list(HORIZON_OPTIONS.keys()), index=1)
        horizon_hours = HORIZON_OPTIONS[horizon_label]

        gran_map   = {"15 min":15,"30 min":30,"1 hr":60}
        gran_label = c3.selectbox("Granularity", list(gran_map.keys()))
        gran_min   = gran_map[gran_label]

        c4.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        run_btn = c4.button("RUN FORECAST", type="primary", use_container_width=True)

    if run_btn or "forecast_result" not in st.session_state:
        hist_data = st.session_state.get("uploaded_df")
        hist_for_feeder = None
        if hist_data is not None:
            if feeder_id in hist_data["feeder_id"].values:
                hist_for_feeder = hist_data[hist_data["feeder_id"]==feeder_id].copy()
            else:
                hist_for_feeder = hist_data.copy()
        with st.spinner("Running Vidyut Ensemble Forecast..."):
            summary, points, warn = run_forecast(feeder_id, horizon_hours, gran_min, hist_df=hist_for_feeder)
        st.session_state["forecast_result"] = (summary, points, feeder_id, horizon_hours, transformer_kva)

    if "forecast_result" in st.session_state:
        summary, points, f_id, h_hrs, t_kva = st.session_state["forecast_result"]
        alert_lv, load_pct, max_kw = check_load_shedding(summary["max_kw"], t_kva)
        if alert_lv != "NORMAL":
            render_load_alert(alert_lv, load_pct, max_kw)

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Peak Demand",    f"{summary['max_kw']:.1f} kW")
        m2.metric("Mean Load",      f"{summary['mean_kw']:.1f} kW")
        m3.metric("Min Load",       f"{summary['min_kw']:.1f} kW")
        m4.metric("Total Energy",   f"{summary['total_kwh']:.0f} kWh")
        m5.metric("Capacity Used",  f"{load_pct*100:.1f}%")
        m6.metric("Transformer",    f"{t_kva} kVA")

        df = pd.DataFrame(points)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Interactive chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat_ensemble"], name="Ensemble Forecast",
            line=dict(color=TEAL, width=3), fill="tozeroy",
            fillcolor="rgba(0,212,170,0.06)",
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Demand: %{y:.1f} kW<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat_prophet"], name="Prophet (40%)",
            line=dict(color=BLUE, width=1.5, dash="dash"),
            hovertemplate="%{y:.1f} kW<extra>Prophet</extra>"))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["yhat_lgbm"], name="LightGBM (60%)",
            line=dict(color=AMBER, width=1.5, dash="dot"),
            hovertemplate="%{y:.1f} kW<extra>LightGBM</extra>"))

        cap_kw = t_kva * 0.85
        fig.add_hline(y=cap_kw, line_dash="dot", line_color=RED, opacity=0.7,
            annotation_text=f"Transformer limit {cap_kw:.0f} kW",
            annotation_font_color=RED, annotation_font_size=10)

        peak_idx = df["yhat_ensemble"].idxmax()
        fig.add_trace(go.Scatter(
            x=[df.loc[peak_idx,"timestamp"]], y=[df.loc[peak_idx,"yhat_ensemble"]],
            mode="markers+text", name="Peak",
            marker=dict(color=RED, size=12, symbol="triangle-up"),
            text=[f" {df.loc[peak_idx,'yhat_ensemble']:.0f} kW"],
            textposition="middle right", textfont=dict(color=RED, size=11),
            hovertemplate="<b>PEAK</b><br>%{x|%H:%M}<br>%{y:.1f} kW<extra></extra>"))

        fig.update_layout(**{**PLOTLY_LAYOUT,
            "title": f"11kV Feeder Load Profile — {f_id} | {h_hrs}h Horizon",
            "yaxis": {"title":"Demand (kW)", "gridcolor":"#1A2D45"},
            "xaxis": {"title":"Time (IST)", "gridcolor":"#1A2D45"},
            "height": 420,
        })
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline","eraseshape"], "scrollZoom": True})

        # Peak hour breakdown
        df_h = df.groupby(df["timestamp"].dt.hour)["yhat_ensemble"].mean().reset_index()
        df_h.columns = ["Hour","Avg kW"]
        top3 = df_h.nlargest(3,"Avg kW")
        st.markdown("**Top-3 Peak Hours**")
        pc1,pc2,pc3 = st.columns(3)
        for col_m, (_, row) in zip([pc1,pc2,pc3], top3.iterrows()):
            col_m.metric(f"{int(row['Hour']):02d}:00 IST", f"{row['Avg kW']:.1f} kW")

        lf = summary["mean_kw"] / summary["max_kw"] * 100
        st.info(f"Load Factor: {lf:.1f}% — Peak at {df.loc[peak_idx,'timestamp'].strftime('%H:%M')} IST")

        st.download_button("Export Forecast CSV",
            df[["timestamp","yhat_ensemble","yhat_prophet","yhat_lgbm"]].to_csv(index=False),
            f"forecast_{f_id}_{h_hrs}h.csv","text/csv")

        # Feature guide — clean table
        with st.expander("HOW IT WORKS — Feature Reference", expanded=False):
            t1,t2,t3,t4 = st.tabs(["Calendar","Lag & Rolling","Weather","Architecture"])
            with t1:
                st.markdown("""
| Feature | Description | Why It Matters |
|---|---|---|
| `hour` | Hour of day (0–23) | Strong intra-day demand cycles |
| `dayofweek` | Mon=0, Sun=6 | Industrial vs residential pattern shift |
| `is_peak_hour` | 07:00–10:00 or 18:00–22:00 IST | BESCOM tariff windows |
| `season` | 0=Winter 1=Spring 2=Summer 3=Monsoon | Bangalore AC load driver |
| `hour_sin/cos` | Cyclic encoding | Prevents midnight discontinuity |
                """)
            with t2:
                st.markdown("""
**Lag Features** — past demand at fixed 15-min period offsets

| Feature | Time Back | Signal |
|---|---|---|
| `lag_4` | 1 hour | Recent load level |
| `lag_96` | 24 hours | Same time yesterday |
| `lag_672` | 7 days | Same time last week |

**Rolling Stats** — `roll_mean_4` to `roll_mean_672` and `roll_std_4` to `roll_std_672`

Upload a CSV with historical data so lags use real values, not a 500 kW baseline.
                """)
            with t3:
                st.markdown("""
| Feature | Source | Description |
|---|---|---|
| `T2M` | NASA POWER | 2m temperature — primary AC load driver |
| `RH2M` | NASA POWER | Relative humidity — compressor load |
| `ALLSKY_SFC_SW_DWN` | NASA POWER | Solar irradiance — rooftop solar proxy |
                """)
            with t4:
                st.code("Prophet (40%) + LightGBM (60%) -> Ensemble\n  Prophet: Fourier seasonality + Karnataka holidays\n  LightGBM: 48 lag/weather features, quantile regression\n  Intervals: P10 / P90 quantile bounds")

# ══════════════════════════════════════════════════════════════════════════════
with tab_zone:
    st.subheader("Zone-Level Risk Heatmap — 8 BESCOM Zones")
    rng = np.random.default_rng(77)
    zone_data = []
    for zone, meta in BESCOM_ZONES.items():
        lf   = rng.uniform(0.55, 1.05)
        mean = rng.uniform(300, 900)
        peak = mean * rng.uniform(1.2, 1.6)
        risk = "HIGH" if lf > 0.90 else ("MEDIUM" if lf > 0.75 else "LOW")
        zone_data.append({"Zone": zone.replace("_"," "), "lat": meta["lat"], "lon": meta["lon"],
                          "Load Factor": f"{lf:.1%}", "Mean kW": round(mean,1),
                          "Peak kW": round(peak,1), "Risk": risk})
    df_zones = pd.DataFrame(zone_data)
    z1,z2,z3,z4 = st.columns(4)
    z1.metric("High-Risk Zones", len(df_zones[df_zones["Risk"]=="HIGH"]))
    z2.metric("Zones Monitored", len(df_zones))
    z3.metric("Peak Zone",       df_zones.loc[df_zones["Peak kW"].idxmax(),"Zone"])
    z4.metric("Avg Load Factor", f"{rng.uniform(0.72,0.88):.1%}")

    df_zones["color"] = df_zones["Risk"].map({"HIGH": [255, 71, 87, 200], "MEDIUM": [255, 184, 0, 180], "LOW": [0, 196, 140, 150]})
    df_zones["radius"] = df_zones["Load Factor"].str.rstrip("%").astype(float) * 50
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_zones,
        get_position=["lon", "lat"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=1,
    )
    view_state = pdk.ViewState(latitude=12.97, longitude=77.59, zoom=9.5, pitch=0)
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{Zone}\nLoad Factor: {Load Factor}\nPeak: {Peak kW} kW\nRisk: {Risk}"}
    ))
    st.dataframe(df_zones[["Zone","Load Factor","Mean kW","Peak kW","Risk"]],
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Side-by-Side Feeder Comparison")
    c1,c2,c3 = st.columns([2,2,1])
    fid1 = c1.text_input("Feeder A", value="FEEDER_001", key="cmp1")
    fid2 = c2.text_input("Feeder B", value="FEEDER_002", key="cmp2")
    c3.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    cmp_btn = c3.button("Compare", type="primary", use_container_width=True)

    # Comparison logic
    cmp_key = f"cmp_{fid1}_{fid2}"
    if cmp_btn or "cmp_res" not in st.session_state or st.session_state.get("cmp_last_key") != cmp_key:
        with st.spinner("Forecasting both feeders..."):
            s1,p1,_ = run_forecast(fid1, 24)
            s2,p2,_ = run_forecast(fid2, 24)
            st.session_state["cmp_res"] = (s1, p1, s2, p2)
            st.session_state["cmp_last_key"] = cmp_key
    
    s1, p1, s2, p2 = st.session_state["cmp_res"]
    df1 = pd.DataFrame(p1); df1["timestamp"] = pd.to_datetime(df1["timestamp"])
    df2 = pd.DataFrame(p2); df2["timestamp"] = pd.to_datetime(df2["timestamp"])
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=df1["timestamp"], y=df1["yhat_ensemble"],
        name=fid1, line=dict(color=TEAL, width=2.5)))
    fig_c.add_trace(go.Scatter(x=df2["timestamp"], y=df2["yhat_ensemble"],
        name=fid2, line=dict(color=AMBER, width=2.5)))
    fig_c.update_layout(**{**PLOTLY_LAYOUT, "title":"Feeder Comparison — 24h", "height":360})
    st.plotly_chart(fig_c, use_container_width=True)
    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric(f"{fid1} Peak",   f"{s1['max_kw']:.1f} kW")
    mc2.metric(f"{fid1} Energy", f"{s1['total_kwh']:.0f} kWh")
    mc3.metric(f"{fid2} Peak",   f"{s2['max_kw']:.1f} kW")
    mc4.metric(f"{fid2} Energy", f"{s2['total_kwh']:.0f} kWh")

# ══════════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.subheader("Model Architecture and Feature Reference")
    st.markdown("""
**Vidyut Ensemble** = Prophet (40%) + LightGBM (60%) trained on BESCOM 15-minute interval data.

| Category | Features | Count |
|---|---|---|
| Calendar | hour, dayofweek, dayofmonth, dayofyear, weekofyear, month, is_weekend, is_holiday, is_peak_hour, season + cyclic encodings | 13 |
| Lag | lag_4, lag_8, lag_12, lag_24, lag_48, lag_96, lag_192, lag_672 | 8 |
| Rolling Mean | roll_mean_4 / 8 / 16 / 32 / 96 / 192 / 672 | 7 |
| Rolling Std | roll_std_4 / 8 / 16 / 32 / 96 / 192 / 672 | 7 |
| Weather | T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, PRECTOTCORR, ALLSKY_SFC_SW_DWN | 7 |
| **Total** | | **42** |

### Load-Shedding Alert Thresholds

| Alert | Load % | Action |
|---|---|---|
| YELLOW | >= 80% | Warning — monitor closely |
| ORANGE | >= 90% | High risk — prepare load shedding |
| RED    | >= 95% | Imminent — execute shedding in 15-30 min |
| TRIP   | >= 110% | Transformer auto-protection will activate |
    """)
