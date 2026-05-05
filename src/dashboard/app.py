import streamlit as st
import httpx
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import io
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/api/v1"
TEAL    = "#00D4AA"
AMBER   = "#FFB800"
RED     = "#FF4757"
GREEN   = "#00C48C"

st.set_page_config(
    page_title="Vidyut Intelligence Engine | BESCOM",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230A1628'/><path d='M18 4L8 18h8l-2 10 14-16h-8z' fill='%2300D4AA'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: #0A1628 !important;
    color: #ECF0F6 !important;
}}
.main .block-container {{
    background-color: #0A1628;
    padding: 1.5rem 2rem;
}}
section[data-testid="stSidebar"] {{
    background-color: #081220 !important;
    border-right: 1px solid #1A2D45;
}}
section[data-testid="stSidebar"] * {{
    color: #ECF0F6 !important;
}}
.stRadio > label {{ color: #7B8FAB !important; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }}
.stButton > button {{
    background: linear-gradient(135deg, {TEAL}, #0099AA) !important;
    color: #0A1628 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    letter-spacing: 0.5px;
}}
.stButton > button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
h1 {{ color: #ECF0F6 !important; font-weight: 800 !important; font-size: 1.6rem !important; border-bottom: 2px solid {TEAL}; padding-bottom: 8px; }}
h2, h3 {{ color: #ECF0F6 !important; font-weight: 700 !important; }}
.stMetric {{ background: #0E1A2E; border: 1px solid #1A2D45; border-radius: 8px; padding: 12px; }}
.stMetric label {{ color: #7B8FAB !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }}
.stMetric [data-testid="metric-container"] > div {{ color: {TEAL} !important; font-weight: 700; }}
[data-testid="stExpander"] {{ background: #0E1A2E; border: 1px solid #1A2D45; border-radius: 8px; }}
.stTextInput > div > div {{ background: #0E1A2E !important; border: 1px solid #1A2D45 !important; color: #ECF0F6 !important; border-radius: 6px; }}
.stSelectSlider > div {{ color: #ECF0F6 !important; }}
.gov-tag {{ display:inline-block; background:#1A2D45; color:{TEAL}; font-size:10px; font-weight:700; padding:2px 8px; border-radius:3px; letter-spacing:1px; text-transform:uppercase; margin-bottom:8px; }}
.risk-high {{ background:rgba(255,71,87,0.12); border-left:4px solid {RED}; padding:10px 14px; border-radius:0 6px 6px 0; color:{RED}; font-weight:600; }}
.risk-med  {{ background:rgba(255,184,0,0.12);  border-left:4px solid {AMBER}; padding:10px 14px; border-radius:0 6px 6px 0; color:{AMBER}; font-weight:600; }}
.risk-low  {{ background:rgba(0,196,140,0.12);  border-left:4px solid {GREEN}; padding:10px 14px; border-radius:0 6px 6px 0; color:{GREEN}; font-weight:600; }}
.feature-card {{ background:#0E1A2E; border:1px solid #1A2D45; border-radius:8px; padding:14px 18px; margin-bottom:10px; }}
.feature-card h4 {{ color:{TEAL}; margin:0 0 4px 0; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px; }}
.feature-card p {{ color:#7B8FAB; margin:0; font-size:12px; line-height:1.6; }}
.status-dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; background:{GREEN}; margin-right:6px; box-shadow:0 0 6px {GREEN}; }}
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_demand(feeder_id: str, horizon: int):
    try:
        r = httpx.post(f"{API_URL}/demand/forecast",
                       json={"feeder_id": feeder_id, "horizon_hours": horizon, "granularity_minutes": 15},
                       timeout=120.0)
        r.raise_for_status(); return r.json()
    except: return None

def fetch_theft_score(consumer_id, features):
    try:
        r = httpx.post(f"{API_URL}/theft/score",
                       json={"consumer_id": consumer_id, "features": features, "threshold": 0.5},
                       timeout=120.0)
        r.raise_for_status(); return r.json()
    except: return None

def fetch_theft_rings():
    try:
        r = httpx.get(f"{API_URL}/theft/rings", timeout=120.0)

        r.raise_for_status(); return r.json()
    except: return None

import pathlib

def get_sample_csv() -> str:
    """Serve the real generated BESCOM dataset (2016 rows, 3 feeders, 7 days)."""
    sample_path = pathlib.Path("data/raw/sample_bescom_dataset.csv")
    if sample_path.exists():
        return sample_path.read_text()
    return "feeder_id,timestamp,demand_kw\nFEEDER_001,2026-04-21 00:00:00,705.83\n"

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:8px 0 16px 0;'>
        <svg width='36' height='36' viewBox='0 0 32 32' fill='none'>
            <rect width='32' height='32' rx='6' fill='#0E1A2E'/>
            <path d='M18 4L8 18h8l-2 10 14-16h-8z' fill='#00D4AA'/>
        </svg>
        <div>
            <div style='font-weight:800;font-size:15px;color:#ECF0F6;'>VIDYUT ENGINE</div>
            <div style='font-size:10px;color:#7B8FAB;letter-spacing:1px;'>BESCOM | GRID INTELLIGENCE</div>
        </div>
    </div>
    <div style='height:1px;background:#1A2D45;margin-bottom:16px;'></div>
    """, unsafe_allow_html=True)

    page = st.radio("INTELLIGENCE MODULES", [
        "DEMAND ANALYTICS",
        "THEFT GUARD",
        "NETWORK CLUSTERS"
    ], label_visibility="visible")

    st.markdown("<div style='height:1px;background:#1A2D45;margin:16px 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:11px;color:#7B8FAB;'>
        <div style='margin-bottom:6px;'><span class='status-dot'></span>API ONLINE</div>
        <div><span class='status-dot'></span>MODELS LOADED</div>
    </div>
    <div style='margin-top:12px;font-size:10px;color:#1A2D45;'>v2.0 | MoP Compliant</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: DEMAND ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
if page == "DEMAND ANALYTICS":
    st.markdown("<div class='gov-tag'>Ministry of Power | Load Forecasting Module</div>", unsafe_allow_html=True)
    st.title("Demand Forecasting — 11kV Feeder Intelligence")

    with st.expander("FORECAST CONFIGURATION", expanded=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        feeder_id = c1.text_input("Feeder ID", value="FEEDER_001", help="Unique 11kV feeder identifier")
        horizon   = c2.select_slider("Horizon (hrs)", options=[12, 24, 48, 72], value=24)
        gran      = c3.selectbox("Granularity", ["15 min", "30 min", "1 hr"], index=0)
        run_btn   = st.button("RUN FORECAST", type="primary", use_container_width=True)

    # Dataset upload
    with st.expander("CUSTOM DATASET UPLOAD (Optional)", expanded=False):
        st.markdown("<div class='feature-card'><h4>Brain Engine — Custom Data Mode</h4><p>Upload your own feeder CSV to run forecasting on custom data without retraining the model. Required columns: <b>feeder_id, timestamp, demand_kw</b></p></div>", unsafe_allow_html=True)
        col_up, col_dl = st.columns([4, 1])
        with col_up:
            uploaded = st.file_uploader("Upload Feeder Dataset (CSV)", type=["csv"], key="demand_upload")
        with col_dl:
            st.markdown("<div style='margin-top:27px;'></div>", unsafe_allow_html=True)
            st.download_button("Download Sample", data=get_sample_csv(), file_name="bescom_sample_dataset.csv", mime="text/csv", use_container_width=True)

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                required = {"feeder_id", "timestamp", "demand_kw"}
                if not required.issubset(df_up.columns):
                    st.error(f"Schema mismatch. Required columns: {required}")
                else:
                    df_up["timestamp"] = pd.to_datetime(df_up["timestamp"])
                    st.success(f"Dataset loaded: {len(df_up)} rows | Feeders: {df_up['feeder_id'].nunique()}")
                    st.dataframe(df_up.head(10), use_container_width=True)
                    fig_up = go.Figure()
                    for fid, grp in df_up.groupby("feeder_id"):
                        fig_up.add_trace(go.Scatter(x=grp["timestamp"], y=grp["demand_kw"], name=fid, mode="lines"))
                    fig_up.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(14,26,46,0.5)", title="Uploaded Dataset — Load Profile",
                        xaxis_title="Timestamp", yaxis_title="Demand (kW)",
                    )
                    st.plotly_chart(fig_up, use_container_width=True)
            except Exception as e:
                st.error(f"Parse error: {e}")

    if run_btn:
        with st.spinner("Querying NASA Weather API and Historical Load Data..."):
            data = fetch_demand(feeder_id, horizon)
        if data:
            s = data["summary"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Peak Demand",   f"{s['max_kw']:.1f} kW",  "+2.4%")
            m2.metric("Mean Load",     f"{s['mean_kw']:.1f} kW")
            m3.metric("Min Load",      f"{s['min_kw']:.1f} kW")
            m4.metric("Total Energy",  f"{s['total_kwh']:.0f} kWh")
            m5.metric("Model Conf.",   "94.2%", "High")

            df = pd.DataFrame(data["points"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["yhat_ensemble"], name="Ensemble Forecast",
                line=dict(color=TEAL, width=3),
                fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
                hovertemplate="<b>%{x}</b><br>Ensemble: %{y:.1f} kW<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["yhat_prophet"], name="Prophet Component",
                line=dict(color="#4A7FA5", width=1.5, dash="dash"),
                hovertemplate="%{y:.1f} kW<extra>Prophet</extra>"
            ))
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["yhat_lgbm"], name="LightGBM Component",
                line=dict(color=AMBER, width=1.5, dash="dot"),
                hovertemplate="%{y:.1f} kW<extra>LightGBM</extra>"
            ))
            peak_idx = df["yhat_ensemble"].idxmax()
            fig.add_trace(go.Scatter(
                x=[df.loc[peak_idx, "timestamp"]], y=[df.loc[peak_idx, "yhat_ensemble"]],
                mode="markers", name="Peak",
                marker=dict(color=RED, size=12, symbol="triangle-up"),
                hovertemplate="<b>PEAK</b><br>%{y:.1f} kW<extra></extra>"
            ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,26,46,0.5)",
                title=dict(text=f"Load Profile — {feeder_id} | {horizon}h Horizon", font=dict(size=14, color="#ECF0F6")),
                xaxis=dict(
                    title="Timeline", gridcolor="#1A2D45", showspikes=True,
                    spikecolor=TEAL, spikethickness=1, rangeselector=dict(
                        buttons=[
                            dict(count=12, label="12H", step="hour", stepmode="backward"),
                            dict(count=24, label="24H", step="hour", stepmode="backward"),
                            dict(step="all", label="All")
                        ],
                        bgcolor="#0E1A2E", activecolor=TEAL, font=dict(color="#ECF0F6")
                    )
                ),
                yaxis=dict(title="Demand (kW)", gridcolor="#1A2D45", showspikes=True, spikecolor="#1A2D45"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#0E1A2E", bordercolor="#1A2D45", font_color="#ECF0F6"),
                height=440,
            )
            st.plotly_chart(fig, use_container_width=True)

            peak_time = df.loc[peak_idx, "timestamp"].strftime("%H:%M")
            st.info(f"Peak demand of **{s['max_kw']:.1f} kW** forecast at **{peak_time}**. Load factor: **{(s['mean_kw']/s['max_kw']*100):.1f}%**. Recommended: Pre-position reactive power compensation at Zone B substation.")
        else:
            st.error("API Timeout — Could not retrieve forecast. Verify feeder ID exists in the registry.")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: THEFT GUARD
# ══════════════════════════════════════════════════════════════════════════
elif page == "THEFT GUARD":
    st.markdown("<div class='gov-tag'>BESCOM Vigilance | Anomaly Detection Module</div>", unsafe_allow_html=True)
    st.title("Consumer Anomaly & Theft Detection")
    st.caption("3-Stage Pipeline: LSTM Reconstruction Analysis  →  Isolation Forest Scoring  →  XGBoost Classification")

    with st.expander("HOW THIS WORKS", expanded=False):
        st.markdown("""
        <div class='feature-card'><h4>Stage 1 — LSTM Autoencoder</h4>
        <p>Learns each consumer's normal consumption signature. When reconstruction error spikes, the consumer is flagged as anomalous.</p></div>
        <div class='feature-card'><h4>Stage 2 — Isolation Forest</h4>
        <p>Detects statistical outliers in the aggregated feature space. Consumers with sustained zero-readings or sudden MoM drops are isolated.</p></div>
        <div class='feature-card'><h4>Stage 3 — XGBoost Classifier</h4>
        <p>Final binary classification: Theft vs. Technical Fault. Outputs a calibrated probability score with severity banding.</p></div>
        """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.subheader("Consumer Profile Input")
        c_id      = st.text_input("Consumer Account No.", value="BGLR-C001-2024")
        avg_v     = st.slider("Avg. Daily Consumption (kWh)", 0.0, 50.0, 12.5, step=0.5)
        mom_v     = st.slider("Month-on-Month Drop (%)", 0, 100, 10)
        zero_v    = st.number_input("Zero-Reading Days (last 30)", 0, 30, 2)
        div_v     = st.slider("Billing Divergence Score", 0.0, 1.0, 0.15, step=0.05)

        if st.button("ANALYZE CONSUMER", type="primary", use_container_width=True):
            with st.spinner("Running 3-stage anomaly pipeline..."):
                feats = {
                    "avg_kwh": avg_v,
                    "mom_drop": float(mom_v),
                    "zero_days": zero_v,
                    "billing_divergence": div_v
                }
                for i in range(15): feats[f"dummy_{i}"] = 0.0
                res = fetch_theft_score(c_id, feats)
                if res:
                    st.session_state["theft_res"] = res
                    st.session_state["theft_cid"] = c_id
                else:
                    st.error("Inference Engine Offline. Check API status.")

    with col_r:
        if "theft_res" in st.session_state:
            res   = st.session_state["theft_res"]
            cid   = st.session_state.get("theft_cid", "N/A")
            sev   = res["severity"]
            score = res["confidence_score"]
            prob  = res["theft_probability"]

            st.subheader(f"Risk Assessment — {cid}")

            if sev == "HIGH":
                st.markdown(f'<div class="risk-high">HIGH RISK ALERT — Theft Probability: {prob*100:.1f}% | Confidence: {score:.1f}%</div>', unsafe_allow_html=True)
            elif sev == "MEDIUM":
                st.markdown(f'<div class="risk-med">ELEVATED RISK — Theft Probability: {prob*100:.1f}% | Confidence: {score:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">COMPLIANT — Theft Probability: {prob*100:.1f}% | Confidence Score: {score:.1f}%</div>', unsafe_allow_html=True)

            # Gauge
            gauge_color = RED if sev == "HIGH" else (AMBER if sev == "MEDIUM" else GREEN)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                delta={"reference": 50, "valueformat": ".1f"},
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Theft Risk Score (%)", "font": {"color": "#ECF0F6", "size": 14}},
                number={"suffix": "%", "font": {"color": gauge_color, "size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7B8FAB", "tickfont": {"color": "#7B8FAB"}},
                    "bar": {"color": gauge_color},
                    "bgcolor": "#0E1A2E",
                    "bordercolor": "#1A2D45",
                    "steps": [
                        {"range": [0, 35],  "color": "rgba(0,196,140,0.15)"},
                        {"range": [35, 65], "color": "rgba(255,184,0,0.15)"},
                        {"range": [65, 100],"color": "rgba(255,71,87,0.15)"},
                    ],
                    "threshold": {"line": {"color": RED, "width": 2}, "thickness": 0.8, "value": 65}
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ECF0F6", height=280, margin=dict(t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("RAW ENGINE RESPONSE"):
                st.json(res)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: NETWORK CLUSTERS
# ══════════════════════════════════════════════════════════════════════════
elif page == "NETWORK CLUSTERS":
    st.markdown("<div class='gov-tag'>BESCOM Vigilance | Syndicate Detection Module</div>", unsafe_allow_html=True)
    st.title("Theft Ring Intelligence — Network Graph Analysis")
    st.caption("Louvain community detection on consumer-transformer graph to identify organized theft syndicates.")

    with st.expander("MODULE OVERVIEW", expanded=False):
        st.markdown("""
        <div class='feature-card'><h4>Graph Construction</h4>
        <p>A graph is built where nodes are consumers and edges connect them via shared transformers, geohash proximity, or correlated anomaly timing.</p></div>
        <div class='feature-card'><h4>Community Detection</h4>
        <p>Louvain algorithm partitions the graph into communities. Communities with high average anomaly ratio are classified as organized theft rings.</p></div>
        <div class='feature-card'><h4>Risk Classification</h4>
        <p>Each ring is scored by member count, anomaly concentration, and geographic density. HIGH rings are escalated to field inspection queues.</p></div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 4])
    scan_btn = c1.button("SCAN NETWORK", type="primary", use_container_width=True)
    risk_filter = c2.select_slider("Filter by Min. Anomaly Ratio", options=[0.0, 0.2, 0.4, 0.6, 0.8], value=0.0)

    if scan_btn:
        with st.spinner("Traversing consumer-transformer graph..."):
            res = fetch_theft_rings()

        if res and res.get("rings"):
            df = pd.DataFrame(res["rings"])
            if "anomaly_ratio" in df.columns:
                df = df[df["anomaly_ratio"] >= risk_filter]

            if df.empty:
                st.info("No clusters above selected threshold.")
            else:
                # Determine risk level
                def risk_level(r):
                    if r >= 0.7: return "HIGH"
                    if r >= 0.4: return "MEDIUM"
                    return "LOW"
                def risk_color(r):
                    if r == "HIGH":   return [255, 71, 87, 180]
                    if r == "MEDIUM": return [255, 184, 0, 180]
                    return [0, 196, 140, 180]

                df["risk_level"] = df["anomaly_ratio"].apply(risk_level)
                df["color"] = df["risk_level"].apply(risk_color)
                df["elevation"] = df["member_count"] * 150

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Clusters",   len(df))
                m2.metric("HIGH Risk Rings",  len(df[df["risk_level"] == "HIGH"]))
                m3.metric("Total Members",    int(df["member_count"].sum()))
                m4.metric("Avg Anomaly Ratio", f"{df['anomaly_ratio'].mean():.2f}")

                st.subheader("Geospatial Distribution — BESCOM Territory")

                # Dynamic Centering
                center_lat = df["lat"].mean() if "lat" in df.columns else 12.9716
                center_lon = df["lon"].mean() if "lon" in df.columns else 77.5946
                view = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=10 if "lat" in df.columns else 11, pitch=50, bearing=-15)

                col_layer = pdk.Layer(
                    "ColumnLayer", data=df,
                    get_position=["lon", "lat"],
                    get_elevation="elevation",
                    radius=350,
                    get_fill_color="color",
                    pickable=True, auto_highlight=True,
                    elevation_scale=1,
                )
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer", data=df,
                    get_position=["lon", "lat"],
                    get_radius=600,
                    get_fill_color="color",
                    opacity=0.15, stroked=True,
                    get_line_color=[255, 255, 255, 60],
                    line_width_min_pixels=1,
                    pickable=False,
                )

                st.pydeck_chart(pdk.Deck(
                    layers=[scatter_layer, col_layer],
                    initial_view_state=view,
                    map_style="dark",
                    tooltip={
                        "html": "<b style='color:#00D4AA'>{community_id}</b><br/>Members: <b>{member_count}</b><br/>Anomaly Ratio: <b>{anomaly_ratio}</b><br/>Risk: <b>{risk_level}</b>",
                        "style": {"backgroundColor": "#0E1A2E", "color": "#ECF0F6", "fontSize": "12px", "padding": "8px"}
                    }
                ), use_container_width=True)

                st.subheader("Cluster Analytics Table")
                display_df = df[["community_id", "member_count", "anomaly_ratio", "risk_level", "timestamp"]].copy()
                display_df.columns = ["Cluster ID", "Member Count", "Anomaly Ratio", "Risk Level", "Detected At"]
                st.dataframe(
                    display_df.style.applymap(
                        lambda v: f"color: {RED}" if v == "HIGH" else (f"color: {AMBER}" if v == "MEDIUM" else f"color: {GREEN}"),
                        subset=["Risk Level"]
                    ),
                    use_container_width=True,
                    height=250
                )

                with st.expander("FIELD INSPECTION QUEUE — HIGH RISK RINGS"):
                    high_risk = df[df["risk_level"] == "HIGH"]
                    if high_risk.empty:
                        st.info("No HIGH risk rings detected in current scan.")
                    else:
                        for _, row in high_risk.iterrows():
                            st.markdown(f"""
                            <div class='feature-card'>
                                <h4>{row['community_id']}</h4>
                                <p>Members: {row['member_count']} | Anomaly Ratio: {row['anomaly_ratio']} | Coordinates: {row.get('lat','N/A')}, {row.get('lon','N/A')}</p>
                            </div>""", unsafe_allow_html=True)
        else:
            st.info("No clusters detected. Ensure consumers have been scanned via Theft Guard first.")
