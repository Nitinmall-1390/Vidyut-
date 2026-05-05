"""
VIDYUT — Page 3: Geospatial Map
PyDeck map of Bangalore with consumer anomaly heatmap,
feeder zone overlays, and click-to-expand consumer popups.
"""

import sys
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.components.shared import (
    inject_css, render_sidebar_brand, render_sidebar_status,
    get_synthetic_consumers, run_batch_theft_analysis,
    PLOTLY_LAYOUT, RED, AMBER, GREEN, TEAL, BLUE, PURPLE, ORANGE,
)
from src.config.feature_config import BESCOM_ZONES

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()
    st.markdown("---")
    st.markdown("**🗺️ MAP SETTINGS**")
    map_layer    = st.radio("Layer", ["Consumer Pins", "Theft Heatmap", "Zone Overlay", "All Layers"])
    zone_filter  = st.multiselect("Filter Zones", list(BESCOM_ZONES.keys()),
                                  default=list(BESCOM_ZONES.keys()))
    risk_filter  = st.multiselect("Risk Levels", ["HIGH", "MEDIUM", "LOW"],
                                  default=["HIGH", "MEDIUM"])
    n_consumers  = st.slider("Dataset Size", 100, 500, 200, 50)
    st.markdown("---")
    render_sidebar_status()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>BESCOM GIS | Geospatial Intelligence Module</div>", unsafe_allow_html=True)
st.title("🗺️ Geospatial Theft Intelligence Map")
st.caption("Bangalore-wide anomaly heatmap · Feeder zone overlays · Consumer-level drill-down")

# ── Load/generate data ────────────────────────────────────────────────────────
if st.button("🔄 LOAD MAP DATA", type="primary") or "geo_data" not in st.session_state:
    with st.spinner("Generating geospatial intelligence layer..."):
        df_alerts = run_batch_theft_analysis(str(n_consumers), n_consumers)
        # Map risk levels
        def _risk(prob):
            return "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")
        df_alerts["risk_level"] = df_alerts["prob_theft"].apply(_risk)
        st.session_state["geo_data"] = df_alerts

df_geo = st.session_state.get("geo_data", pd.DataFrame())

if df_geo.empty:
    st.info("Click **LOAD MAP DATA** to generate the geospatial intelligence layer.")
    st.stop()

# Apply filters
df_filtered = df_geo.copy()
if zone_filter:
    df_filtered = df_filtered[df_filtered["zone"].isin(zone_filter)]
if risk_filter:
    df_filtered = df_filtered[df_filtered["risk_level"].isin(risk_filter)]

# ── Summary Metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Consumers", len(df_filtered))
m2.metric("Flagged Anomalies", len(df_filtered[df_filtered["prob_theft"] >= 0.5]))
m3.metric("HIGH Risk", len(df_filtered[df_filtered["risk_level"] == "HIGH"]))
m4.metric("Zones Active", len(df_filtered["zone"].unique()))
m5.metric("Avg Risk Score", f"{df_filtered['prob_theft'].mean():.1%}")

st.markdown("---")

# ── Map View ──────────────────────────────────────────────────────────────────
col_map, col_panel = st.columns([3, 1])

with col_map:
    try:
        import pydeck as pdk

        # Color mapping
        def get_rgba(row):
            if row["risk_level"] == "HIGH":
                return [255, 71, 87, 200]
            elif row["risk_level"] == "MEDIUM":
                return [255, 184, 0, 180]
            return [0, 196, 140, 150]

        df_filtered = df_filtered.copy()
        df_filtered["color"] = df_filtered.apply(get_rgba, axis=1)
        df_filtered["radius"] = df_filtered["prob_theft"].apply(lambda p: int(100 + p * 300))
        df_filtered["tooltip_text"] = df_filtered.apply(
            lambda r: f"{r['consumer_id']} · {r['loss_type'].upper()} · {r['prob_theft']:.1%}", axis=1
        )

        # Scatterplot layer (consumer pins)
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.85,
            stroked=True,
            get_line_color=[255, 255, 255, 80],
            line_width_min_pixels=1,
        )

        # Heatmap layer
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=df_filtered[df_filtered["prob_theft"] >= 0.5],
            get_position=["longitude", "latitude"],
            get_weight="prob_theft",
            radius_pixels=60,
            intensity=1,
            threshold=0.05,
        )

        # Zone bubble layer
        zone_rows = []
        for zone, meta in BESCOM_ZONES.items():
            if zone in zone_filter:
                zone_df = df_filtered[df_filtered["zone"] == zone]
                zone_rows.append({
                    "zone": zone, "latitude": meta["lat"], "longitude": meta["lon"],
                    "count": len(zone_df),
                    "avg_risk": zone_df["prob_theft"].mean() if len(zone_df) > 0 else 0,
                })
        df_zones_map = pd.DataFrame(zone_rows) if zone_rows else pd.DataFrame()

        layers = []
        if "Theft Heatmap" in map_layer or "All Layers" in map_layer:
            layers.append(heatmap_layer)
        if "Consumer Pins" in map_layer or "All Layers" in map_layer:
            layers.append(scatter_layer)

        view = pdk.ViewState(
            latitude=12.97, longitude=77.59,
            zoom=10.5, pitch=35, bearing=0,
        )

        tooltip = {
            "html": "<b>{tooltip_text}</b><br>Zone: {zone}<br>Type: {loss_type}",
            "style": {
                "backgroundColor": "#0E1A2E",
                "color": "#ECF0F6",
                "fontSize": "12px",
                "padding": "8px",
                "border": "1px solid #1A2D45",
                "borderRadius": "6px",
            },
        }

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/dark-v11",
            tooltip=tooltip,
        )
        st.pydeck_chart(deck, use_container_width=True)

    except ImportError:
        # Fallback: Plotly scattermap
        st.info("📍 Using Plotly map (pydeck not available)")

        color_map = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}
        fig_scatter = go.Figure()

        for risk in ["HIGH", "MEDIUM", "LOW"]:
            df_r = df_filtered[df_filtered["risk_level"] == risk]
            if len(df_r) == 0:
                continue
            fig_scatter.add_trace(go.Scattermapbox(
                lat=df_r["latitude"], lon=df_r["longitude"],
                mode="markers",
                marker=dict(size=df_r["prob_theft"].apply(lambda p: max(8, p * 20)).tolist(),
                            color=color_map[risk], opacity=0.8),
                name=f"{risk} Risk",
                text=df_r["consumer_id"],
                hovertemplate="<b>%{text}</b><br>Zone: %{customdata[0]}<br>Type: %{customdata[1]}<extra></extra>",
                customdata=df_r[["zone", "loss_type"]].values,
            ))

        # Zone labels
        for zone, meta in BESCOM_ZONES.items():
            if zone in zone_filter:
                fig_scatter.add_trace(go.Scattermapbox(
                    lat=[meta["lat"]], lon=[meta["lon"]],
                    mode="text",
                    text=[zone.replace("_", " ")],
                    textfont=dict(size=10, color="#7B8FAB"),
                    showlegend=False,
                ))

        fig_scatter.update_layout(
            mapbox=dict(style="dark", center=dict(lat=12.97, lon=77.59), zoom=10.5),
            paper_bgcolor="rgba(0,0,0,0)", height=500,
            legend=dict(orientation="h", y=-0.05, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=0, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with col_panel:
    st.markdown("**Zone Summary**")
    for zone in zone_filter[:6]:
        z_df = df_filtered[df_filtered["zone"] == zone]
        n_flagged = len(z_df[z_df["prob_theft"] >= 0.5])
        risk_col = RED if n_flagged > 10 else (AMBER if n_flagged > 3 else GREEN)
        st.markdown(
            f"<div style='background:#0E1A2E;border:1px solid #1A2D45;border-left:3px solid {risk_col};"
            f"border-radius:0 6px 6px 0;padding:8px 10px;margin-bottom:6px;font-size:12px;'>"
            f"<b>{zone.replace('_',' ')}</b><br>"
            f"<span style='color:#7B8FAB;'>Consumers: {len(z_df)} · Flagged: {n_flagged}</span>"
            f"</div>", unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown(f"""
<div style='font-size:12px;line-height:2;'>
  <span style='color:{RED};'>●</span> HIGH Risk (≥65%)<br>
  <span style='color:{AMBER};'>●</span> MEDIUM Risk (35-65%)<br>
  <span style='color:{GREEN};'>●</span> LOW Risk (&lt;35%)<br>
  <span style='color:#7B8FAB;font-size:10px;'>Circle size ∝ theft probability</span>
</div>""", unsafe_allow_html=True)

# ── Consumer Detail Table ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Consumer Hotspot Table")

top_hotspots = df_filtered.nlargest(50, "prob_theft")[
    ["consumer_id", "zone", "feeder_id", "consumer_type", "loss_type",
     "prob_theft", "confidence", "latitude", "longitude"]
].copy()
top_hotspots["prob_theft"] = top_hotspots["prob_theft"].apply(lambda x: f"{x:.1%}")
top_hotspots.columns = ["Consumer", "Zone", "Feeder", "Type", "Loss",
                         "Risk %", "Conf", "Lat", "Lon"]

st.dataframe(top_hotspots, use_container_width=True, height=320, hide_index=True)

csv_geo = df_filtered.to_csv(index=False)
st.download_button("⬇ Export Geospatial Data CSV", csv_geo,
                   "vidyut_geospatial_alerts.csv", "text/csv")
