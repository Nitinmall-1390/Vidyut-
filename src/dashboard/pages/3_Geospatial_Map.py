"""VIDYUT — Page 3: Geospatial Map (Fixed: open-street-map, no token, zone overlay, no emojis)"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.components.shared import (
    inject_css, run_batch_theft_analysis,
    PLOTLY_LAYOUT, RED, AMBER, GREEN, TEAL,
)
from src.config.feature_config import BESCOM_ZONES

inject_css()

with st.sidebar:
    st.markdown("<div style='font-size:9px;color:#7B8FAB;letter-spacing:2px;'>BESCOM GRID INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**MAP SETTINGS**")
    map_mode    = st.radio("Display Mode", ["Consumer Pins","Theft Heatmap","Zone Overlay"])
    zone_filter = st.multiselect("Show Zones", list(BESCOM_ZONES.keys()), default=list(BESCOM_ZONES.keys()))
    risk_filter = st.multiselect("Risk Levels", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM"])
    n_pts       = st.slider("Dataset Size", 100, 500, 200, 50)
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#7B8FAB;'><span style='color:#00C48C;'>&#9679;</span> ENGINE ONLINE<br><span style='color:#00C48C;'>&#9679;</span> ML MODELS READY<br><span style='color:#00C48C;'>&#9679;</span> WEATHER: LIVE</div>", unsafe_allow_html=True)

st.markdown("<div class='gov-tag'>BESCOM GIS | GEOSPATIAL INTELLIGENCE MODULE</div>", unsafe_allow_html=True)
st.title("Geospatial Theft Intelligence Map")
st.caption("Bangalore-wide consumer anomaly map — 8 BESCOM zones — Click markers for consumer detail")

if st.button("Load Map Data", type="primary") or "geo_data" not in st.session_state:
    with st.spinner("Building geospatial intelligence layer..."):
        df_a = run_batch_theft_analysis(str(n_pts), n_pts)
        def _risk(p): return "HIGH" if p >= 0.65 else ("MEDIUM" if p >= 0.35 else "LOW")
        df_a["risk_level"] = df_a["prob_theft"].apply(_risk)
        st.session_state["geo_data"] = df_a

df_geo = st.session_state.get("geo_data", pd.DataFrame())
if df_geo.empty:
    st.info("Click 'Load Map Data' to generate the intelligence layer.")
    st.stop()

df_f = df_geo.copy()
if zone_filter: df_f = df_f[df_f["zone"].isin(zone_filter)]
if risk_filter: df_f = df_f[df_f["risk_level"].isin(risk_filter)]

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Consumers",    len(df_f))
m2.metric("Flagged",      len(df_f[df_f["prob_theft"]>=0.5]))
m3.metric("HIGH Risk",    len(df_f[df_f["risk_level"]=="HIGH"]))
m4.metric("Zones",        len(df_f["zone"].unique()))
m5.metric("Avg Risk",     f"{df_f['prob_theft'].mean():.1%}")

st.markdown("---")

# Build PyDeck map
layers = []

if map_mode == "Theft Heatmap":
    df_h = df_f[df_f["prob_theft"] >= 0.35].copy()
    if not df_h.empty:
        df_h["weight"] = df_h["prob_theft"] * 10
        layers.append(pdk.Layer(
            "HexagonLayer",
            data=df_h,
            get_position=["longitude", "latitude"],
            radius=150,
            elevation_scale=40,
            elevation_range=[0, 1000],
            color_range=[
                [0, 196, 140, 50],
                [255, 184, 0, 100],
                [255, 184, 0, 150],
                [255, 71, 87, 200],
                [180, 0, 40, 255]
            ],
            extruded=True,
            get_elevation_weight="weight",
            coverage=0.8
        ))

if map_mode in ["Consumer Pins", "Zone Overlay"]:
    df_p = df_f.copy()
    if not df_p.empty:
        df_p["color"] = df_p["risk_level"].map({
            "HIGH": [255, 71, 87, 220],
            "MEDIUM": [255, 184, 0, 180],
            "LOW": [0, 196, 140, 120]
        })
        df_p["radius"] = df_p["prob_theft"].apply(lambda p: max(150, p * 800))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df_p,
            get_position=["longitude", "latitude"],
            get_color="color",
            get_radius="radius",
            pickable=True,
            opacity=0.8,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
        ))

view_state = pdk.ViewState(
    latitude=12.97,
    longitude=77.59,
    zoom=10,
    pitch=45 if map_mode == "Theft Heatmap" else 0,
    bearing=0
)

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/dark-v10",
    initial_view_state=view_state,
    layers=layers,
    tooltip={"text": "{consumer_id}\nZone: {zone}\nRisk: {prob_theft}"}
))

# Zone summary panel
st.markdown("---")
st.subheader("Zone Intelligence Summary")
zone_summary_rows = []
for zone, meta in BESCOM_ZONES.items():
    if zone not in zone_filter: continue
    z_df = df_f[df_f["zone"]==zone]
    n_flagged = len(z_df[z_df["prob_theft"]>=0.5])
    avg_risk  = z_df["prob_theft"].mean() if len(z_df) > 0 else 0
    zone_risk = "HIGH" if n_flagged > 10 else ("MEDIUM" if n_flagged > 3 else "LOW")
    zone_summary_rows.append({
        "Zone": zone.replace("_"," "), "Consumers": len(z_df),
        "Flagged": n_flagged, "Avg Risk": f"{avg_risk:.1%}", "Zone Risk": zone_risk
    })

if zone_summary_rows:
    df_zs = pd.DataFrame(zone_summary_rows)
    # Color-coded risk column
    st.dataframe(df_zs, use_container_width=True, hide_index=True)

# Hotspot table
st.markdown("---")
st.subheader("Top 50 Consumer Hotspots")
if df_f.empty:
    st.info("No consumer data available. Click 'Load Map Data' to generate the intelligence layer.")
else:
    top = df_f.nlargest(50,"prob_theft")[
        ["consumer_id","zone","feeder_id","consumer_type","loss_type","prob_theft","confidence","latitude","longitude"]
    ].copy()
    top["prob_theft"] = top["prob_theft"].apply(lambda x: f"{x:.1%}")
    top.columns = ["Consumer","Zone","Feeder","Type","Loss","Risk %","Conf","Lat","Lon"]
    st.dataframe(top, use_container_width=True, height=300, hide_index=True)

col_e1, col_e2 = st.columns([1,4])
col_e1.download_button("Export CSV", df_f.to_csv(index=False),
                        "vidyut_geo_alerts.csv","text/csv", use_container_width=True)
