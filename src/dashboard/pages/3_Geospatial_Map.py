"""VIDYUT — Page 3: Geospatial Map (Fixed: open-street-map, no token, zone overlay, no emojis)"""
import sys, pathlib, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

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
    st.markdown("<div style='font-weight:900;font-size:16px;color:#ECF0F6;padding:8px 0 4px 0;'>VIDYUT</div>", unsafe_allow_html=True)
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

# Build Plotly figure — open-street-map (no token needed)
color_map = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}

fig = go.Figure()

# ─── Layer 1: Zone Overlay circles ───────────────────────────────────────────
if map_mode == "Zone Overlay" or map_mode == "Consumer Pins":
    for zone, meta in BESCOM_ZONES.items():
        if zone not in zone_filter:
            continue
        z_df = df_f[df_f["zone"] == zone]
        n_flagged = len(z_df[z_df["prob_theft"] >= 0.5])
        risk_pct  = n_flagged / max(len(z_df), 1)
        zone_risk = "HIGH" if risk_pct > 0.35 else ("MEDIUM" if risk_pct > 0.15 else "LOW")
        fill_col  = color_map[zone_risk]
        # Draw zone label bubble
        fig.add_trace(go.Scattermapbox(
            lat=[meta["lat"]], lon=[meta["lon"]],
            mode="markers+text",
            marker=dict(size=40, color=fill_col, opacity=0.12),
            text=[zone.replace("_"," ")],
            textfont=dict(size=11, color="#ECF0F6"),
            textposition="middle center",
            name=f"Zone: {zone}",
            hovertemplate=(f"<b>{zone.replace('_',' ')}</b><br>"
                           f"Consumers: {len(z_df)}<br>"
                           f"Flagged: {n_flagged}<br>"
                           f"Zone Risk: {zone_risk}<extra></extra>"),
            showlegend=False,
        ))

# ─── Layer 2: Heatmap density ────────────────────────────────────────────────
if map_mode == "Theft Heatmap":
    df_h = df_f[df_f["prob_theft"] >= 0.4]
    if not df_h.empty:
        fig.add_trace(go.Densitymapbox(
            lat=df_h["latitude"], lon=df_h["longitude"],
            z=df_h["prob_theft"], radius=25,
            colorscale=[[0,"rgba(0,196,140,0.0)"],
                        [0.4,"rgba(255,184,0,0.4)"],
                        [0.7,"rgba(255,71,87,0.5)"],
                        [1.0,"rgba(180,0,40,0.8)"]],
            showscale=True, colorbar=dict(title="Risk Score",thickness=12,
                tickfont=dict(color="#7B8FAB",size=10)),
            name="Theft Density",
        ))

# ─── Layer 3: Consumer scatter pins ──────────────────────────────────────────
if map_mode in ["Consumer Pins","Zone Overlay"]:
    for risk in ["HIGH","MEDIUM","LOW"]:
        if risk not in risk_filter: continue
        df_r = df_f[df_f["risk_level"]==risk]
        if df_r.empty: continue
        fig.add_trace(go.Scattermapbox(
            lat=df_r["latitude"], lon=df_r["longitude"],
            mode="markers",
            marker=dict(
                size=df_r["prob_theft"].apply(lambda p: max(7, p*18)).tolist(),
                color=color_map[risk],
                opacity=0.82 if risk=="HIGH" else (0.65 if risk=="MEDIUM" else 0.45),
            ),
            text=df_r["consumer_id"],
            customdata=df_r[["zone","loss_type","prob_theft"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Zone: %{customdata[0]}<br>"
                "Type: %{customdata[1]}<br>"
                "Risk: %{customdata[2]:.1%}<extra></extra>"
            ),
            name=f"{risk} Risk",
        ))

fig.update_layout(
    mapbox=dict(style="open-street-map", center=dict(lat=12.97, lon=77.59), zoom=10.5),
    paper_bgcolor="rgba(0,0,0,0)",
    height=520,
    legend=dict(orientation="h", y=-0.08, x=0, bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7B8FAB", size=11)),
    margin=dict(t=0, b=0, l=0, r=0),
)
st.plotly_chart(fig, use_container_width=True)

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
top = df_f.nlargest(50,"prob_theft")[
    ["consumer_id","zone","feeder_id","consumer_type","loss_type","prob_theft","confidence","latitude","longitude"]
].copy()
top["prob_theft"] = top["prob_theft"].apply(lambda x: f"{x:.1%}")
top.columns = ["Consumer","Zone","Feeder","Type","Loss","Risk %","Conf","Lat","Lon"]
st.dataframe(top, use_container_width=True, height=300, hide_index=True)

col_e1, col_e2 = st.columns([1,4])
col_e1.download_button("Export CSV", df_f.to_csv(index=False),
                        "vidyut_geo_alerts.csv","text/csv", use_container_width=True)
