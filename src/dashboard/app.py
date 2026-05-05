"""
VIDYUT Intelligence Engine — Entry Point
Minimal app.py. All pages auto-discovered from pages/ directory.
"""

import pathlib
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Vidyut — BESCOM Grid Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_css_path = pathlib.Path(__file__).parent / "assets" / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Shared sidebar brand (shown on home page) ─────────────────────────────────
import sys
_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.components.shared import render_sidebar_brand, render_sidebar_status

with st.sidebar:
    render_sidebar_brand()
    st.markdown("---")
    render_sidebar_status(weather_ok=False)

# ── Home / Landing ────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>Ministry of Power · BESCOM Karnataka</div>", unsafe_allow_html=True)
st.title("Vidyut — Smart Grid Intelligence Engine")

st.markdown("""
<p style='color:#7B8FAB;font-size:14px;line-height:1.8;max-width:760px;'>
An end-to-end AI platform for BESCOM's 11kV distribution network — combining
demand forecasting, anomaly detection, theft ring analysis, and geospatial intelligence
into a single real-time decision dashboard.
</p>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class='vid-card'>
        <div class='vid-card-title'>⚡ Demand Forecast</div>
        <p style='color:#7B8FAB;font-size:12px;'>Multi-horizon ensemble prediction per 11kV feeder. Prophet + LightGBM with weather integration.</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class='vid-card'>
        <div class='vid-card-title'>🚨 Theft Guard</div>
        <p style='color:#7B8FAB;font-size:12px;'>LSTM AE → Isolation Forest → XGBoost 3-class pipeline with SHAP explainability.</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class='vid-card'>
        <div class='vid-card-title'>🗺️ Geospatial Map</div>
        <p style='color:#7B8FAB;font-size:12px;'>Bangalore-wide consumer anomaly heatmap with feeder zone overlays.</p>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class='vid-card'>
        <div class='vid-card-title'>🕸️ Ring Detection</div>
        <p style='color:#7B8FAB;font-size:12px;'>Louvain community detection on consumer-transformer graph identifies theft syndicates.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#1A2D45;font-size:11px;font-weight:600;letter-spacing:1px;'>
👈 SELECT A MODULE FROM THE SIDEBAR TO BEGIN
</div>""", unsafe_allow_html=True)
