"""VIDYUT Intelligence Engine — Entry Point with clean navigation"""
import pathlib, sys

import streamlit as st

st.set_page_config(
    page_title="VIDYUT | BESCOM Grid Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_css = pathlib.Path(__file__).parent / "assets" / "style.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

# ── Navigation (no emojis, no "app" label) ────────────────────────────────────
try:
    pages = st.navigation(
        {
            "GRID INTELLIGENCE": [
                st.Page("pages/1_Demand_Forecast.py",  title="Demand Forecast"),
                st.Page("pages/2_Theft_Alerts.py",     title="Theft Alerts"),
                st.Page("pages/3_Geospatial_Map.py",   title="Geospatial Map"),
                st.Page("pages/4_Ring_Detection.py",   title="Ring Detection"),
                st.Page("pages/5_Audit_Trail.py",      title="Audit Trail"),
            ]
        },
        position="sidebar",
    )
    # Sidebar brand — appears above nav links
    with st.sidebar:
        st.markdown(
            "<div style='padding:12px 0 14px 0;border-bottom:1px solid #1A2D45;margin-bottom:8px;'>"
            "<div style='font-weight:900;font-size:18px;color:#ECF0F6;letter-spacing:-0.5px;'>VIDYUT</div>"
            "<div style='font-size:9px;color:#7B8FAB;letter-spacing:2px;margin-top:3px;text-transform:uppercase;'>"
            "BESCOM &nbsp;&bull;&nbsp; Grid Intelligence Engine</div>"
            "</div>", unsafe_allow_html=True)

    pages.run()

except AttributeError:
    # Fallback for Streamlit < 1.31 — auto-discovery from pages/
    with st.sidebar:
        st.markdown(
            "<div style='padding:10px 0 12px 0;border-bottom:1px solid #1A2D45;margin-bottom:8px;'>"
            "<div style='font-weight:900;font-size:18px;color:#ECF0F6;'>VIDYUT</div>"
            "<div style='font-size:9px;color:#7B8FAB;letter-spacing:2px;'>BESCOM GRID INTELLIGENCE</div>"
            "</div>", unsafe_allow_html=True)

    # Landing page content (shown when app.py is selected)
    st.markdown("<div class='gov-tag'>MINISTRY OF POWER | BESCOM KARNATAKA</div>", unsafe_allow_html=True)
    st.title("VIDYUT — Smart Grid Intelligence Engine")
    st.caption("Select a module from the sidebar to begin.")

    c1,c2,c3,c4 = st.columns(4)
    for col, title, desc in [
        (c1,"Demand Forecast","Multi-horizon ensemble prediction per 11kV feeder. Prophet + LightGBM with weather integration."),
        (c2,"Theft Alerts","LSTM AE + Isolation Forest + XGBoost 3-class pipeline with SHAP explainability."),
        (c3,"Geospatial Map","Bangalore-wide consumer anomaly heatmap with feeder zone overlays."),
        (c4,"Ring Detection","Louvain community detection on consumer-transformer graph identifies syndicates."),
    ]:
        col.markdown(
            f"<div class='vid-card'>"
            f"<div class='vid-card-title'>{title}</div>"
            f"<p style='color:#7B8FAB;font-size:12px;'>{desc}</p>"
            f"</div>", unsafe_allow_html=True)
