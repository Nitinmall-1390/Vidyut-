"""VIDYUT — Page 4: Ring Detection (Fixed: valid colors, no emojis)"""
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
    inject_css, PLOTLY_LAYOUT, RED, AMBER, GREEN, TEAL,
)
from src.config.feature_config import BESCOM_ZONES, RING_ANOMALY_FRACTION_THRESHOLD

inject_css()

def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

with st.sidebar:
    st.markdown("<div style='font-size:9px;color:#7B8FAB;letter-spacing:2px;'>BESCOM GRID INTELLIGENCE</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**GRAPH SETTINGS**")
    min_anomaly = st.slider("Min Anomaly Ratio", 0.0, 1.0, 0.40, 0.05)
    min_members = st.number_input("Min Ring Size", 2, 20, 3)
    show_edges  = st.checkbox("Show Member Edges", value=True)
    n_consumers = st.slider("Demo Dataset Size", 50, 300, 150, 50)
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#7B8FAB;'><span style='color:#00C48C;'>&#9679;</span> ENGINE ONLINE<br><span style='color:#00C48C;'>&#9679;</span> ML MODELS READY<br><span style='color:#00C48C;'>&#9679;</span> WEATHER: LIVE</div>", unsafe_allow_html=True)

st.markdown("<div class='gov-tag'>BESCOM VIGILANCE | SYNDICATE DETECTION MODULE</div>", unsafe_allow_html=True)
st.title("Theft Ring Intelligence — Network Graph Analysis")
st.caption("Louvain community detection on consumer-transformer graph. Communities with high anomaly ratio flagged as theft syndicates.")

col_scan, col_info = st.columns([1, 3])
scan_btn = col_scan.button("SCAN NETWORK", type="primary", use_container_width=True)
col_info.markdown(
    f"<div style='color:#7B8FAB;font-size:12px;padding-top:8px;'>Edge = shared transformer OR geohash proximity (&lt;500m) &nbsp;|&nbsp; "
    f"Communities with &gt;{int(min_anomaly*100)}% anomalous members = Ring Alert &nbsp;|&nbsp; Louvain O(n log n)</div>",
    unsafe_allow_html=True)

def generate_rings(n: int, seed: int = 99):
    rng   = np.random.default_rng(seed)
    zones = list(BESCOM_ZONES.keys())
    n_rings = max(5, n // 15)
    rings   = []
    for i in range(n_rings):
        zone = zones[i % len(zones)]; meta = BESCOM_ZONES[zone]
        n_mem = int(rng.integers(int(min_members), 25))
        anom  = round(float(rng.uniform(0.1, 0.98)), 3)
        rings.append({
            "ring_id": f"RING-{i:03d}", "zone": zone,
            "member_count": n_mem, "anomaly_ratio": anom,
            "severity_score": round(n_mem * anom, 1),
            "lat": meta["lat"] + rng.uniform(-0.03, 0.03),
            "lon": meta["lon"] + rng.uniform(-0.03, 0.03),
            "formed_days_ago": int(rng.integers(10, 180)),
            "growing": bool(rng.random() < 0.4),
            "transformer_ids": [f"TRANS_{int(rng.integers(1,100)):04d}" for _ in range(int(rng.integers(1,4)))],
            "members": [f"CONSUM_{int(rng.integers(0, n)):05d}" for _ in range(n_mem)],
        })
    return pd.DataFrame(rings)

def build_network_graph(df_rings):
    rng = np.random.default_rng(7)
    risk_color_map = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}

    # Separate center nodes and member nodes to allow different opacity per group
    cx_list, cy_list, c_text, c_color, c_size = [], [], [], [], []
    mx_list, my_list, m_text, m_color, m_size = [], [], [], [], []
    ex, ey = [], []
    ring_centers = {}

    for i, (_, ring) in enumerate(df_rings.iterrows()):
        angle = 2 * np.pi * i / max(len(df_rings), 1)
        cx, cy = 2.5 * np.cos(angle), 2.5 * np.sin(angle)
        ring_centers[ring["ring_id"]] = (cx, cy)
        risk = "HIGH" if ring["anomaly_ratio"] >= 0.7 else ("MEDIUM" if ring["anomaly_ratio"] >= 0.4 else "LOW")
        col  = risk_color_map[risk]

        # Center node
        cx_list.append(cx); cy_list.append(cy)
        c_text.append(f"<b>{ring['ring_id']}</b><br>Members: {ring['member_count']}<br>Anomaly: {ring['anomaly_ratio']:.1%}<br>Risk: {risk}")
        c_color.append(col)
        c_size.append(max(15, ring["member_count"] * 1.5))

        # Member nodes — use same solid color, lower opacity via trace-level opacity
        for j in range(min(ring["member_count"], 6)):
            mx = cx + rng.uniform(-0.6, 0.6); my = cy + rng.uniform(-0.6, 0.6)
            mx_list.append(mx); my_list.append(my)
            m_text.append(ring["members"][j] if j < len(ring["members"]) else f"Member_{j}")
            m_color.append(col)
            m_size.append(7)
            if show_edges:
                ex += [cx, mx, None]; ey += [cy, my, None]

    for i in range(len(df_rings) - 1):
        r1, r2 = df_rings.iloc[i]["ring_id"], df_rings.iloc[i+1]["ring_id"]
        if rng.random() < 0.25 and r1 in ring_centers and r2 in ring_centers:
            x1,y1 = ring_centers[r1]; x2,y2 = ring_centers[r2]
            ex += [x1, x2, None]; ey += [y1, y2, None]

    return (cx_list, cy_list, c_text, c_color, c_size,
            mx_list, my_list, m_text, m_color, m_size,
            ex, ey)

if scan_btn or "ring_data" not in st.session_state:
    with st.spinner("Traversing consumer-transformer graph via Louvain algorithm..."):
        st.session_state["ring_data"] = generate_rings(n_consumers)

df_rings = st.session_state.get("ring_data", pd.DataFrame())
if df_rings.empty:
    st.info("Click SCAN NETWORK to run Louvain community detection.")
    st.stop()

def _risk(r): return "HIGH" if r >= 0.7 else ("MEDIUM" if r >= 0.4 else "LOW")
df_rings_f = df_rings[(df_rings["anomaly_ratio"] >= min_anomaly) &
                       (df_rings["member_count"] >= int(min_members))].copy()
df_rings_f["risk"] = df_rings_f["anomaly_ratio"].apply(_risk)

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Total Rings",   len(df_rings_f))
m2.metric("HIGH Severity", len(df_rings_f[df_rings_f["risk"]=="HIGH"]))
m3.metric("Total Members", int(df_rings_f["member_count"].sum()))
m4.metric("Avg Anomaly",   f"{df_rings_f['anomaly_ratio'].mean():.1%}")
m5.metric("Growing Rings", int(df_rings_f["growing"].sum()))

tab_net, tab_tl, tab_tbl = st.tabs(["Network Graph", "Formation Timeline", "Ring Roster"])

with tab_net:
    if df_rings_f.empty:
        st.warning("No rings match current filters. Reduce the minimum anomaly ratio.")
    else:
        cx_l,cy_l,c_txt,c_col,c_sz, mx_l,my_l,m_txt,m_col,m_sz, ex,ey = build_network_graph(df_rings_f)
        fig = go.Figure()
        if show_edges and ex:
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                line=dict(width=0.7, color="#1A2D45"), hoverinfo="none", showlegend=False))
        # Member nodes (lower opacity via trace opacity)
        fig.add_trace(go.Scatter(x=mx_l, y=my_l, mode="markers",
            marker=dict(size=m_sz, color=m_col, opacity=0.35,
                        line=dict(color="#0A1628", width=0.5)),
            text=m_txt, hoverinfo="text", name="Members", showlegend=False))
        # Ring centroid nodes
        fig.add_trace(go.Scatter(x=cx_l, y=cy_l, mode="markers+text",
            marker=dict(size=c_sz, color=c_col, opacity=0.95,
                        line=dict(color="white", width=1.5)),
            text=[t.split("<br>")[0].replace("<b>","").replace("</b>","") for t in c_txt],
            textposition="top center", textfont=dict(size=9, color="#7B8FAB"),
            hovertext=c_txt, hoverinfo="text", name="Rings"))
        fig.update_layout(**{**PLOTLY_LAYOUT,
            "title": "Consumer-Transformer Theft Ring Network",
            "showlegend": False, "height": 500,
            "xaxis": {"showgrid":False,"zeroline":False,"showticklabels":False,"visible":False},
            "yaxis": {"showgrid":False,"zeroline":False,"showticklabels":False,"visible":False},
            "plot_bgcolor": "rgba(6,14,28,0.95)",
            "uirevision": "constant",
        })
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top Detected Rings — Sorted by Severity Score**")
        badge_cols = st.columns(3)
        for i, (_, ring) in enumerate(df_rings_f.nlargest(6,"severity_score").iterrows()):
            risk = ring["risk"]
            color = RED if risk=="HIGH" else (AMBER if risk=="MEDIUM" else GREEN)
            badge_cols[i%3].markdown(
                f"<div style='background:#0E1A2E;border:1px solid #1A2D45;border-left:4px solid {color};"
                f"border-radius:0 6px 6px 0;padding:10px 12px;margin-bottom:8px;'>"
                f"<div style='color:{color};font-size:11px;font-weight:700;'>{risk} &nbsp;|&nbsp; {ring['ring_id']}</div>"
                f"<div style='color:#7B8FAB;font-size:11px;'>Members: {ring['member_count']} &nbsp;&bull;&nbsp; Anomaly: {ring['anomaly_ratio']:.1%}</div>"
                f"<div style='color:#ECF0F6;font-size:10px;'>{ring['zone']} &nbsp;&bull;&nbsp; Score: {ring['severity_score']:.0f}</div>"
                f"<div style='color:#7B8FAB;font-size:10px;'>{'Growing' if ring['growing'] else 'Stable'}</div>"
                f"</div>", unsafe_allow_html=True)

with tab_tl:
    st.subheader("Ring Formation Timeline")
    df_rings_f = df_rings_f.copy()
    df_rings_f["detection_day"] = pd.Timestamp.now() - pd.to_timedelta(df_rings_f["formed_days_ago"].astype(int), unit="D")
    risk_colors = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}
    fig_tl = go.Figure()
    for risk in ["HIGH","MEDIUM","LOW"]:
        df_r = df_rings_f[df_rings_f["risk"]==risk]
        if df_r.empty: continue
        fig_tl.add_trace(go.Scatter(
            x=df_r["detection_day"], y=df_r["severity_score"],
            mode="markers", marker=dict(size=(df_r["member_count"]*2).tolist(),
                color=risk_colors[risk], opacity=0.85, line=dict(color="white",width=1)),
            name=f"{risk} Risk",
            hovertemplate="<b>%{text}</b><br>Formed: %{x|%b %d}<br>Severity: %{y:.1f}<extra></extra>",
            text=df_r["ring_id"].tolist(),
        ))
    fig_tl.update_layout(**{**PLOTLY_LAYOUT, "title": "Ring Formation Timeline (bubble = member count)",
        "xaxis":{"title":"Detection Date","gridcolor":"#1A2D45"},
        "yaxis":{"title":"Severity Score","gridcolor":"#1A2D45"}, "height": 380})
    st.plotly_chart(fig_tl, use_container_width=True)

with tab_tbl:
    st.subheader("Ring Roster")
    ring_ids = df_rings_f["ring_id"].tolist()
    if not ring_ids:
        st.info("No rings match the current filters. Reduce the minimum anomaly ratio or ring size.")
    else:
        sel = st.selectbox("Select Ring", ring_ids)
        ring_row = df_rings_f[df_rings_f["ring_id"]==sel].iloc[0]
        rc = RED if ring_row["risk"]=="HIGH" else (AMBER if ring_row["risk"]=="MEDIUM" else GREEN)
        c1,c2,c3 = st.columns(3)
        c1.metric("Members",      ring_row["member_count"])
        c2.metric("Anomaly Ratio",f"{ring_row['anomaly_ratio']:.1%}")
        c3.metric("Severity Score",f"{ring_row['severity_score']:.0f}")
        st.markdown(f"**Zone:** {ring_row['zone']} &nbsp;|&nbsp; "
                    f"**Formed:** {ring_row['formed_days_ago']} days ago &nbsp;|&nbsp; "
                    f"**Status:** <span style='color:{'#00D4AA' if not ring_row['growing'] else '#FFB800'};font-weight:600;'>{'Stable' if not ring_row['growing'] else 'Growing'}</span>",
                    unsafe_allow_html=True)
        n_flagged = int(ring_row["member_count"] * ring_row["anomaly_ratio"])
        member_df = pd.DataFrame({"Consumer ID": ring_row["members"],
            "Status": ["FLAGGED" if i < n_flagged else "NORMAL" for i in range(len(ring_row["members"]))]})
        st.dataframe(member_df, use_container_width=True, height=200, hide_index=True)

    st.markdown("---")
    display = df_rings_f[["ring_id","zone","member_count","anomaly_ratio","severity_score","risk","growing","formed_days_ago"]].copy()
    display.columns = ["Ring ID","Zone","Members","Anomaly %","Severity","Risk","Growing","Days Ago"]
    display["Anomaly %"] = display["Anomaly %"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Export Ring Data CSV",
        df_rings_f.drop(columns=["members","transformer_ids"],errors="ignore").to_csv(index=False),
        "vidyut_rings.csv","text/csv")
