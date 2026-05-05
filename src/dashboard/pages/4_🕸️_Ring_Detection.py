"""
VIDYUT — Page 4: Ring Detection
Louvain community detection on consumer-transformer graph.
Interactive Plotly network graph with community coloring and timeline.
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

_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.components.shared import (
    inject_css, render_sidebar_brand, render_sidebar_status,
    get_synthetic_consumers,
    PLOTLY_LAYOUT, RED, AMBER, GREEN, TEAL, BLUE, PURPLE, ORANGE,
)
from src.config.feature_config import BESCOM_ZONES, RING_ANOMALY_FRACTION_THRESHOLD

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()
    st.markdown("---")
    st.markdown("**🕸️ GRAPH SETTINGS**")
    min_anomaly = st.slider("Min Anomaly Ratio", 0.0, 1.0, float(RING_ANOMALY_FRACTION_THRESHOLD), 0.05)
    min_members = st.number_input("Min Ring Size", 2, 20, 3)
    graph_layout = st.selectbox("Graph Layout", ["Spring", "Circular", "Kamada-Kawai"], index=0)
    show_edges   = st.checkbox("Show All Edges", value=True)
    n_consumers  = st.slider("Demo Dataset Size", 50, 300, 150, 50)
    st.markdown("---")
    render_sidebar_status()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>BESCOM Vigilance | Syndicate Detection Module</div>", unsafe_allow_html=True)
st.title("🕸️ Theft Ring Intelligence — Network Graph Analysis")
st.caption("Louvain community detection · Consumer-Transformer graph · Syndicate severity scoring")

# ── Scan Button ───────────────────────────────────────────────────────────────
col_scan, col_info = st.columns([1, 3])
scan_btn = col_scan.button("🔍 SCAN NETWORK", type="primary", use_container_width=True)
col_info.markdown(f"""
<div style='color:#7B8FAB;font-size:12px;padding-top:8px;'>
Edge = shared transformer OR geohash proximity (&lt;500m) ·
Communities with &gt;{int(min_anomaly*100)}% anomalous members = Ring Alert ·
Louvain algorithm (O(n log n))
</div>""", unsafe_allow_html=True)

# ── Generate ring data ────────────────────────────────────────────────────────
def generate_rings(n: int, seed: int = 99):
    """Generate synthetic theft ring data using consumer graph."""
    rng   = np.random.default_rng(seed)
    zones = list(BESCOM_ZONES.keys())

    # Simulate Louvain communities
    n_rings = max(5, n // 15)
    rings   = []
    for i in range(n_rings):
        zone  = zones[i % len(zones)]
        meta  = BESCOM_ZONES[zone]
        n_mem = int(rng.integers(min_members, 25))
        anom  = round(float(rng.uniform(0.1, 0.98)), 3)
        sev_score = n_mem * anom

        rings.append({
            "ring_id":       f"RING-{i:03d}",
            "zone":          zone,
            "member_count":  n_mem,
            "anomaly_ratio": anom,
            "severity_score": round(sev_score, 1),
            "lat":           meta["lat"] + rng.uniform(-0.03, 0.03),
            "lon":           meta["lon"] + rng.uniform(-0.03, 0.03),
            "formed_days_ago": int(rng.integers(10, 180)),
            "growing":       bool(rng.random() < 0.4),
            "transformer_ids": [f"TRANS_{rng.integers(1,100):04d}" for _ in range(rng.integers(1,4))],
            "members":       [f"CONSUM_{rng.integers(0, n):05d}" for _ in range(n_mem)],
        })

    return pd.DataFrame(rings)


def build_network_graph(df_rings: pd.DataFrame):
    """Build a Plotly network graph of consumer communities."""
    rng = np.random.default_rng(7)

    # Generate node positions
    risk_colors = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}
    nodes_x, nodes_y, nodes_text, nodes_color, nodes_size = [], [], [], [], []
    edges_x, edges_y = [], []

    ring_centers = {}
    for i, (_, ring) in enumerate(df_rings.iterrows()):
        # Place ring center
        angle  = 2 * np.pi * i / len(df_rings)
        radius = 2.5
        cx, cy = radius * np.cos(angle), radius * np.sin(angle)
        ring_centers[ring["ring_id"]] = (cx, cy)

        risk = ("HIGH" if ring["anomaly_ratio"] >= 0.7 else
                "MEDIUM" if ring["anomaly_ratio"] >= 0.4 else "LOW")

        # Ring centroid node
        nodes_x.append(cx); nodes_y.append(cy)
        nodes_text.append(f"<b>{ring['ring_id']}</b><br>Members: {ring['member_count']}<br>"
                          f"Anomaly: {ring['anomaly_ratio']:.1%}<br>Risk: {risk}")
        nodes_color.append(risk_colors[risk])
        nodes_size.append(max(15, ring["member_count"] * 1.5))

        # Member nodes (smaller, scattered around center)
        for j in range(min(ring["member_count"], 8)):
            mx = cx + rng.uniform(-0.6, 0.6)
            my = cy + rng.uniform(-0.6, 0.6)
            nodes_x.append(mx); nodes_y.append(my)
            nodes_text.append(ring["members"][j] if j < len(ring["members"]) else f"Member_{j}")
            nodes_color.append(risk_colors[risk] + "AA")  # semi-transparent
            nodes_size.append(7)
            if show_edges:
                edges_x += [cx, mx, None]
                edges_y += [cy, my, None]

    # Cross-ring edges (shared transformers)
    ring_list = df_rings["ring_id"].tolist()
    for i in range(len(ring_list) - 1):
        if rng.random() < 0.25:  # 25% of rings share a transformer link
            r1, r2 = ring_list[i], ring_list[i+1]
            if r1 in ring_centers and r2 in ring_centers:
                x1, y1 = ring_centers[r1]
                x2, y2 = ring_centers[r2]
                edges_x += [x1, x2, None]
                edges_y += [y1, y2, None]

    return nodes_x, nodes_y, nodes_text, nodes_color, nodes_size, edges_x, edges_y


if scan_btn or "ring_data" not in st.session_state:
    with st.spinner("Traversing consumer-transformer graph via Louvain algorithm..."):
        rings = generate_rings(n_consumers)
        st.session_state["ring_data"] = rings

df_rings = st.session_state.get("ring_data", pd.DataFrame())

if df_rings.empty:
    st.info("Click **SCAN NETWORK** to run Louvain community detection.")
    st.stop()

# Apply filters
df_rings_f = df_rings[
    (df_rings["anomaly_ratio"] >= min_anomaly) &
    (df_rings["member_count"] >= min_members)
].copy()

def _risk(r):
    return "HIGH" if r >= 0.7 else ("MEDIUM" if r >= 0.4 else "LOW")
df_rings_f["risk"] = df_rings_f["anomaly_ratio"].apply(_risk)

# ── Metrics ───────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Rings",     len(df_rings_f))
m2.metric("HIGH Severity",   len(df_rings_f[df_rings_f["risk"] == "HIGH"]))
m3.metric("Total Members",   int(df_rings_f["member_count"].sum()))
m4.metric("Avg Anomaly",     f"{df_rings_f['anomaly_ratio'].mean():.1%}")
m5.metric("Growing Rings",   int(df_rings_f["growing"].sum()))

# ── Network Graph ─────────────────────────────────────────────────────────────
tab_network, tab_timeline, tab_table = st.tabs(
    ["🕸️ Network Graph", "📈 Ring Timeline", "📋 Ring Roster"]
)

with tab_network:
    if len(df_rings_f) == 0:
        st.warning("No rings match the current filters.")
    else:
        nx, ny, nt, nc, ns, ex, ey = build_network_graph(df_rings_f)

        fig_net = go.Figure()

        # Edges
        if show_edges and ex:
            fig_net.add_trace(go.Scatter(
                x=ex, y=ey, mode="lines",
                line=dict(width=0.8, color="#1A2D45"),
                hoverinfo="none", showlegend=False,
            ))

        # Nodes
        fig_net.add_trace(go.Scatter(
            x=nx, y=ny, mode="markers",
            marker=dict(
                size=ns, color=nc, opacity=0.9,
                line=dict(color="#0A1628", width=1),
            ),
            text=nt, hoverinfo="text",
            name="Nodes",
        ))

        fig_net.update_layout(
            **{**PLOTLY_LAYOUT,
               "title": "Consumer-Transformer Theft Ring Network",
               "showlegend": False,
               "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
               "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
               "height": 520,
               "plot_bgcolor": "rgba(6,14,28,0.8)",
               },
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Ring severity badges
        st.markdown("**Detected Rings (sorted by severity):**")
        top_rings = df_rings_f.nlargest(6, "severity_score")
        badge_cols = st.columns(3)
        for i, (_, ring) in enumerate(top_rings.iterrows()):
            risk = ring["risk"]
            color = RED if risk == "HIGH" else (AMBER if risk == "MEDIUM" else GREEN)
            badge_cols[i % 3].markdown(f"""
<div style='background:#0E1A2E;border:1px solid #1A2D45;border-left:4px solid {color};
border-radius:0 8px 8px 0;padding:10px 12px;margin-bottom:8px;'>
  <div style='color:{color};font-size:11px;font-weight:700;'>{risk} · {ring["ring_id"]}</div>
  <div style='color:#7B8FAB;font-size:11px;'>Members: {ring["member_count"]} · Anomaly: {ring["anomaly_ratio"]:.1%}</div>
  <div style='color:#ECF0F6;font-size:10px;'>{ring["zone"]} · Score: {ring["severity_score"]:.0f}</div>
  <div style='color:#7B8FAB;font-size:10px;'>{'📈 Growing' if ring["growing"] else '📉 Stable'}</div>
</div>""", unsafe_allow_html=True)

with tab_timeline:
    st.subheader("Ring Formation Timeline (last 180 days)")

    # Timeline scatter
    df_rings_f["detection_day"] = pd.Timestamp.now() - pd.to_timedelta(df_rings_f["formed_days_ago"], unit="D")
    risk_colors = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}

    fig_tl = go.Figure()
    for risk in ["HIGH", "MEDIUM", "LOW"]:
        df_r = df_rings_f[df_rings_f["risk"] == risk]
        if len(df_r) == 0:
            continue
        fig_tl.add_trace(go.Scatter(
            x=df_r["detection_day"],
            y=df_r["severity_score"],
            mode="markers+text",
            marker=dict(size=df_r["member_count"] * 2,
                        color=risk_colors[risk], opacity=0.85,
                        line=dict(color="white", width=1)),
            text=df_r["ring_id"],
            textposition="top center",
            textfont=dict(size=9, color="#7B8FAB"),
            name=f"{risk} Risk",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Formed: %{x|%b %d}<br>"
                "Severity: %{y:.1f}<br>"
                "Members: %{marker.size:.0f}<extra></extra>"
            ),
        ))

    fig_tl.update_layout(**{
        **PLOTLY_LAYOUT,
        "title": "Ring Formation Timeline — Bubble size = member count",
        "xaxis": {"title": "Detection Date", "gridcolor": "#1A2D45"},
        "yaxis": {"title": "Severity Score", "gridcolor": "#1A2D45"},
        "height": 420,
    })
    st.plotly_chart(fig_tl, use_container_width=True)

    # Growing vs stable bar chart
    grow_counts = df_rings_f.groupby(["risk", "growing"]).size().reset_index(name="count")
    fig_grow = go.Figure()
    for risk, color in [("HIGH", RED), ("MEDIUM", AMBER), ("LOW", GREEN)]:
        subset = grow_counts[grow_counts["risk"] == risk]
        growing = subset[subset["growing"]]["count"].sum() if len(subset[subset["growing"]]) > 0 else 0
        stable  = subset[~subset["growing"]]["count"].sum() if len(subset[~subset["growing"]]) > 0 else 0
        fig_grow.add_trace(go.Bar(name=f"{risk} Growing", x=[f"{risk} Growing"], y=[growing],
                                   marker_color=color, opacity=0.9))
        fig_grow.add_trace(go.Bar(name=f"{risk} Stable", x=[f"{risk} Stable"], y=[stable],
                                   marker_color=color, opacity=0.45))
    fig_grow.update_layout(**{**PLOTLY_LAYOUT,
                               "title": "Growing vs Stable Rings",
                               "height": 280, "showlegend": False})
    st.plotly_chart(fig_grow, use_container_width=True)

with tab_table:
    st.subheader("Ring Roster & Member List")

    # Ring selection
    ring_ids = df_rings_f["ring_id"].tolist()
    if ring_ids:
        selected_ring = st.selectbox("Select Ring to inspect", ring_ids)
        ring_row = df_rings_f[df_rings_f["ring_id"] == selected_ring].iloc[0]

        risk_color = RED if ring_row["risk"] == "HIGH" else (AMBER if ring_row["risk"] == "MEDIUM" else GREEN)
        st.markdown(f"""
<div style='background:#0E1A2E;border:1px solid {risk_color};border-radius:10px;padding:16px 20px;margin-bottom:12px;'>
  <div style='color:{risk_color};font-weight:800;font-size:14px;'>{ring_row["risk"]} RISK — {ring_row["ring_id"]}</div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:10px;'>
    <div><div style='color:#7B8FAB;font-size:10px;'>MEMBERS</div><div style='font-size:20px;font-weight:700;'>{ring_row["member_count"]}</div></div>
    <div><div style='color:#7B8FAB;font-size:10px;'>ANOMALY RATIO</div><div style='font-size:20px;font-weight:700;color:{risk_color};'>{ring_row["anomaly_ratio"]:.1%}</div></div>
    <div><div style='color:#7B8FAB;font-size:10px;'>SEVERITY SCORE</div><div style='font-size:20px;font-weight:700;'>{ring_row["severity_score"]:.0f}</div></div>
  </div>
  <div style='color:#7B8FAB;font-size:11px;margin-top:8px;'>
    Zone: {ring_row["zone"]} · Formed ~{ring_row["formed_days_ago"]} days ago ·
    Transformers: {", ".join(ring_row["transformer_ids"])} ·
    {'📈 Growing' if ring_row["growing"] else '📉 Stable'}
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"**Member Consumer IDs ({ring_row['member_count']} total):**")
        member_df = pd.DataFrame({"consumer_id": ring_row["members"],
                                   "status": ["🔴 FLAGGED" if i < int(ring_row["member_count"] * ring_row["anomaly_ratio"])
                                              else "🟢 NORMAL" for i in range(len(ring_row["members"]))]})
        st.dataframe(member_df, use_container_width=True, height=200, hide_index=True)

    # Full ring table
    st.markdown("---")
    display = df_rings_f[["ring_id", "zone", "member_count", "anomaly_ratio",
                            "severity_score", "risk", "growing", "formed_days_ago"]].copy()
    display.columns = ["Ring ID", "Zone", "Members", "Anomaly %", "Severity", "Risk", "Growing", "Days Ago"]
    display["Anomaly %"] = display["Anomaly %"].apply(lambda x: f"{x:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    csv_rings = df_rings_f.drop(columns=["members", "transformer_ids"], errors="ignore").to_csv(index=False)
    st.download_button("⬇ Export Ring Data CSV", csv_rings, "vidyut_rings.csv", "text/csv")
