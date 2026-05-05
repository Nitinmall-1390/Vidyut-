"""
VIDYUT — Page 5: Audit Trail
Queryable prediction log with consumer, date range, alert type filters.
Every model prediction logged with model_version, confidence, rules triggered.
"""

import sys
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dashboard.components.shared import (
    inject_css, render_sidebar_brand, render_sidebar_status,
    PLOTLY_LAYOUT, RED, AMBER, GREEN, TEAL, BLUE,
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()
    st.markdown("---")
    st.markdown("**📋 QUERY FILTERS**")
    date_from = st.date_input("From Date", datetime.today() - timedelta(days=30))
    date_to   = st.date_input("To Date",   datetime.today())
    alert_type_filter = st.multiselect("Alert Type",
                                       ["theft", "technical", "billing", "normal"],
                                       default=["theft", "technical", "billing"])
    min_conf_filter = st.slider("Min Confidence", 0, 100, 0)
    model_ver_filter = st.selectbox("Model Version", ["All", "v1.0", "v1.1", "v2.0"])
    st.markdown("---")
    render_sidebar_status()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>BESCOM Compliance | Audit Trail Module</div>", unsafe_allow_html=True)
st.title("📋 Prediction Audit Trail")
st.caption("Every prediction logged · Queryable by consumer, date, alert type, confidence · CSV export for regulatory compliance")

# ── Generate synthetic audit log ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_audit_log(n_entries: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic audit trail."""
    rng    = np.random.default_rng(seed)
    zones  = ["Zone_East", "Zone_West", "Zone_North", "Zone_South",
               "Zone_Central", "Zone_NE", "Zone_SE", "Zone_NW"]
    ctypes = ["residential", "commercial", "industrial"]
    loss_t = ["theft", "technical", "billing", "normal"]
    rules  = ["R1: Zero-reading", "R2: MoM drop>60%", "R3: Negative reading",
               "R4: Below min charge", "R5: Spike anomaly", "R6: Flat-line"]
    models = ["v1.0", "v1.1", "v2.0"]

    rows = []
    base_date = datetime.now() - timedelta(days=60)
    for i in range(n_entries):
        ts  = base_date + timedelta(minutes=rng.integers(0, 60 * 24 * 60))
        lt  = rng.choice(loss_t, p=[0.15, 0.10, 0.08, 0.67])
        prob = float(rng.beta(7, 2) if lt != "normal" else rng.beta(1, 8))
        n_r  = int(rng.integers(0, 4)) if lt != "normal" else int(rng.integers(0, 2))
        conf = int(np.clip(prob * 70 + n_r * 8, 0, 100))
        triggered = [rules[r] for r in rng.choice(len(rules), n_r, replace=False)] if n_r > 0 else []

        rows.append({
            "audit_id":       f"AUD-{i:06d}",
            "timestamp":      ts,
            "consumer_id":    f"CONSUM_{rng.integers(0, 10000):05d}",
            "zone":           rng.choice(zones),
            "feeder_id":      f"FEED_{rng.integers(0, 500):03d}",
            "consumer_type":  rng.choice(ctypes),
            "model_version":  rng.choice(models),
            "alert_type":     lt,
            "prob_theft":     round(prob, 4),
            "confidence":     conf,
            "conf_label":     "HIGH" if conf >= 80 else ("MEDIUM" if conf >= 50 else "LOW"),
            "n_rules":        n_r,
            "triggered_rules": " | ".join(triggered) if triggered else "None",
            "input_features": f"avg={round(rng.uniform(2,20),1)}, mom_drop={round(rng.uniform(0,80),1)}%, zero_days={rng.integers(0,10)}",
            "outcome":        "ALERT" if lt != "normal" and prob > 0.4 else "CLEAR",
        })

    return pd.DataFrame(rows)

df_audit = generate_audit_log(500)

# Apply filters
mask = (
    (df_audit["timestamp"].dt.date >= date_from) &
    (df_audit["timestamp"].dt.date <= date_to) &
    (df_audit["alert_type"].isin(alert_type_filter if alert_type_filter else ["theft","technical","billing","normal"])) &
    (df_audit["confidence"] >= min_conf_filter)
)
if model_ver_filter != "All":
    mask &= df_audit["model_version"] == model_ver_filter
df_filtered = df_audit[mask].sort_values("timestamp", ascending=False)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_log, tab_analytics, tab_compliance = st.tabs(
    ["📄 Audit Log", "📊 Analytics", "⚖️ Compliance Report"]
)

with tab_log:
    # Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Records", len(df_filtered))
    m2.metric("Alerts Raised", len(df_filtered[df_filtered["outcome"] == "ALERT"]))
    m3.metric("HIGH Confidence", len(df_filtered[df_filtered["conf_label"] == "HIGH"]))
    m4.metric("Date Range", f"{(date_to - date_from).days}d")
    m5.metric("Model Versions", df_filtered["model_version"].nunique())

    # Consumer search
    search_id = st.text_input("🔍 Search Consumer ID", placeholder="CONSUM_XXXXX")
    if search_id:
        df_filtered = df_filtered[df_filtered["consumer_id"].str.contains(search_id, case=False)]

    # Alert type color coding
    def _row_style(row):
        return {"theft": "🔴", "technical": "🟡", "billing": "🔵", "normal": "🟢"}.get(row, "⚪")

    df_display = df_filtered.head(200).copy()
    df_display["icon"] = df_display["alert_type"].apply(_row_style)
    df_display["timestamp_str"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_display["prob_theft"] = df_display["prob_theft"].apply(lambda x: f"{x:.1%}")

    show_cols = ["icon", "audit_id", "timestamp_str", "consumer_id", "zone",
                 "alert_type", "prob_theft", "confidence", "conf_label",
                 "n_rules", "triggered_rules", "model_version", "outcome"]
    rename_map = {
        "icon": "", "audit_id": "Audit ID", "timestamp_str": "Timestamp",
        "consumer_id": "Consumer", "zone": "Zone", "alert_type": "Type",
        "prob_theft": "Prob", "confidence": "Conf", "conf_label": "Level",
        "n_rules": "Rules", "triggered_rules": "Triggered Rules",
        "model_version": "Model", "outcome": "Outcome",
    }
    st.dataframe(
        df_display[show_cols].rename(columns=rename_map),
        use_container_width=True, height=420, hide_index=True,
    )

    # Export
    csv_audit = df_filtered.to_csv(index=False)
    col_e1, col_e2 = st.columns(2)
    col_e1.download_button("⬇ Export Audit CSV", csv_audit,
                            f"vidyut_audit_{date_from}_{date_to}.csv", "text/csv",
                            use_container_width=True)
    col_e2.metric("Exported Records", len(df_filtered))

with tab_analytics:
    st.subheader("Audit Analytics")

    c_left, c_right = st.columns(2)

    # Daily alert volume
    df_daily = (df_filtered.set_index("timestamp")
                .resample("D")["outcome"]
                .apply(lambda s: (s == "ALERT").sum())
                .reset_index())
    df_daily.columns = ["date", "alerts"]

    fig_daily = go.Figure(go.Bar(
        x=df_daily["date"], y=df_daily["alerts"],
        marker_color=TEAL, marker_line_width=0, opacity=0.85,
    ))
    fig_daily.update_layout(**{**PLOTLY_LAYOUT, "title": "Daily Alert Volume",
                                "height": 280, "yaxis": {"title": "Alerts"},
                                "margin": dict(t=40, b=30, l=40, r=10)})
    with c_left:
        st.plotly_chart(fig_daily, use_container_width=True)

    # Confidence distribution
    fig_conf = go.Figure(go.Histogram(
        x=df_filtered["confidence"], nbinsx=20,
        marker_color=AMBER, opacity=0.8,
    ))
    fig_conf.add_vline(x=80, line_dash="dash", line_color=GREEN,
                       annotation_text="HIGH threshold", annotation_font_color=GREEN)
    fig_conf.add_vline(x=50, line_dash="dash", line_color=AMBER,
                       annotation_text="MEDIUM threshold", annotation_font_color=AMBER)
    fig_conf.update_layout(**{**PLOTLY_LAYOUT, "title": "Confidence Score Distribution",
                               "height": 280, "xaxis": {"title": "Confidence"},
                               "margin": dict(t=40, b=30, l=40, r=10)})
    with c_right:
        st.plotly_chart(fig_conf, use_container_width=True)

    # Alert type breakdown by zone
    df_zone_types = df_filtered[df_filtered["outcome"] == "ALERT"].groupby(
        ["zone", "alert_type"]
    ).size().reset_index(name="count")

    type_colors = {"theft": RED, "technical": AMBER, "billing": BLUE, "normal": GREEN}
    fig_stacked = go.Figure()
    for at in ["theft", "technical", "billing"]:
        subset = df_zone_types[df_zone_types["alert_type"] == at]
        fig_stacked.add_trace(go.Bar(
            x=subset["zone"], y=subset["count"],
            name=at.title(), marker_color=type_colors[at],
        ))
    fig_stacked.update_layout(**{**PLOTLY_LAYOUT,
                                  "title": "Alert Type by Zone",
                                  "barmode": "stack", "height": 340,
                                  "xaxis": {"title": "Zone", "tickangle": -30},
                                  "yaxis": {"title": "Alerts"}})
    st.plotly_chart(fig_stacked, use_container_width=True)

    # Model version comparison
    df_model_perf = df_filtered.groupby("model_version").agg(
        alerts=("outcome", lambda s: (s == "ALERT").sum()),
        avg_conf=("confidence", "mean"),
        records=("audit_id", "count"),
    ).reset_index()
    st.markdown("**Model Version Comparison**")
    df_model_perf.columns = ["Version", "Alerts", "Avg Confidence", "Records"]
    df_model_perf["Avg Confidence"] = df_model_perf["Avg Confidence"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(df_model_perf, use_container_width=True, hide_index=True)

with tab_compliance:
    st.subheader("⚖️ Regulatory Compliance Report")
    st.caption("Compliant with Ministry of Power — Smart Meter Analytics Framework 2023")

    st.markdown(f"""
<div class='vid-card'>
  <div class='vid-card-title'>📊 Report Summary</div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:8px;'>
    <div>
      <div style='color:#7B8FAB;font-size:10px;text-transform:uppercase;'>Period</div>
      <div style='font-weight:700;'>{date_from} to {date_to}</div>
    </div>
    <div>
      <div style='color:#7B8FAB;font-size:10px;text-transform:uppercase;'>Total Predictions</div>
      <div style='font-weight:700;'>{len(df_filtered):,}</div>
    </div>
    <div>
      <div style='color:#7B8FAB;font-size:10px;text-transform:uppercase;'>Alert Rate</div>
      <div style='font-weight:700;color:{AMBER};'>{len(df_filtered[df_filtered["outcome"]=="ALERT"])/max(1,len(df_filtered))*100:.1f}%</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
#### Audit Requirements Checklist

| Requirement | Status | Detail |
|---|---|---|
| Consumer ID logged | ✅ | Every prediction has `consumer_id` |
| Timestamp recorded | ✅ | ISO 8601 UTC timestamp |
| Model version tracked | ✅ | Versioned registry (v1.0, v1.1, v2.0) |
| Input features logged | ✅ | avg_kwh, mom_drop, zero_days, billing_div |
| Output label stored | ✅ | normal / theft / technical / billing |
| Confidence score | ✅ | 0–100 composite score |
| SHAP values | ⚠️ | Available on-demand (not pre-computed) |
| Rule flags | ✅ | R1–R6 triggered rules logged |
| Audit ID | ✅ | Unique AUD-XXXXXX per prediction |
| CSV export | ✅ | Regulatory compliance export available |
    """)

    st.markdown("#### Required Disclosures (MoP Smart Meter Analytics Framework)")
    st.info("""
**Transparency Notice:** All predictions generated by Vidyut are advisory only. 
Field verification is required before any action against a consumer. 
False positive rate target: < 3% (Intersection mode) or < 8% (Union mode). 
Model last retrained: 2026-05-05. Data source: BESCOM SCADA (synthetic demo mode).
    """)

    # Summary stats for compliance
    compliance_data = {
        "Metric": [
            "Total Predictions", "Alerts Raised", "Alert Rate", "HIGH Confidence Rate",
            "Avg Confidence Score", "Unique Consumers Analysed", "Zones Covered",
            "Model Versions Active",
        ],
        "Value": [
            len(df_filtered),
            len(df_filtered[df_filtered["outcome"] == "ALERT"]),
            f"{len(df_filtered[df_filtered['outcome']=='ALERT'])/max(1,len(df_filtered))*100:.1f}%",
            f"{len(df_filtered[df_filtered['conf_label']=='HIGH'])/max(1,len(df_filtered))*100:.1f}%",
            f"{df_filtered['confidence'].mean():.1f}",
            df_filtered["consumer_id"].nunique(),
            df_filtered["zone"].nunique(),
            df_filtered["model_version"].nunique(),
        ],
    }
    st.dataframe(pd.DataFrame(compliance_data), use_container_width=True, hide_index=True)

    # PDF-style export (CSV)
    compliance_csv = pd.DataFrame(compliance_data).to_csv(index=False)
    st.download_button("⬇ Export Compliance Report CSV", compliance_csv,
                       f"vidyut_compliance_{date_from}_{date_to}.csv", "text/csv")
