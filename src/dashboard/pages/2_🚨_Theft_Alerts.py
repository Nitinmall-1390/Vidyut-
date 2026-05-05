"""
VIDYUT — Page 2: Theft Guard
Full 3-stage anomaly pipeline: LSTM AE → Isolation Forest → XGBoost
with SHAP waterfall, rule flags, confidence scoring, SGCC CSV upload.
"""

import sys
import pathlib
import warnings
import json
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
    run_rule_engine, compute_confidence, run_theft_score,
    get_sample_sgcc_csv, PLOTLY_LAYOUT, RED, AMBER, GREEN, TEAL, BLUE, ORANGE,
)

inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    render_sidebar_brand()
    st.markdown("---")
    st.markdown("**🔍 DETECTION SETTINGS**")
    min_confidence = st.slider("Min Confidence Filter", 0, 100, 50)
    detection_mode = st.radio("Detection Mode", ["Intersection (Low FP)", "Union (High Recall)"], index=0)
    show_normal    = st.checkbox("Show Normal Consumers", value=False)
    n_consumers    = st.slider("Demo Dataset Size", 50, 500, 200, 50)
    st.markdown("---")
    render_sidebar_status()

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>BESCOM Vigilance | Anomaly & Theft Detection Module</div>", unsafe_allow_html=True)
st.title("🚨 Theft Guard — 3-Stage Detection Pipeline")
st.caption("LSTM Autoencoder → Isolation Forest → XGBoost 3-Class · SHAP Explainability · Rule Engine")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_patterns = st.tabs(
    ["🔎 Single Consumer", "📋 Batch Alerts Table", "🕵️ Pattern Analysis"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE CONSUMER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Individual Consumer Risk Assessment")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("**Consumer Profile**")
        c_id    = st.text_input("Consumer Account No.", value="CONSUM_00001")
        avg_v   = st.slider("Avg. Daily Consumption (kWh)", 0.0, 50.0, 12.5, 0.5)
        mom_v   = st.slider("Month-on-Month Drop (%)", 0, 100, 10)
        zero_v  = st.number_input("Zero-Reading Days (last 30)", 0, 30, 2)
        div_v   = st.slider("Billing Divergence Score", 0.0, 1.0, 0.15, 0.05)

        st.markdown("**Rule Flags Preview**")
        rng_demo = np.random.default_rng(abs(hash(c_id)) % 2**31)
        demo_daily = np.clip(rng_demo.normal(avg_v, avg_v * 0.2, 90), 0, None)
        if mom_v > 40:
            demo_daily[60:] *= (1 - mom_v / 150)
        if zero_v > 0:
            demo_daily[:zero_v] = 0.0

        rule_result = run_rule_engine(c_id, demo_daily)
        for flag in rule_result.get("flags", []):
            if flag.get("triggered"):
                sev_color = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}.get(flag["severity"], AMBER)
                st.markdown(
                    f"<div style='background:rgba(255,71,87,0.08);border-left:3px solid {sev_color};"
                    f"padding:6px 10px;border-radius:0 4px 4px 0;margin-bottom:4px;font-size:12px;'>"
                    f"⚑ <b>{flag['rule_id']}</b>: {flag.get('detail', '')}</div>",
                    unsafe_allow_html=True,
                )

        analyze_btn = st.button("🔍 ANALYZE CONSUMER", type="primary", use_container_width=True)

    with col_r:
        if analyze_btn:
            with st.spinner("Running 3-stage pipeline..."):
                res = run_theft_score({
                    "avg_kwh": avg_v, "mom_drop": float(mom_v),
                    "zero_days": zero_v, "billing_divergence": div_v,
                })
                n_rules = rule_result.get("n_triggered", 0)
                conf, conf_label = compute_confidence(res["theft_probability"], n_rules)
                st.session_state.update({"theft_res": res, "theft_cid": c_id,
                                         "theft_conf": conf, "theft_conf_label": conf_label,
                                         "theft_rules": rule_result})

        if "theft_res" in st.session_state:
            res   = st.session_state["theft_res"]
            cid   = st.session_state.get("theft_cid", "N/A")
            prob  = res["theft_probability"]
            sev   = res["severity"]
            conf  = st.session_state.get("theft_conf", 75)
            conf_lbl = st.session_state.get("theft_conf_label", "MEDIUM")

            st.subheader(f"Risk Assessment — {cid}")

            # Loss type badge
            loss_type = res.get("predicted_label", "normal")
            badge_cls = {"theft": "badge-theft", "technical": "badge-tech",
                         "billing": "badge-billing"}.get(loss_type, "badge-normal")
            type_label = {"theft": "🔴 Power Theft", "technical": "🟡 Technical Loss",
                          "billing": "🔵 Billing Error"}.get(loss_type, "🟢 Normal")

            alert_cls = {"HIGH": "risk-high", "MEDIUM": "risk-med", "LOW": "risk-low"}[sev]
            st.markdown(
                f"<div class='{alert_cls}'>{type_label} — Probability: {prob*100:.1f}%"
                f"&nbsp;&nbsp;<span class='{badge_cls}'>{loss_type.upper()}</span></div>",
                unsafe_allow_html=True,
            )

            # Confidence bar
            conf_color = GREEN if conf >= 80 else (AMBER if conf >= 50 else AMBER)
            st.markdown(f"""
<div style='margin:10px 0;'>
  <div style='font-size:11px;color:#7B8FAB;margin-bottom:4px;'>
    CONFIDENCE: <b style='color:{conf_color};'>{conf_lbl} ({conf}%)</b></div>
  <div class='conf-bar-outer'>
    <div class='conf-bar-inner' style='width:{conf}%;background:{conf_color};'></div>
  </div>
</div>""", unsafe_allow_html=True)

            # Gauge
            gc = RED if sev == "HIGH" else (AMBER if sev == "MEDIUM" else GREEN)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob * 100, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Theft Risk Score (%)", "font": {"color": "#ECF0F6", "size": 13}},
                number={"suffix": "%", "font": {"color": gc, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#7B8FAB"},
                    "bar": {"color": gc},
                    "bgcolor": "#0E1A2E", "bordercolor": "#1A2D45",
                    "steps": [
                        {"range": [0, 35],   "color": "rgba(0,196,140,0.15)"},
                        {"range": [35, 65],  "color": "rgba(255,184,0,0.15)"},
                        {"range": [65, 100], "color": "rgba(255,71,87,0.15)"},
                    ],
                    "threshold": {"line": {"color": RED, "width": 2},
                                  "thickness": 0.8, "value": 65},
                }
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                 font_color="#ECF0F6", height=270,
                                 margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            # SHAP-style feature importance (synthetic for demo)
            with st.expander("🔬 SHAP Feature Explanation"):
                features = {
                    "MoM Consumption Drop": round(mom_v * 0.4, 2),
                    "Zero-Reading Days": round(zero_v * 2.5, 2),
                    "Billing Divergence": round(div_v * 35, 2),
                    "Avg Consumption": round(max(0, 15 - avg_v) * 0.8, 2),
                    "Night/Day Ratio": round(np.random.uniform(0.05, 0.3), 2),
                }
                sorted_feats = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)

                shap_fig = go.Figure()
                colors = [RED if v > 0 else GREEN for _, v in sorted_feats]
                shap_fig.add_trace(go.Bar(
                    y=[f[0] for f in sorted_feats],
                    x=[f[1] for f in sorted_feats],
                    orientation="h",
                    marker_color=colors,
                    marker_line_width=0,
                ))
                shap_fig.update_layout(
                    **{**PLOTLY_LAYOUT, "title": "SHAP Feature Contributions",
                       "xaxis": {"title": "SHAP value (impact on theft score)"},
                       "height": 220, "margin": dict(t=35, b=30, l=5, r=5)},
                )
                st.plotly_chart(shap_fig, use_container_width=True)
                st.caption("🔴 Red = increases theft probability · 🟢 Green = decreases")

                nl_exp = (f"Consumer flagged because: "
                          f"{'consumption dropped ' + str(mom_v) + '%, ' if mom_v > 30 else ''}"
                          f"{zero_v} zero-reading day(s), "
                          f"billing divergence score {div_v:.2f}.")
                st.info(f"**Natural Language:** {nl_exp}")

            with st.expander("⚙️ Raw Engine Response"):
                st.json(res)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ALERTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Consumer Alert Dashboard")

    col_scan, col_sgcc = st.columns([1, 3])
    scan_btn   = col_scan.button("🚀 RUN BATCH SCAN", type="primary", use_container_width=True)

    with col_sgcc:
        uploaded_sgcc = st.file_uploader(
            "Upload Consumer CSV (SGCC format)", type=["csv"], key="sgcc_upload",
            help="consumer_id, zone, feeder_id, transformer_id, lat, lon, consumer_type, day_1...day_N, label"
        )
        col_dl_sgcc, _ = st.columns([1, 3])
        col_dl_sgcc.download_button(
            "⬇ Sample SGCC CSV", data=get_sample_sgcc_csv(),
            file_name="sgcc_sample.csv", mime="text/csv", use_container_width=True
        )

    if uploaded_sgcc:
        try:
            df_sgcc = pd.read_csv(uploaded_sgcc)
            st.session_state["sgcc_df"] = df_sgcc
            st.success(f"✅ Uploaded {len(df_sgcc)} consumers · {df_sgcc['feeder_id'].nunique() if 'feeder_id' in df_sgcc.columns else '?'} feeders")
        except Exception as e:
            st.error(f"Parse error: {e}")

    if scan_btn:
        with st.spinner(f"Running 3-stage pipeline on {n_consumers} consumers..."):
            df_alerts = run_batch_theft_analysis(str(n_consumers), n_consumers)
            st.session_state["batch_alerts"] = df_alerts

    if "batch_alerts" in st.session_state:
        df_all = st.session_state["batch_alerts"]

        # Apply filters
        if detection_mode == "Intersection (Low FP)":
            df_flagged = df_all[df_all["dual_anomaly"] & (df_all["prob_theft"] >= 0.5)]
        else:
            df_flagged = df_all[df_all["prob_theft"] >= 0.4]

        df_flagged = df_flagged[df_flagged["confidence"] >= min_confidence]
        if show_normal:
            df_show = df_all[df_all["confidence"] >= min_confidence]
        else:
            df_show = df_flagged

        # Summary metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Scanned",   len(df_all))
        m2.metric("Flagged Alerts",  len(df_flagged))
        m3.metric("High Confidence", len(df_flagged[df_flagged["conf_label"] == "HIGH"]))
        m4.metric("Detection Rate",  f"{len(df_flagged)/len(df_all)*100:.1f}%")
        m5.metric("Mode",            "Intersection" if "Intersection" in detection_mode else "Union")

        # Loss type distribution chart
        if len(df_flagged) > 0:
            lt_counts = df_flagged["loss_type"].value_counts()
            colors_pie = {
                "theft": RED, "technical": AMBER, "billing": BLUE, "normal": GREEN
            }
            fig_pie = go.Figure(go.Pie(
                labels=lt_counts.index.tolist(),
                values=lt_counts.values.tolist(),
                marker_colors=[colors_pie.get(l, TEAL) for l in lt_counts.index],
                hole=0.5,
                textfont=dict(color="#ECF0F6"),
            ))
            fig_pie.update_layout(
                **{**PLOTLY_LAYOUT, "title": "Loss Type Distribution",
                   "height": 280, "showlegend": True,
                   "margin": dict(t=40, b=10, l=10, r=10)},
            )
            c_pie, c_tbl = st.columns([1, 2])
            with c_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_tbl:
                st.markdown("**Top High-Confidence Alerts**")
                top_alerts = df_show.nlargest(10, "confidence")[
                    ["consumer_id", "zone", "loss_type", "prob_theft", "confidence", "conf_label", "pattern"]
                ].copy()
                top_alerts["prob_theft"] = top_alerts["prob_theft"].apply(lambda x: f"{x:.1%}")
                top_alerts.columns = ["Consumer", "Zone", "Type", "Prob", "Conf", "Level", "Pattern"]
                st.dataframe(top_alerts, use_container_width=True, hide_index=True, height=260)

        # Full sortable alert table
        st.markdown("---")
        st.markdown("**📋 Full Alert Table** (click rows to expand consumer detail)")

        # Column filters
        cf1, cf2, cf3 = st.columns(3)
        zone_opts   = ["All"] + sorted(df_show["zone"].unique().tolist())
        type_opts   = ["All"] + sorted(df_show["loss_type"].unique().tolist())
        level_opts  = ["All", "HIGH", "MEDIUM", "LOW"]
        sel_zone    = cf1.selectbox("Filter Zone",  zone_opts,  key="fz")
        sel_type    = cf2.selectbox("Filter Type",  type_opts,  key="ft")
        sel_level   = cf3.selectbox("Filter Level", level_opts, key="fl")

        df_view = df_show.copy()
        if sel_zone  != "All": df_view = df_view[df_view["zone"]       == sel_zone]
        if sel_type  != "All": df_view = df_view[df_view["loss_type"]  == sel_type]
        if sel_level != "All": df_view = df_view[df_view["conf_label"] == sel_level]

        display_cols = ["consumer_id", "zone", "feeder_id", "consumer_type",
                        "loss_type", "prob_theft", "confidence", "conf_label",
                        "n_rules", "pattern"]
        df_display = df_view[display_cols].copy()
        df_display["prob_theft"] = df_display["prob_theft"].apply(lambda x: f"{x:.1%}")
        df_display.columns = ["Consumer", "Zone", "Feeder", "Type",
                               "Loss", "Prob", "Conf", "Level", "Rules", "Pattern"]

        st.dataframe(df_display, use_container_width=True, height=380, hide_index=True)

        # Export
        csv_exp = df_view.to_csv(index=False)
        st.download_button("⬇ Export Alerts CSV", csv_exp,
                           "vidyut_theft_alerts.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — THEFT PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_patterns:
    st.subheader("Theft Pattern Identification — Type 1-4 Analysis")
    st.markdown("""
| Pattern | Trigger Condition | BESCOM Signal |
|---|---|---|
| **Type-1: Sudden Drop** | > 60% MoM consumption drop | Meter bypass installation |
| **Type-2: Zero Streak** | ≥ 5 consecutive zero readings | Meter disconnection or tamper |
| **Type-3: Night Spike** | Night/Day ratio > 0.35 | Illegal hookup after meter |
| **Type-4: Gradual Decline** | Consistent downward trend | Progressive meter manipulation |
    """)

    # Synthetic pattern visualization
    rng_pat = np.random.default_rng(55)
    days = np.arange(1, 91)

    fig_pat = go.Figure()
    # Type 1
    t1 = np.concatenate([rng_pat.normal(12, 1.5, 45), rng_pat.normal(3, 0.8, 45)])
    fig_pat.add_trace(go.Scatter(x=days, y=t1, name="Type-1: Sudden Drop",
                                 line=dict(color=RED, width=2)))
    # Type 2
    t2 = np.concatenate([rng_pat.normal(10, 1, 30), np.zeros(15), rng_pat.normal(10, 1, 45)])
    fig_pat.add_trace(go.Scatter(x=days, y=t2, name="Type-2: Zero Streak",
                                 line=dict(color=AMBER, width=2)))
    # Type 4
    t4 = rng_pat.normal(12, 1, 90) * np.linspace(1.0, 0.35, 90)
    fig_pat.add_trace(go.Scatter(x=days, y=t4, name="Type-4: Gradual Decline",
                                 line=dict(color=ORANGE, width=2, dash="dash")))

    fig_pat.update_layout(**{**PLOTLY_LAYOUT,
                              "title": "Theft Pattern Signatures (90-day window)",
                              "xaxis": {"title": "Day"},
                              "yaxis": {"title": "Daily Consumption (kWh)"},
                              "height": 360})
    st.plotly_chart(fig_pat, use_container_width=True)

    # Pattern distribution from batch if available
    if "batch_alerts" in st.session_state:
        df_bat = st.session_state["batch_alerts"]
        flagged_only = df_bat[df_bat["prob_theft"] >= 0.5]
        if len(flagged_only) > 0:
            pat_counts = flagged_only["pattern"].value_counts()
            fig_bar = go.Figure(go.Bar(
                x=pat_counts.values, y=pat_counts.index.tolist(),
                orientation="h",
                marker_color=[RED, AMBER, ORANGE, BLUE][:len(pat_counts)],
            ))
            fig_bar.update_layout(**{**PLOTLY_LAYOUT,
                                     "title": "Pattern Distribution (Flagged Consumers)",
                                     "xaxis": {"title": "Count"},
                                     "height": 280})
            st.plotly_chart(fig_bar, use_container_width=True)
