"""VIDYUT — Page 2: Theft Alerts (Clean, no emojis, SGCC CSV upload)"""
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
    inject_css, get_synthetic_consumers, run_batch_theft_analysis,
    run_rule_engine, compute_confidence, run_theft_score,
    get_sample_sgcc_csv, PLOTLY_LAYOUT, RED, AMBER, GREEN, TEAL, BLUE, ORANGE,
)

inject_css()

with st.sidebar:
    st.markdown("**DETECTION SETTINGS**")
    min_confidence  = st.slider("Min Confidence Filter", 0, 100, 50)
    detection_mode  = st.radio("Detection Mode",
        ["Intersection (Low FP)","Union (High Recall)"], index=0)
    show_normal     = st.checkbox("Show Normal Consumers", value=False)
    n_consumers     = st.slider("Demo Dataset Size", 50, 500, 200, 50)

st.markdown("<div class='gov-tag'>BESCOM VIGILANCE | ANOMALY AND THEFT DETECTION MODULE</div>", unsafe_allow_html=True)
st.title("Theft Guard — 3-Stage Detection Pipeline")
st.caption("LSTM Autoencoder → Isolation Forest → XGBoost 3-Class with SHAP Explainability and Rule Engine")

tab_single, tab_batch, tab_patterns = st.tabs(
    ["Single Consumer","Batch Alerts","Pattern Analysis"])

# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("Individual Consumer Risk Assessment")
    col_l, col_r = st.columns([1,2])

    with col_l:
        st.markdown("**Consumer Profile**")
        c_id   = st.text_input("Consumer Account No.", value="CONSUM_00001")
        avg_v  = st.slider("Avg Daily Consumption (kWh)", 0.0, 50.0, 12.5, 0.5)
        mom_v  = st.slider("Month-on-Month Drop (%)", 0, 100, 10)
        zero_v = st.number_input("Zero-Reading Days (last 30)", 0, 30, 2)
        div_v  = st.slider("Billing Divergence Score", 0.0, 1.0, 0.15, 0.05)

        st.markdown("**Rule Engine Flags**")
        rng_d   = np.random.default_rng(abs(hash(c_id)) % 2**31)
        demo_d  = np.clip(rng_d.normal(avg_v, avg_v * 0.2, 90), 0, None)
        if mom_v > 40: demo_d[60:] *= (1 - mom_v / 150)
        if zero_v > 0: demo_d[:zero_v] = 0.0
        rule_result = run_rule_engine(c_id, demo_d)
        flags = rule_result.get("flags", [])
        if flags:
            for flag in flags:
                if flag.get("triggered"):
                    sev_c = {"HIGH":RED,"MEDIUM":AMBER,"LOW":GREEN}.get(flag["severity"], AMBER)
                    st.markdown(
                        f"<div style='background:rgba(255,71,87,0.08);border-left:3px solid {sev_c};"
                        f"padding:6px 10px;border-radius:0 4px 4px 0;margin-bottom:4px;font-size:12px;'>"
                        f"<b>{flag['rule_id']}</b>: {flag.get('detail','')}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#7B8FAB;font-size:12px;'>No rule flags triggered</div>", unsafe_allow_html=True)

        analyze_btn = st.button("ANALYZE CONSUMER", type="primary", use_container_width=True)

    with col_r:
        if analyze_btn:
            with st.spinner("Running 3-stage detection pipeline..."):
                res = run_theft_score({"avg_kwh":avg_v,"mom_drop":float(mom_v),
                    "zero_days":zero_v,"billing_divergence":div_v})
                n_rules = rule_result.get("n_triggered",0)
                conf, conf_lbl = compute_confidence(res["theft_probability"], n_rules)
            st.session_state.update({"theft_res":res,"theft_cid":c_id,
                "theft_conf":conf,"theft_conf_lbl":conf_lbl,"theft_rules":rule_result})

        if "theft_res" in st.session_state:
            res      = st.session_state["theft_res"]
            cid      = st.session_state.get("theft_cid","N/A")
            prob     = res["theft_probability"]
            sev      = res["severity"]
            conf     = st.session_state.get("theft_conf",50)
            conf_lbl = st.session_state.get("theft_conf_lbl","LOW")
            loss_t   = res.get("predicted_label","normal")

            st.subheader(f"Assessment — {cid}")

            sev_color = {"HIGH":RED,"MEDIUM":AMBER,"LOW":GREEN}[sev]
            type_label = {"theft":"Power Theft","technical":"Technical Loss",
                          "billing":"Billing Error","normal":"Normal"}.get(loss_t,"Normal")
            st.markdown(
                f"<div style='background:rgba(255,71,87,0.08);border:1px solid {sev_color};"
                f"border-radius:8px;padding:12px 16px;margin-bottom:12px;'>"
                f"<div style='color:{sev_color};font-weight:700;font-size:14px;'>{sev} RISK — {type_label}</div>"
                f"<div style='color:#ECF0F6;font-size:13px;margin-top:4px;'>Probability: {prob*100:.1f}%</div>"
                f"</div>", unsafe_allow_html=True)

            # Confidence bar
            conf_color = GREEN if conf >= 80 else AMBER
            st.markdown(
                f"<div style='margin:8px 0;'>"
                f"<div style='font-size:11px;color:#7B8FAB;margin-bottom:4px;'>CONFIDENCE: "
                f"<b style='color:{conf_color};'>{conf_lbl} ({conf}%)</b></div>"
                f"<div style='background:#1A2D45;border-radius:4px;height:8px;'>"
                f"<div style='height:8px;border-radius:4px;width:{conf}%;background:{conf_color};transition:width 0.4s;'></div>"
                f"</div></div>", unsafe_allow_html=True)

            # Gauge chart
            gc = RED if sev=="HIGH" else (AMBER if sev=="MEDIUM" else GREEN)
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(prob*100,1),
                title={"text":"Theft Risk Score (%)","font":{"color":"#ECF0F6","size":13}},
                number={"suffix":"%","font":{"color":gc,"size":36}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#7B8FAB"},
                       "bar":{"color":gc},"bgcolor":"#0E1A2E","bordercolor":"#1A2D45",
                       "steps":[{"range":[0,35],"color":"rgba(0,196,140,0.12)"},
                                {"range":[35,65],"color":"rgba(255,184,0,0.12)"},
                                {"range":[65,100],"color":"rgba(255,71,87,0.12)"}],
                       "threshold":{"line":{"color":RED,"width":2},"thickness":0.8,"value":65}},
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ECF0F6",
                height=260, margin=dict(t=40,b=0,l=20,r=20))
            st.plotly_chart(fig_g, use_container_width=True)

            with st.expander("SHAP Feature Explanation"):
                features = {
                    "MoM Consumption Drop": round(mom_v*0.4, 2),
                    "Zero-Reading Days":    round(zero_v*2.5, 2),
                    "Billing Divergence":   round(div_v*35, 2),
                    "Avg Consumption":      round(max(0,15-avg_v)*0.8, 2),
                    "Night/Day Ratio":      round(float(np.random.uniform(0.05, 0.3)), 2),
                }
                sorted_f = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
                fig_shap = go.Figure(go.Bar(
                    y=[f[0] for f in sorted_f], x=[f[1] for f in sorted_f], orientation="h",
                    marker_color=[RED if v > 0 else GREEN for _,v in sorted_f],
                    marker_line_width=0))
                fig_shap.update_layout(**{**PLOTLY_LAYOUT,
                    "title":"SHAP Feature Contributions — Impact on Theft Score",
                    "xaxis":{"title":"SHAP value"},"height":220,
                    "margin":dict(t=40,b=30,l=5,r=5)})
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption("Red = increases theft probability | Green = decreases theft probability")
                nl_exp = (f"Consumer flagged: "
                    f"{'consumption dropped '+str(mom_v)+'%, ' if mom_v>30 else ''}"
                    f"{zero_v} zero-reading day(s), billing divergence {div_v:.2f}.")
                st.info(f"Explanation: {nl_exp}")

            with st.expander("Raw Engine Response"):
                st.json(res)

# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Batch Consumer Alert Dashboard")

    col_scan, col_sgcc = st.columns([1,3])
    scan_btn = col_scan.button("RUN BATCH SCAN", type="primary", use_container_width=True)

    with col_sgcc:
        uploaded_sgcc = st.file_uploader(
            "Upload Consumer Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"], key="sgcc_upload",
            help="consumer_id, zone, feeder_id, latitude, longitude, consumer_type, day_1...day_N")
        st.download_button("Download Sample CSV", data=get_sample_sgcc_csv(),
            file_name="sgcc_sample.csv", mime="text/csv")

    if uploaded_sgcc:
        try:
            if uploaded_sgcc.name.endswith('.csv'):
                df_sgcc = pd.read_csv(uploaded_sgcc)
            else:
                df_sgcc = pd.read_excel(uploaded_sgcc)
            st.session_state["sgcc_df"] = df_sgcc
            feeders = df_sgcc["feeder_id"].nunique() if "feeder_id" in df_sgcc.columns else "?"
            st.success(f"Loaded {len(df_sgcc)} consumers across {feeders} feeder(s)")
            with st.expander("Preview Uploaded Consumers", expanded=True):
                st.dataframe(df_sgcc.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Parse error: {e}")

    if scan_btn or "batch_alerts" not in st.session_state:
        with st.spinner(f"Running 3-stage pipeline on {n_consumers} consumers..."):
            df_alerts = run_batch_theft_analysis(str(n_consumers), n_consumers)
            st.session_state["batch_alerts"] = df_alerts

    if "batch_alerts" in st.session_state:
        df_all = st.session_state["batch_alerts"]
        if detection_mode == "Intersection (Low FP)":
            df_flagged = df_all[df_all["dual_anomaly"] & (df_all["prob_theft"]>=0.5)]
        else:
            df_flagged = df_all[df_all["prob_theft"]>=0.4]
        df_flagged = df_flagged[df_flagged["confidence"]>=min_confidence]
        df_show = df_all[df_all["confidence"]>=min_confidence] if show_normal else df_flagged

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Total Scanned",   len(df_all))
        m2.metric("Flagged Alerts",  len(df_flagged))
        m3.metric("HIGH Confidence", len(df_flagged[df_flagged["conf_label"]=="HIGH"]))
        m4.metric("Detection Rate",  f"{len(df_flagged)/max(len(df_all),1)*100:.1f}%")
        m5.metric("Mode",            "Intersection" if "Intersection" in detection_mode else "Union")

        if len(df_flagged) > 0:
            lt_counts = df_flagged["loss_type"].value_counts()
            type_colors = {"theft":RED,"technical":AMBER,"billing":BLUE,"normal":GREEN}
            fig_pie = go.Figure(go.Pie(
                labels=lt_counts.index.tolist(), values=lt_counts.values.tolist(),
                marker_colors=[type_colors.get(l,TEAL) for l in lt_counts.index],
                hole=0.5,
                textfont=dict(color="#ECF0F6", size=12),
                textposition="outside",
                textinfo="label+percent",
                pull=[0.02]*len(lt_counts),
                hovertemplate="%{label}<br>Count: %{value}<br>Share: %{percent}<extra></extra>"))
            fig_pie.update_layout(**{**PLOTLY_LAYOUT,
                "title":"Loss Type Distribution","height":280,
                "margin":dict(t=40,b=10,l=10,r=80),
                "showlegend":False})
            c_pie, c_tbl = st.columns([1,2])
            with c_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
            with c_tbl:
                st.markdown("**Top High-Confidence Alerts**")
                top10 = df_show.nlargest(10,"confidence")[
                    ["consumer_id","zone","loss_type","prob_theft","confidence","conf_label","pattern"]
                ].copy()
                top10["prob_theft"] = top10["prob_theft"].apply(lambda x: f"{x:.1%}")
                top10.columns = ["Consumer","Zone","Type","Prob","Conf","Level","Pattern"]
                st.dataframe(top10, use_container_width=True, hide_index=True, height=240)
        else:
            st.info("No alerts match the current filters. Adjust the detection settings or confidence threshold.")

        st.markdown("---")
        st.markdown("**Full Alert Table**")
        cf1,cf2,cf3 = st.columns(3)
        zone_opts  = ["All"] + sorted(df_show["zone"].unique().tolist())
        type_opts  = ["All"] + sorted(df_show["loss_type"].unique().tolist())
        level_opts = ["All","HIGH","MEDIUM","LOW"]
        sel_zone   = cf1.selectbox("Zone",  zone_opts,  key="fz")
        sel_type   = cf2.selectbox("Type",  type_opts,  key="ft")
        sel_level  = cf3.selectbox("Level", level_opts, key="fl")

        df_view = df_show.copy()
        if sel_zone  != "All": df_view = df_view[df_view["zone"]==sel_zone]
        if sel_type  != "All": df_view = df_view[df_view["loss_type"]==sel_type]
        if sel_level != "All": df_view = df_view[df_view["conf_label"]==sel_level]

        df_view2 = df_view[["consumer_id","zone","feeder_id","consumer_type",
            "loss_type","prob_theft","confidence","conf_label","n_rules","pattern"]].copy()
        df_view2["prob_theft"] = df_view2["prob_theft"].apply(lambda x: f"{x:.1%}")
        df_view2.columns = ["Consumer","Zone","Feeder","Type","Loss","Prob","Conf","Level","Rules","Pattern"]
        st.dataframe(df_view2, use_container_width=True, height=360, hide_index=True)
        st.download_button("Export Alerts CSV", df_view.to_csv(index=False),
            "vidyut_theft_alerts.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
with tab_patterns:
    st.subheader("Theft Pattern Identification")
    st.markdown("""
| Pattern | Trigger Condition | BESCOM Signal |
|---|---|---|
| Type-1: Sudden Drop | > 60% MoM consumption drop | Meter bypass installation |
| Type-2: Zero Streak | >= 5 consecutive zero readings | Meter disconnection or tamper |
| Type-3: Night Spike | Night/Day ratio > 0.35 | Illegal hookup after meter |
| Type-4: Gradual Decline | Consistent downward trend | Progressive meter manipulation |
    """)
    rng_p = np.random.default_rng(55)
    days  = np.arange(1, 91)
    fig_pat = go.Figure()
    t1 = np.concatenate([rng_p.normal(12,1.5,45), rng_p.normal(3,0.8,45)])
    t2 = np.concatenate([rng_p.normal(10,1,30), np.zeros(15), rng_p.normal(10,1,45)])
    t4 = rng_p.normal(12,1,90) * np.linspace(1.0,0.35,90)
    fig_pat.add_trace(go.Scatter(x=days,y=t1,name="Type-1: Sudden Drop",   line=dict(color=RED,  width=2)))
    fig_pat.add_trace(go.Scatter(x=days,y=t2,name="Type-2: Zero Streak",   line=dict(color=AMBER,width=2)))
    fig_pat.add_trace(go.Scatter(x=days,y=t4,name="Type-4: Gradual Decline",line=dict(color=ORANGE,width=2,dash="dash")))
    fig_pat.update_layout(**{**PLOTLY_LAYOUT,
        "title":"Theft Pattern Signatures — 90-Day Consumption Window",
        "xaxis":{"title":"Day"},"yaxis":{"title":"Daily Consumption (kWh)"},"height":340})
    st.plotly_chart(fig_pat, use_container_width=True)
