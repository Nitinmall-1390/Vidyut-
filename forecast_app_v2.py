"""
VIDYUT — Time Series Forecasting Tool
Senior-grade, high-performance Streamlit app.
Supports CSV / Excel | Holt-Winters (statsmodels) | Plotly interactive charts

IMPLEMENTATION v2 — CHANGES FROM v1:
- Forecast ONLY runs when user explicitly clicks RUN FORECAST.
- If user changes any setting (horizon, trend, seasonal, etc.), the previous
  forecast is cleared and the user must click RUN FORECAST again.
- Data Preview & Statistics tabs remain visible at all times.
- Forecast Results & Export tabs show placeholder until RUN FORECAST is clicked.
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIDYUT | Time Series Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Core dark background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #060E1C;
    color: #ECF0F6;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0A1628;
    border-right: 1px solid #1A2D45;
}
[data-testid="stSidebar"] * { color: #ECF0F6; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0E1A2E;
    border: 1px solid #1A2D45;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #00D4AA; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #7B8FAB; font-size: 11px; text-transform: uppercase; }

/* ── Buttons ── */
.stButton > button {
    background-color: #0E2929 !important;
    color: #00D4AA !important;
    border: 1px solid #00D4AA !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding: 10px 20px !important;
    box-shadow: none !important;
    transition: all 0.2s;
}
.stButton > button:hover {
    background-color: #1A3D3D !important;
    color: #00FFCC !important;
    border-color: #00FFCC !important;
    box-shadow: 0 0 8px rgba(0,255,204,0.1) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 1px solid #1A2D45;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #7B8FAB;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 10px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: #00D4AA !important;
    border-bottom: 2px solid #00D4AA !important;
    background-color: transparent !important;
}

/* ── Inputs ── */
.stSelectbox label, .stSlider label, .stRadio label {
    color: #7B8FAB !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #0E1A2E;
    border: 1px dashed #1A2D45;
    border-radius: 8px;
}

/* ── Gov tag ── */
.gov-tag {
    display: inline-block;
    background: #0E1A2E;
    border: 1px solid #1A2D45;
    color: #7B8FAB;
    font-size: 9px;
    letter-spacing: 2px;
    padding: 3px 10px;
    border-radius: 2px;
    margin-bottom: 6px;
    text-transform: uppercase;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1A2D45; border-radius: 6px; }

/* ── Info/Error boxes ── */
.stAlert { border-radius: 6px; }

/* ── Section divider ── */
hr { border-color: #1A2D45 !important; }

/* ── Expander ── */
details summary { color: #7B8FAB !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TEAL  = "#00D4AA"
AMBER = "#FFB800"
RED   = "#FF4757"
BLUE  = "#3A86FF"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(6,14,28,0)",
    plot_bgcolor="rgba(6,14,28,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#ECF0F6", size=11),
    xaxis=dict(gridcolor="#1A2D45", linecolor="#1A2D45", zerolinecolor="#1A2D45"),
    yaxis=dict(gridcolor="#1A2D45", linecolor="#1A2D45", zerolinecolor="#1A2D45"),
    legend=dict(bgcolor="rgba(10,22,40,0.8)", bordercolor="#1A2D45", borderwidth=1),
    margin=dict(t=50, b=40, l=50, r=30),
    hovermode="x unified",
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load CSV or Excel from bytes. Cached on file content hash."""
    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def run_holt_winters(
    values_tuple: tuple,
    dates_tuple: tuple,
    periods: int,
    freq: str,
    trend: str,
    seasonal: str,
    seasonal_periods: int,
) -> tuple:
    """
    Fit ExponentialSmoothing model and return forecast + confidence intervals.
    Input as tuples (hashable) so Streamlit can cache the result.
    """
    series = pd.Series(list(values_tuple), index=pd.DatetimeIndex(list(dates_tuple)))
    series = series.asfreq(freq, method="pad")
    series = series.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # Guard: multiplicative seasonal requires all positive values
    if seasonal == "mul" and (series <= 0).any():
        seasonal = "add"

    model = ExponentialSmoothing(
        series,
        trend=trend if trend != "none" else None,
        seasonal=seasonal if seasonal != "none" else None,
        seasonal_periods=seasonal_periods if seasonal not in ("none", None) else None,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True, remove_bias=True)

    forecast = fit.forecast(periods)

    # In-sample fitted values for residuals
    fitted = fit.fittedvalues
    residuals = series - fitted
    sigma = residuals.std()

    # Confidence intervals (approx — statsmodels doesn't return CIs for HW natively)
    z80, z95 = 1.282, 1.960
    lower80 = forecast - z80 * sigma
    upper80 = forecast + z80 * sigma
    lower95 = forecast - z95 * sigma
    upper95 = forecast + z95 * sigma

    return (
        series,
        fitted,
        forecast,
        lower80, upper80,
        lower95, upper95,
        sigma,
    )


def compute_metrics(series: pd.Series, fitted: pd.Series) -> dict:
    """Compute MAE, MAPE, RMSE on in-sample fit."""
    err = series - fitted
    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err**2).mean()))
    mape = float((np.abs(err / series.replace(0, np.nan))).mean() * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:10px 0 12px 0;border-bottom:1px solid #1A2D45;margin-bottom:12px;'>"
        "<div style='font-weight:900;font-size:18px;color:#ECF0F6;'>VIDYUT</div>"
        "<div style='font-size:9px;color:#7B8FAB;letter-spacing:2px;'>TIME SERIES FORECASTING ENGINE</div>"
        "</div>", unsafe_allow_html=True)

    st.markdown("**FORECAST SETTINGS**")

    forecast_period = st.slider(
        "Forecast Horizon (Days)", min_value=7, max_value=365, value=90, step=7,
        help="Number of future days to forecast"
    )

    freq_label = st.selectbox(
        "Data Frequency",
        ["Daily (D)", "Weekly (W)", "Monthly (MS)", "Business Daily (B)"],
        index=0,
    )
    FREQ_MAP = {"Daily (D)": "D", "Weekly (W)": "W", "Monthly (MS)": "MS", "Business Daily (B)": "B"}
    freq = FREQ_MAP[freq_label]

    trend_label = st.radio(
        "Trend Component",
        ["Additive (add)", "Multiplicative (mul)", "None"],
        index=0,
    )
    TREND_MAP = {"Additive (add)": "add", "Multiplicative (mul)": "mul", "None": "none"}
    trend = TREND_MAP[trend_label]

    seasonal_label = st.radio(
        "Seasonal Component",
        ["Additive (add)", "Multiplicative (mul)", "None"],
        index=0,
    )
    seasonal = TREND_MAP[seasonal_label]

    seasonal_periods = st.number_input(
        "Seasonal Period (e.g. 7=weekly, 12=monthly, 365=yearly)",
        min_value=2, max_value=365, value=7, step=1,
        help="Number of time steps in one seasonal cycle",
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#7B8FAB;'>"
        "<span style='color:#00C48C;'>&#9679;</span> ENGINE ONLINE<br>"
        "<span style='color:#00C48C;'>&#9679;</span> STATSMODELS READY<br>"
        "<span style='color:#00C48C;'>&#9679;</span> HW EXPONENTIAL SMOOTHING</div>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='gov-tag'>BESCOM ANALYTICS | AI FORECASTING ENGINE</div>", unsafe_allow_html=True)
st.title("Time Series Forecasting — Holt-Winters Engine")
st.caption("Upload any time series dataset (CSV or Excel) and generate accurate forecasts using Holt-Winters Exponential Smoothing.")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Upload a file containing at least one date column and one numeric target column",
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel file to begin. The app will auto-detect your date and value columns.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    try:
        file_bytes = uploaded_file.read()
        df_raw = load_file(file_bytes, uploaded_file.name)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

if df_raw.empty:
    st.error("The uploaded file is empty. Please upload a file with data.")
    st.stop()

st.success(f"Dataset loaded: **{len(df_raw):,} rows x {len(df_raw.columns)} columns**")

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN SELECTION
# ─────────────────────────────────────────────────────────────────────────────
all_cols = df_raw.columns.tolist()
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

# Auto-detect date column: prefer column named 'date', 'Date', 'timestamp', 'ds'
auto_date = next(
    (c for c in all_cols if c.lower() in ("date","datetime","timestamp","ds","time","month","week")), all_cols[0]
)
# Auto-detect target: first numeric column that isn't the date
auto_target = numeric_cols[0] if numeric_cols else None

c1, c2 = st.columns(2)
date_col = c1.selectbox("Date Column", all_cols, index=all_cols.index(auto_date))
if not numeric_cols:
    st.error("No numeric columns found. Please check your file and ensure at least one column contains numeric values.")
    st.stop()
target_col = c2.selectbox("Target / Value Column", numeric_cols,
                           index=numeric_cols.index(auto_target) if auto_target in numeric_cols else 0)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
df = df_raw[[date_col, target_col]].copy()
df.columns = ["ds", "y"]

# Convert date
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
n_bad_dates = df["ds"].isna().sum()
if n_bad_dates > 0:
    st.warning(f"{n_bad_dates} rows had unparseable dates and will be dropped.")
df = df.dropna(subset=["ds"])

# Convert target to numeric
df["y"] = pd.to_numeric(df["y"], errors="coerce")
n_bad_vals = df["y"].isna().sum()
if n_bad_vals > 0:
    st.warning(f"{n_bad_vals} missing/non-numeric values found -- filled via forward fill.")
df["y"] = df["y"].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

# Remove duplicates (keep mean)
n_dupes = df.duplicated(subset=["ds"]).sum()
if n_dupes > 0:
    st.warning(f"{n_dupes} duplicate timestamps detected -- aggregated by mean.")
df = df.groupby("ds", as_index=False)["y"].mean()

# Sort
df = df.sort_values("ds").reset_index(drop=True)

if len(df) < 10:
    st.error("Not enough data points after cleaning (minimum 10 required). Please upload a larger dataset.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREVIEW TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_preview, tab_stats, tab_forecast, tab_export = st.tabs([
    "Data Preview", "Statistics", "Forecast Results", "Export"
])

with tab_preview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",    f"{len(df):,}")
    c2.metric("Date Range",    f"{df['ds'].dt.date.min()} -> {df['ds'].dt.date.max()}")
    c3.metric("Min Value",     f"{df['y'].min():,.2f}")
    c4.metric("Max Value",     f"{df['y'].max():,.2f}")
    st.markdown("**First 10 Rows of Cleaned Dataset**")
    st.dataframe(
        df.head(10).rename(columns={"ds": date_col, "y": target_col}),
        use_container_width=True, hide_index=True
    )

with tab_stats:
    stats = df["y"].describe().to_frame().T
    stats.columns = ["Count","Mean","Std Dev","Min","25th %ile","Median","75th %ile","Max"]
    st.dataframe(stats.round(3), use_container_width=True, hide_index=True)
    # Historical chart (raw data only)
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=df["ds"], y=df["y"],
        mode="lines",
        name="Historical",
        line=dict(color=TEAL, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>Value: %{y:,.2f}<extra></extra>",
    ))
    fig_raw.update_layout(**{**PLOTLY_LAYOUT, "title": f"{target_col} -- Historical Data", "height": 320})
    st.plotly_chart(fig_raw, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN FORECAST BUTTON  (v2 BEHAVIOUR)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_btn, col_info = st.columns([1, 4])
run_btn = col_btn.button("RUN FORECAST", type="primary", use_container_width=True)
col_info.markdown(
    f"<div style='color:#7B8FAB;font-size:12px;padding-top:12px;'>"
    f"Holt-Winters Exponential Smoothing &nbsp;|&nbsp; "
    f"Horizon: <b style='color:#00D4AA'>{forecast_period} days</b> &nbsp;|&nbsp; "
    f"Trend: <b style='color:#FFB800'>{trend_label}</b> &nbsp;|&nbsp; "
    f"Seasonal: <b style='color:#FFB800'>{seasonal_label}</b> &nbsp;|&nbsp; "
    f"Period: <b style='color:#00D4AA'>{seasonal_periods}</b>"
    f"</div>",
    unsafe_allow_html=True
)

# v2: Only run when user EXPLICITLY clicks the button.
# If settings changed, wipe previous result so user must click again.
forecast_key = f"fc_{date_col}_{target_col}_{forecast_period}_{freq}_{trend}_{seasonal}_{seasonal_periods}_{len(df)}"
if st.session_state.get("fc_key") != forecast_key:
    st.session_state.pop("fc_result", None)
    st.session_state["fc_key"] = forecast_key

if run_btn:
    with st.spinner("Fitting Holt-Winters model..."):
        try:
            result = run_holt_winters(
                values_tuple=tuple(df["y"].tolist()),
                dates_tuple=tuple(df["ds"].tolist()),
                periods=forecast_period,
                freq=freq,
                trend=trend,
                seasonal=seasonal if seasonal != "none" else "none",
                seasonal_periods=int(seasonal_periods),
            )
            st.session_state["fc_result"] = result
            st.session_state["fc_target"] = target_col
            st.session_state["fc_date"]   = date_col
        except Exception as e:
            st.error(f"Forecasting failed: {e}")
            st.info("Tip: Try switching Trend/Seasonal to 'None', or ensure your data has enough rows for the chosen seasonal period.")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FORECAST RESULTS  (only displayed after RUN FORECAST is clicked)
# ─────────────────────────────────────────────────────────────────────────────
if "fc_result" not in st.session_state:
    with tab_forecast:
        st.info("Click **RUN FORECAST** above to generate predictions.")
    with tab_export:
        st.info("Forecast results will be available here after you run the forecast.")
else:
    series, fitted, forecast, lower80, upper80, lower95, upper95, sigma = st.session_state["fc_result"]
    metrics = compute_metrics(series, fitted)

    with tab_forecast:
        # ── KPI Strip ─────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("MAE",                  f"{metrics['mae']:,.2f}")
        k2.metric("RMSE",                 f"{metrics['rmse']:,.2f}")
        k3.metric("MAPE",                 f"{metrics['mape']:.1f}%")
        k4.metric("Forecast Peak",        f"{forecast.max():,.2f}")
        k5.metric("Forecast Peak Date",   str(forecast.idxmax().date()))

        st.markdown("---")

        # ── Interactive Chart ─────────────────────────────────────────────────
        fig = go.Figure()

        # 95% CI band
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(upper95.values) + list(lower95.values[::-1]),
            fill="toself",
            fillcolor="rgba(0,212,170,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Confidence",
            hoverinfo="skip",
        ))

        # 80% CI band
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(upper80.values) + list(lower80.values[::-1]),
            fill="toself",
            fillcolor="rgba(0,212,170,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence",
            hoverinfo="skip",
        ))

        # Historical data
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines",
            name="Historical",
            line=dict(color=TEAL, width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Actual: %{y:,.2f}<extra></extra>",
        ))

        # Fitted values (in-sample)
        fig.add_trace(go.Scatter(
            x=fitted.index, y=fitted.values,
            mode="lines",
            name="Model Fit",
            line=dict(color=BLUE, width=1, dash="dot"),
            opacity=0.6,
            hovertemplate="%{x|%Y-%m-%d}<br>Fitted: %{y:,.2f}<extra></extra>",
        ))

        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values,
            mode="lines",
            name="Forecast",
            line=dict(color=AMBER, width=2.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:,.2f}<extra></extra>",
        ))

        # Vertical divider at forecast start
        fig.add_vline(
            x=forecast.index[0], line_dash="dot", line_color="#1A2D45",
            annotation_text="Forecast Start", annotation_font_color="#7B8FAB",
            annotation_font_size=10,
        )

        fig.update_layout(**{
            **PLOTLY_LAYOUT,
            "title": f"{st.session_state.get('fc_target', target_col)} -- Historical vs Forecast ({forecast_period} Days)",
            "height": 480,
            "xaxis": {**PLOTLY_LAYOUT["xaxis"], "rangeslider": dict(visible=True, thickness=0.05)},
        })
        st.plotly_chart(fig, use_container_width=True)

        # ── Forecast Data Table ────────────────────────────────────────────────
        with st.expander("View Forecast Table", expanded=False):
            df_fc = pd.DataFrame({
                "Date":          forecast.index.strftime("%Y-%m-%d"),
                "Forecast":      forecast.values.round(2),
                "Lower 80%":     lower80.values.round(2),
                "Upper 80%":     upper80.values.round(2),
                "Lower 95%":     lower95.values.round(2),
                "Upper 95%":     upper95.values.round(2),
            })
            st.dataframe(df_fc, use_container_width=True, height=280, hide_index=True)

    with tab_export:
        st.subheader("Export Forecast Results")

        # Build combined export DF
        df_hist_export = pd.DataFrame({
            "Date":     series.index.strftime("%Y-%m-%d"),
            "Type":     "Historical",
            "Value":    series.values.round(2),
            "Fitted":   fitted.reindex(series.index).values.round(2),
            "Lower80":  np.nan,
            "Upper80":  np.nan,
            "Lower95":  np.nan,
            "Upper95":  np.nan,
        })
        df_fc_export = pd.DataFrame({
            "Date":     forecast.index.strftime("%Y-%m-%d"),
            "Type":     "Forecast",
            "Value":    forecast.values.round(2),
            "Fitted":   np.nan,
            "Lower80":  lower80.values.round(2),
            "Upper80":  upper80.values.round(2),
            "Lower95":  lower95.values.round(2),
            "Upper95":  upper95.values.round(2),
        })
        df_full = pd.concat([df_hist_export, df_fc_export], ignore_index=True)

        st.download_button(
            label="Download Full Forecast CSV",
            data=df_full.to_csv(index=False),
            file_name=f"vidyut_forecast_{target_col}_{forecast_period}d.csv",
            mime="text/csv",
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Historical Points",  f"{len(series):,}")
        c2.metric("Forecast Points",    f"{len(forecast):,}")
        st.markdown("**Accuracy Summary**")
        acc_df = pd.DataFrame({
            "Metric": ["MAE", "RMSE", "MAPE"],
            "Value":  [f"{metrics['mae']:,.3f}", f"{metrics['rmse']:,.3f}", f"{metrics['mape']:.2f}%"],
            "Interpretation": [
                "Average absolute error per data point",
                "Root mean square error (penalises large errors)",
                "Mean absolute percentage error (lower = better)"
            ]
        })
        st.dataframe(acc_df, use_container_width=True, hide_index=True)
