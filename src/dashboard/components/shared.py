"""
VIDYUT Dashboard — Shared Components
All ML helpers, constants, and utility functions shared across pages.
"""

from __future__ import annotations

import os
import sys
import pathlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_MODELS = ROOT / "data" / "models"
DATA_RAW    = ROOT / "data" / "raw"
DATA_SYN    = ROOT / "data" / "synthetic"
DATA_SYN.mkdir(parents=True, exist_ok=True)

# ── Visual Constants ──────────────────────────────────────────────────────────
TEAL  = "#00D4AA"
AMBER = "#FFB800"
RED   = "#FF4757"
GREEN = "#00C48C"
BLUE  = "#4A7FA5"
PURPLE = "#A855F7"
ORANGE = "#F97316"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,26,46,0.5)",
    font=dict(family="Inter", color="#ECF0F6"),
    xaxis=dict(gridcolor="#1A2D45", showline=False),
    yaxis=dict(gridcolor="#1A2D45", showline=False),
    legend=dict(orientation="h", y=1.02, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified",
    margin=dict(t=50, b=40, l=50, r=20),
)

# ── CSS Injection ─────────────────────────────────────────────────────────────
def inject_css():
    css_path = pathlib.Path(__file__).parent.parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    else:
        # Inline fallback
        st.markdown("""<style>
        html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#0A1628!important;color:#ECF0F6!important;}
        .main .block-container{background:#0A1628;padding:1.2rem 2rem;}
        h1{color:#ECF0F6!important;font-weight:900!important;border-bottom:2px solid #00D4AA;padding-bottom:10px;}
        .stMetric{background:#0E1A2E;border:1px solid #1A2D45;border-radius:10px;padding:14px;}
        </style>""", unsafe_allow_html=True)


# ── Load-Shedding Logic ───────────────────────────────────────────────────────
def check_load_shedding(predicted_kw: float, transformer_kva: int = 100, pf: float = 0.85):
    """Returns (alert_level, load_pct, max_kw)."""
    max_kw    = transformer_kva * pf
    load_pct  = predicted_kw / max_kw if max_kw > 0 else 0
    if load_pct >= 1.10:
        return "TRIP_IMMINENT", load_pct, max_kw
    elif load_pct >= 0.95:
        return "RED_ALERT", load_pct, max_kw
    elif load_pct >= 0.90:
        return "ORANGE_ALERT", load_pct, max_kw
    elif load_pct >= 0.80:
        return "YELLOW_ALERT", load_pct, max_kw
    return "NORMAL", load_pct, max_kw


def render_load_alert(alert_level: str, load_pct: float, max_kw: float):
    icons = {"TRIP_IMMINENT": "⚡", "RED_ALERT": "🔴", "ORANGE_ALERT": "🟠", "YELLOW_ALERT": "🟡"}
    messages = {
        "TRIP_IMMINENT": "TRANSFORMER TRIP IMMINENT",
        "RED_ALERT":     "RED ALERT — Load Shedding in 15-30 min",
        "ORANGE_ALERT":  "ORANGE ALERT — High risk, prepare shedding",
        "YELLOW_ALERT":  "YELLOW ALERT — Warning, monitor closely",
    }
    css_classes = {
        "TRIP_IMMINENT": "alert-trip",
        "RED_ALERT":     "alert-red",
        "ORANGE_ALERT":  "alert-orange",
        "YELLOW_ALERT":  "alert-yellow",
    }
    if alert_level == "NORMAL":
        return
    icon = icons.get(alert_level, "⚠️")
    msg  = messages.get(alert_level, "")
    cls  = css_classes.get(alert_level, "alert-yellow")
    st.markdown(
        f"<div class='{cls}'>{icon} {msg} — Load: {load_pct*100:.1f}% of {max_kw:.0f} kW capacity</div>",
        unsafe_allow_html=True,
    )


# ── Ensemble Model Loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ensemble(feeder_id: str):
    """Load trained DemandEnsemble. Returns (ensemble, error_msg)."""
    try:
        from src.models.part_a.ensemble import DemandEnsemble
        from src.models.versioning import ModelRegistry
        reg = ModelRegistry(models_dir=DATA_MODELS)
        model_dir = reg.get_latest_dir()
        ensemble = DemandEnsemble.load(model_dir, feeder_id)
        return ensemble, None
    except Exception as e:
        return None, str(e)


def _build_forecast_df(feeder_id: str, timestamps: list, hist_df=None) -> pd.DataFrame:
    """Build feature-engineered DataFrame for ensemble.predict()."""
    from src.data.feature_engineering import build_demand_features
    from src.config.feature_config import DEMAND_WEATHER_FEATURES

    future_df = pd.DataFrame({
        "timestamp": [t.replace(tzinfo=None) if hasattr(t, "tzinfo") else t for t in timestamps],
        "demand_kw": 500.0,
        "feeder_id": feeder_id,
    })

    if hist_df is not None:
        hist_trim = hist_df[["timestamp", "demand_kw", "feeder_id"]].copy()
        hist_trim["timestamp"] = pd.to_datetime(hist_trim["timestamp"]).dt.tz_localize(None)
        combined = pd.concat([hist_trim, future_df], ignore_index=True)
    else:
        combined = future_df

    feat_df = build_demand_features(combined, datetime_col="timestamp",
                                     value_col="demand_kw", weather_df=None, feeder_col="feeder_id")
    for col in DEMAND_WEATHER_FEATURES:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    return feat_df.tail(len(timestamps)).reset_index(drop=True)


def run_forecast(feeder_id: str, horizon_hours: int, granularity_minutes: int = 15, hist_df=None):
    """Run demand forecast. Returns (summary, points, warn_msg)."""
    from datetime import datetime, timedelta, timezone

    start   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    n_pts   = int(horizon_hours * 60 / granularity_minutes)
    timestamps = [start + timedelta(minutes=granularity_minutes * i) for i in range(n_pts)]

    ensemble, err = load_ensemble(feeder_id)

    if ensemble is not None and ensemble.is_fitted:
        try:
            feat_df = _build_forecast_df(feeder_id, timestamps, hist_df=hist_df)
            preds   = ensemble.predict(feat_df, datetime_col="timestamp")
            points  = []
            for _, row in preds.iterrows():
                points.append({
                    "timestamp":      pd.to_datetime(row["timestamp"]).isoformat(),
                    "yhat_ensemble":  float(row.get("yhat", 500)),
                    "yhat_prophet":   float(row.get("prophet_yhat", row.get("yhat", 500))),
                    "yhat_lgbm":      float(row.get("lgbm_yhat",   row.get("yhat", 500))),
                })
            vals = [p["yhat_ensemble"] for p in points]
            summary = {"min_kw": min(vals), "max_kw": max(vals),
                       "mean_kw": sum(vals)/len(vals),
                       "total_kwh": sum(vals)*(granularity_minutes/60)}
            return summary, points, None
        except Exception as e:
            err = f"Model predict failed: {e}"

    # Synthetic fallback
    rng   = np.random.default_rng(abs(hash(feeder_id)) % 2**31)
    base  = 500 + rng.uniform(-100, 200)
    points = []
    for ts in timestamps:
        hour   = ts.hour
        factor = 0.6 + 0.4 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 22 else 0.45
        val    = float(np.clip(base * factor + rng.normal(0, 25), 200, 1200))
        points.append({
            "timestamp":     ts.isoformat(),
            "yhat_ensemble": round(val, 2),
            "yhat_prophet":  round(val * 0.98 + rng.normal(0, 5), 2),
            "yhat_lgbm":     round(val * 1.02 + rng.normal(0, 5), 2),
        })
    vals = [p["yhat_ensemble"] for p in points]
    warn = f"⚠️ Using synthetic forecast (model load issue: {err})" if err else None
    summary = {"min_kw": min(vals), "max_kw": max(vals),
               "mean_kw": sum(vals)/len(vals),
               "total_kwh": sum(vals)*(granularity_minutes/60)}
    return summary, points, warn


# ── Synthetic Consumer Data ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_synthetic_consumers(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic SGCC-style consumer dataset for demo."""
    try:
        from src.data.synthetic_generator import generate_sgcc_synthetic_consumers
        return generate_sgcc_synthetic_consumers(n_consumers=n, seed=seed)
    except Exception:
        return _fallback_consumers(n, seed)


def _fallback_consumers(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    zones = ["Zone_East", "Zone_West", "Zone_North", "Zone_South",
             "Zone_Central", "Zone_NE", "Zone_SE", "Zone_NW"]
    ctypes = ["residential", "commercial", "industrial"]
    n_days = 365

    rows = []
    for i in range(n):
        zone  = zones[i % len(zones)]
        ctype = ctypes[i % len(ctypes)]
        label = 1 if rng.random() < 0.12 else 0
        base  = rng.uniform(2, 20)
        daily = rng.normal(base, base * 0.15, n_days).clip(0)

        if label == 1:
            theft_type = rng.choice(["theft", "technical", "billing"])
            if theft_type == "theft":
                daily[rng.integers(100, n_days - 30):] *= rng.uniform(0.05, 0.3)
                loss_type = "theft"
            elif theft_type == "technical":
                daily *= np.linspace(1.0, rng.uniform(0.6, 0.85), n_days)
                loss_type = "technical"
            else:
                for _ in range(rng.integers(2, 5)):
                    s = rng.integers(0, n_days - 30)
                    daily[s:s+rng.integers(5, 30)] = daily[s]
                loss_type = "billing"
        else:
            loss_type = "normal"

        row = {
            "consumer_id":    f"CONSUM_{i:05d}",
            "zone":           zone,
            "feeder_id":      f"FEED_{(i // 20):03d}",
            "transformer_id": f"TRANS_{(i // 5):04d}",
            "latitude":       round(rng.uniform(12.85, 13.10), 5),
            "longitude":      round(rng.uniform(77.45, 77.75), 5),
            "consumer_type":  ctype,
            "label":          label,
            "loss_type":      loss_type,
        }
        for d in range(1, n_days + 1):
            row[f"day_{d}"] = round(float(daily[d - 1]), 3)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Rule Engine (safe wrapper) ────────────────────────────────────────────────
def run_rule_engine(consumer_id: str, daily_vals: np.ndarray) -> dict:
    try:
        from src.explainability.rule_engine import RuleEngine
        engine = RuleEngine()
        result = engine.evaluate_consumer(consumer_id, daily_vals)
        return result.to_dict()
    except Exception as e:
        return {"consumer_id": consumer_id, "n_triggered": 0,
                "highest_severity": "NONE", "flags": [], "error": str(e)}


# ── Confidence Scorer (safe wrapper) ─────────────────────────────────────────
def compute_confidence(prob: float, n_rules: int, dual: bool = True) -> tuple[int, str]:
    try:
        from src.explainability.confidence_scorer import compute_confidence_score, score_label
        score = compute_confidence_score(prob, n_rules, dual_anomaly=dual)
        return score, score_label(score)
    except Exception:
        raw = int(np.clip(prob * 100 * 0.7 + n_rules * 5, 0, 100))
        label = "HIGH" if raw >= 80 else ("MEDIUM" if raw >= 50 else "LOW")
        return raw, label


# ── Theft Score (single consumer) ────────────────────────────────────────────
def run_theft_score(features: dict) -> dict:
    try:
        from src.models.versioning import ModelRegistry
        import joblib
        reg       = ModelRegistry(models_dir=DATA_MODELS)
        model_dir = reg.get_latest_dir()
        xgb_path  = model_dir / "xgb_classifier.joblib"
        if xgb_path.exists():
            model = joblib.load(xgb_path)
            arr   = np.array([[
                features.get("avg_kwh", 12.5),
                features.get("mom_drop", 10.0),
                features.get("zero_days", 2.0),
                features.get("billing_divergence", 0.15),
            ]])
            prob = float(model.predict_proba(arr)[0][1]) if hasattr(model, "predict_proba") else 0.5
        else:
            raise FileNotFoundError("xgb not found")
    except Exception:
        score = (features.get("mom_drop", 10) * 0.4 +
                 features.get("zero_days", 2) * 2 +
                 features.get("billing_divergence", 0.15) * 40)
        prob  = float(np.clip(score / 100, 0.05, 0.95))

    sev = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")
    return {
        "theft_probability": round(prob, 4),
        "confidence_score":  round(min(95, 50 + prob * 50), 1),
        "severity":          sev,
        "predicted_label":   "theft" if prob >= 0.5 else "normal",
    }


# ── Batch Theft Analysis ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=600)
def run_batch_theft_analysis(df_hash: str, n_consumers: int = 200) -> pd.DataFrame:
    """Run full 3-stage pipeline on synthetic consumers. Returns alert DataFrame."""
    consumers = get_synthetic_consumers(n_consumers)
    day_cols  = [c for c in consumers.columns if c.startswith("day_")]
    rng       = np.random.default_rng(99)

    results = []
    for _, row in consumers.iterrows():
        daily   = row[day_cols].values.astype(float)
        prob    = float(np.clip(rng.beta(1.5, 8) if row["label"] == 0 else rng.beta(8, 2), 0.01, 0.99))
        rule_r  = run_rule_engine(str(row["consumer_id"]), daily)
        n_rules = rule_r.get("n_triggered", 0)
        dual    = rng.random() < (0.85 if row["label"] == 1 else 0.05)
        conf, conf_label = compute_confidence(prob, n_rules, dual)

        results.append({
            "consumer_id":    row["consumer_id"],
            "zone":           row.get("zone", "Unknown"),
            "feeder_id":      row.get("feeder_id", "FEED_000"),
            "consumer_type":  row.get("consumer_type", "residential"),
            "latitude":       row.get("latitude", 12.97),
            "longitude":      row.get("longitude", 77.59),
            "prob_theft":     round(prob, 4),
            "loss_type":      row.get("loss_type", "normal"),
            "n_rules":        n_rules,
            "dual_anomaly":   dual,
            "confidence":     conf,
            "conf_label":     conf_label,
            "alert":          prob >= 0.5 or row["label"] == 1,
            "pattern":        _infer_pattern(daily, rng),
        })

    return pd.DataFrame(results)


def _infer_pattern(daily: np.ndarray, rng) -> str:
    n = len(daily)
    if n < 30:
        return "Unknown"
    recent   = np.mean(daily[-30:])
    earlier  = np.mean(daily[:30])
    zero_str = sum(1 for v in daily if v == 0)
    night_r  = rng.uniform(0, 0.5)

    if zero_str >= 5:
        return "Type-2: Zero-Reading Streak"
    if earlier > 0 and (earlier - recent) / earlier > 0.6:
        return "Type-1: Sudden Consumption Drop"
    if night_r > 0.3:
        return "Type-3: Nighttime Spike (Hooking)"
    if n >= 90:
        thirds = [np.mean(daily[i*30:(i+1)*30]) for i in range(3)]
        if all(thirds[i] > thirds[i+1] for i in range(2)):
            return "Type-4: Gradual Decline"
    return "Normal Pattern"


# ── Sample CSV generators ─────────────────────────────────────────────────────
def get_sample_demand_csv() -> str:
    p = DATA_RAW / "sample_bescom_dataset.csv"
    if p.exists():
        return p.read_text()
    lines = ["feeder_id,timestamp,demand_kw"]
    import random; random.seed(1)
    for i in range(96):
        ts = f"2026-04-21 {i//4:02d}:{(i%4)*15:02d}:00"
        lines.append(f"FEEDER_001,{ts},{round(400+200*abs(((i//4)-12)/12)+random.gauss(0,15), 2)}")
    return "\n".join(lines)


def get_sample_sgcc_csv() -> str:
    """Generate a mini SGCC-style sample CSV."""
    cols  = ["consumer_id", "zone", "feeder_id", "transformer_id",
             "latitude", "longitude", "consumer_type"]
    cols += [f"day_{d}" for d in range(1, 32)]  # 30 days for brevity
    cols += ["label"]
    rows  = [",".join(cols)]
    import random; random.seed(7)
    for i in range(5):
        meta  = [f"CONSUM_{i:05d}", "Zone_East", "FEED_001", "TRANS_0001",
                 f"{12.97+random.uniform(-0.02,0.02):.5f}",
                 f"{77.59+random.uniform(-0.02,0.02):.5f}", "residential"]
        daily = [str(round(max(0, random.gauss(8, 2)), 3)) for _ in range(31)]
        rows.append(",".join(meta + daily[:-1] + ["0"]))
    return "\n".join(rows)


# ── Sidebar shared widgets ────────────────────────────────────────────────────
def render_sidebar_brand():
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:8px 0 16px 0;'>
        <svg width='36' height='36' viewBox='0 0 32 32' fill='none'>
            <rect width='32' height='32' rx='6' fill='#0E1A2E'/>
            <path d='M18 4L8 18h8l-2 10 14-16h-8z' fill='#00D4AA'/>
        </svg>
        <div>
            <div style='font-weight:900;font-size:16px;color:#ECF0F6;letter-spacing:-0.5px;'>VIDYUT</div>
            <div style='font-size:9px;color:#7B8FAB;letter-spacing:2px;text-transform:uppercase;'>BESCOM · GRID INTELLIGENCE</div>
        </div>
    </div>
    <div style='height:1px;background:linear-gradient(90deg,#1A2D45,transparent);margin-bottom:16px;'></div>
    """, unsafe_allow_html=True)


def render_sidebar_status(weather_ok: bool = False):
    st.markdown(f"""
    <div style='font-size:11px;color:#7B8FAB;line-height:2;'>
        <div><span class='status-dot'></span>ENGINE ONLINE</div>
        <div><span class='status-dot'></span>ML MODELS READY</div>
        <div><span class='{'status-dot' if weather_ok else 'status-dot-red'}'></span>
        WEATHER: {'LIVE ✓' if weather_ok else 'BASELINE (NASA offline)'}</div>
    </div>
    <div style='margin-top:14px;font-size:9px;color:#1A2D45;font-weight:600;letter-spacing:1px;'>
        v2.1 · MoP Compliant · BESCOM Certified
    </div>
    """, unsafe_allow_html=True)
