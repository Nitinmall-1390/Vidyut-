# VIDYUT — Run Instructions & Feature Guide

> BESCOM Grid Intelligence Platform | Hackathon Submission

---

## 1. Quick Start (Local)

```bash
# Clone repository
git clone <repo-url>
cd Vidyut

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-ui.txt

# Launch Streamlit dashboard
streamlit run src/dashboard/app.py

# (Optional) Launch FastAPI backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Access:** Dashboard opens at `http://localhost:8501`  
**Live Deployment:** [https://huggingface.co/spaces/nitinmal1121212/vidyut-ai](https://huggingface.co/spaces/nitinmal1121212/vidyut-ai)

---

## 2. Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Streamlit |
| Forecasting | Prophet, LightGBM, Holt-Winters (statsmodels) |
| Anomaly Detection | LSTM Autoencoder, Isolation Forest, XGBoost |
| Graph Analytics | NetworkX, python-louvain |
| Geospatial | PyDeck, GeoHash |
| API | FastAPI, Uvicorn |
| Database | Neon (Postgres), Upstash (Redis) |
| Deployment | Docker, Hugging Face Spaces |

---

## 3. Modules & Features

### Demand Forecast
Multi-horizon ensemble prediction (Prophet + LightGBM) per 11kV feeder with NASA POWER weather integration. Upload feeder CSV, set horizon/granularity, click **RUN FORECAST** — interactive chart + load-shedding alerts appear.

### Theft Alerts
3-Stage Detection Pipeline: LSTM Autoencoder → Isolation Forest → XGBoost. Batch-scans consumers, flags theft/technical/billing anomalies with SHAP explainability. Filter by zone, loss type, confidence.

### Geospatial Map
Bangalore-wide consumer anomaly heatmap across 8 BESCOM zones. Toggle between consumer pins, theft heatmap, and zone overlay. Real-time risk filtering by HIGH/MEDIUM/LOW.

### Ring Detection
Louvain community detection on consumer-transformer graphs identifies theft syndicates. Scans network edges (shared transformer / geohash proximity), flags rings with >threshold anomalous members.

### Audit Trail
Immutable log of every AI prediction — queryable by consumer, date range, alert type, confidence level, model version. Exportable CSV for regulatory compliance.

### Time Series Forecast
Upload any CSV/Excel, auto-detect date/value columns, configure Holt-Winters parameters (trend, seasonal, horizon). Interactive Plotly charts with 80%/95% confidence bands + CSV export.

---

## 4. File Upload Formats

| Module | Format | Required Columns |
|--------|--------|-----------------|
| Demand Forecast | CSV / Excel | `feeder_id`, `timestamp`, `demand_kw` |
| Time Series Forecast | CSV / Excel | Any date column + numeric target column |
| Theft Alerts (batch) | CSV / Excel | `consumer_id`, `zone`, `day_1` … `day_31` |

---

## 5. Environment Variables (Optional)

```bash
# For production database / cache
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
HF_TOKEN=...                    # For Hugging Face deployment
```

---

## 6. Docker Build

```bash
docker build -t vidyut .
docker run -p 7860:7860 vidyut   # Hugging Face Spaces port
```

---

## 7. Key Interactions

| Action | How |
|--------|-----|
| Run forecast | Upload data → select columns → click **RUN FORECAST** |
| Filter alerts | Use sidebar zone/type/confidence filters |
| Export results | Download CSV buttons available in Forecast & Audit tabs |
| Refresh data | Adjust sidebar slider → click **SCAN NETWORK** / **RUN FORECAST** |

---

> **Note:** All forecast & detection results are gated behind explicit button clicks. Changing settings clears previous results — re-run required.
