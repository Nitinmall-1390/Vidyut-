# Graph Report - Vidyut  (2026-05-05)

## Corpus Check
- 59 files · ~32,530 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 702 nodes · 988 edges · 60 communities (47 shown, 13 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 135 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2e9ddeaf`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]

## God Nodes (most connected - your core abstractions)
1. `CacheKeyBuilder` - 41 edges
2. `RedisCache` - 27 edges
3. `ModelRegistry` - 20 edges
4. `_sanitize()` - 14 edges
5. `InferenceCache` - 14 edges
6. `train_engine()` - 13 edges
7. `_LRUMemoryCache` - 13 edges
8. `RuleEngine` - 13 edges
9. `DemandBatchPredictor` - 13 edges
10. `XGBoostTheftClassifier` - 13 edges

## Surprising Connections (you probably didn't know these)
- `train_engine()` --calls--> `ModelRegistry`  [INFERRED]
  scripts/train_all.py → src/models/versioning.py
- `train_engine()` --calls--> `generate_multi_feeder_demand()`  [INFERRED]
  scripts/train_all.py → src/data/synthetic_generator.py
- `train_engine()` --calls--> `load_or_generate_weather()`  [INFERRED]
  scripts/train_all.py → src/data/ingestion.py
- `train_engine()` --calls--> `augment_theft_patterns()`  [INFERRED]
  scripts/train_all.py → src/data/synthetic_generator.py
- `train_engine()` --calls--> `DemandEnsemble`  [INFERRED]
  scripts/train_all.py → src/models/part_a/ensemble.py

## Communities (60 total, 13 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (36): lifespan(), Startup/shutdown hooks., Run full theft detection pipeline.          Parameters         ----------, Generate ensemble demand predictions for all feeders in df.          Parameter, _cache_key(), clear_model_cache(), get_model_loader(), load_demand_ensemble() (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (33): CacheKeyBuilder, CacheNamespace, _json_default(), _normalize_datetime(), =========================================================================== VID, Build cache key for theft probability score., Build cache key for anomaly score (LSTM-AE + IsoForest intersection)., Build cache key for SHAP explanation (expensive — high TTL). (+25 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (40): BaseModel, compute_confidence_score(), ConfidenceScorer, VIDYUT Confidence Scorer ======================================================, Compute a 0-100 confidence score for a single theft prediction.      The score, Return HIGH / MEDIUM / LOW based on score thresholds., Batch confidence scorer for the theft detection pipeline., Attach confidence_score and confidence_label columns to a predictions DataFrame. (+32 more)

### Community 3 - "Community 3"
Cohesion: 0.07
Nodes (27): AuditEvent, get_audit_db(), get_audit_engine(), VIDYUT Audit Database =========================================================, Return (and cache) the audit database engine, creating tables if needed., Immutable audit event record.      Every model prediction, generated alert, an, AuditLogger, get_audit_logger() (+19 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (22): DemandBatchPredictor, VIDYUT Batch Predictor ========================================================, Runs demand forecasting for a list of feeders.      Returns a combined DataFra, InferenceCache, _make_cache_key(), VIDYUT Inference Cache ========================================================, Store a prediction result in the cache.          Returns True on success, Fals, Delete a specific cached result. (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (13): create_app(), =========================================================================== VID, Application factory — used by Uvicorn entrypoint and tests., client(), rate_limited_client(), =========================================================================== VID, TestAnomaly, TestDemand (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.1
Nodes (15): VIDYUT Part B — Theft Ring Detector ===========================================, Run Louvain community detection on the graph.          Returns list of sets, e, Identify theft rings from detected communities.          Parameters         -, End-to-end pipeline: build graph → detect communities → identify rings., Export graph data for Streamlit network visualisation.          Returns, Detects organised theft rings via community detection on a consumer graph., Build the consumer network graph.          Parameters         ----------, TheftRingDetector (+7 more)

### Community 7 - "Community 7"
Cohesion: 0.1
Nodes (12): VIDYUT Part B — XGBoost 3-Class Theft Classifier ==============================, Predict class labels and probabilities.          Returns         -------, Predict and return a DataFrame with consumer_id + class info., Return full multi-class metrics., XGBoost multi-class classifier for theft type attribution.      Pipeline: Robu, Train the classifier with SMOTE augmentation.          Parameters         ---, XGBoostTheftClassifier, binary_theft_metrics() (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.12
Nodes (15): ConsumerRuleResult, _count_low_month_windows(), _max_consecutive(), _max_consecutive_equal(), _max_mom_drop_pct(), VIDYUT Rule Engine ============================================================, Evaluate all rules for one consumer.          Parameters         ----------, Apply rules to all Stage-2 flagged consumers and attach rule columns. (+7 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (9): _LSTMAutoencoder, LSTMAutoencoderModel, VIDYUT Part B — LSTM Autoencoder for Anomaly Detection ========================, Train the autoencoder on normal (unlabelled) consumption sequences.          P, Compute per-sequence mean squared reconstruction error., Predict anomaly flags for a set of sequences.          Returns         ------, Aggregate sequence-level anomaly scores to consumer level.          A consumer, Sequence-to-sequence LSTM Autoencoder.      Input:  (batch, seq_len, n_feature (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.16
Nodes (8): Production-grade cache for Vidyut. Wraps Redis with intelligent fallback     to, Delete a single key. Returns True if deleted., Delete all keys matching a glob pattern. Used for invalidating         an entir, Invalidate all keys in a namespace (optionally only current version)., Invalidate all keys for a specific entity in a namespace., Delete every Vidyut key. DESTRUCTIVE — use with care., Switch to memory backend after Redis failure., RedisCache

### Community 11 - "Community 11"
Cohesion: 0.17
Nodes (16): compute_all_demand_metrics(), interval_coverage(), mae(), mape(), pinball_loss(), r2(), VIDYUT Demand Forecasting Metrics =============================================, Mean Absolute Percentage Error. Returns value in [0, 100]. (+8 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (10): BaseHTTPMiddleware, AuthMiddleware, _extract_key(), =========================================================================== VID, API key authentication., _secure_compare(), _client_id(), RateLimitMiddleware (+2 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (11): _compute_js_divergence(), _compute_psi(), detect_drift(), DriftReport, DriftResult, VIDYUT Data Drift Detector ====================================================, Compute Jensen-Shannon divergence between two distributions., Detect distribution drift between reference and current DataFrames.      Param (+3 more)

### Community 14 - "Community 14"
Cohesion: 0.18
Nodes (9): check_demand_dataframe(), check_sgcc_dataframe(), QualityCheck, QualityReport, VIDYUT Data Quality Checks ====================================================, Run quality checks on a SGCC theft dataset., Result of a single data quality assertion., Aggregated quality check results for a dataset. (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (7): ProphetModel, VIDYUT Part A — Prophet Forecasting Model =====================================, Fit the Prophet model.          Parameters         ----------         df : p, Generate demand forecasts for a future horizon DataFrame.          Parameters, Evaluate on held-out test set. Returns metrics dict., Return component DataFrame (trend, seasonalities) for decomposition plot., Prophet-based 15-minute electricity demand forecaster.      One instance per f

### Community 16 - "Community 16"
Cohesion: 0.13
Nodes (8): EvaluationReport, VIDYUT Evaluation Report Generator ============================================, Collects metrics and metadata from a model evaluation run and persists     them, Add a named metrics dict to the report., Attach arbitrary metadata key-value pairs., Add a free-text section (for HTML report narrative)., Persist report as JSON., Persist report as a self-contained HTML file.

### Community 17 - "Community 17"
Cohesion: 0.21
Nodes (13): _download_sgcc(), fetch_nasa_power(), load_or_generate_weather(), load_sgcc(), VIDYUT Data Ingestion =========================================================, Generate a plausible SGCC-shaped DataFrame for testing without real data., Fetch daily weather data from the NASA POWER API.      Returns     -------, Attempt NASA POWER fetch; fall back to synthetic weather on failure. (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.19
Nodes (6): LGBMForecastModel, Train point + quantile models.          Parameters         ----------, Generate demand predictions.          Returns         -------         pd.Dat, Evaluate on test set. Returns full metrics dict., Return feature importances as a sorted DataFrame., LightGBM point + interval forecasting model.      Fits three separate boosters

### Community 19 - "Community 19"
Cohesion: 0.18
Nodes (9): _build_forecast_df(), load_ensemble(), VIDYUT Intelligence Dashboard — Self-Contained Streamlit App No FastAPI depende, Run demand forecast. Returns (summary_dict, points_list, error_msg).     Falls, Score a consumer for theft risk. Returns result dict., Load trained DemandEnsemble from disk. Returns (ensemble, error_msg)., Build a properly feature-engineered DataFrame ready for ensemble.predict()., run_forecast() (+1 more)

### Community 20 - "Community 20"
Cohesion: 0.19
Nodes (10): build_demand_features(), build_lstm_sequences(), build_theft_aggregate_features(), VIDYUT Feature Engineering ====================================================, Compute aggregate statistical features per consumer for theft detection., Build sliding-window sequences for LSTM Autoencoder training.      Parameters, Build a full feature matrix for demand forecasting.      Parameters     -----, VIDYUT Data Integration Pipeline =============================================== (+2 more)

### Community 21 - "Community 21"
Cohesion: 0.15
Nodes (7): VIDYUT SHAP Explainer =========================================================, Compute mean absolute SHAP values across a batch of predictions.          Retu, Render a SHAP waterfall chart using matplotlib.          Returns matplotlib Fi, SHAP explanation engine.      Supports TreeExplainer for LightGBM and XGBoost, Initialise the SHAP TreeExplainer with background data.          Parameters, Generate SHAP explanation for a single prediction.          Returns         -, SHAPExplainer

### Community 22 - "Community 22"
Cohesion: 0.17
Nodes (6): DemandEnsemble, VIDYUT Part A — Prophet + LightGBM Ensemble ===================================, Evaluate ensemble on a held-out test set., Weighted ensemble: yhat = 0.4 × Prophet + 0.6 × LightGBM      Prediction inter, Train both component models., Generate ensemble predictions.          Returns         -------         pd.D

### Community 23 - "Community 23"
Cohesion: 0.2
Nodes (7): BaseSettings, get_settings(), VIDYUT Settings ===============================================================, Create all required data directories if they don't exist., Return a cached singleton Settings instance., All configuration drawn from .env / environment variables.     Pydantic validat, Settings

### Community 24 - "Community 24"
Cohesion: 0.2
Nodes (5): cached(), get_cache(), =========================================================================== VID, Returns the global RedisCache singleton, initialized from environment vars., Decorator to cache a function's output using a key derived from its args.

### Community 25 - "Community 25"
Cohesion: 0.25
Nodes (3): _LRUMemoryCache, Thread-safe in-memory LRU cache used when Redis is unavailable.     Stores (val, Glob-style pattern deletion. Supports trailing '*' wildcards.

### Community 26 - "Community 26"
Cohesion: 0.25
Nodes (5): DemandDataLoader, Return all feature columns (exclude timestamp, demand_kw, ids)., Loads, cleans, and prepares demand forecasting data.      Attributes     ----, Execute the full pipeline: ingest → clean → features → split., Split 70/15/15 by time (no random shuffling — time series rule).

### Community 27 - "Community 27"
Cohesion: 0.32
Nodes (5): _infer_ttl(), Store a value with optional TTL. Returns True on success., Bulk-store multiple {key: value} pairs. Returns count successfully set., Cache-aside pattern: return cached value, or compute+cache if missing., _serialize()

### Community 28 - "Community 28"
Cohesion: 0.25
Nodes (7): profile_dataframe(), profile_demand_series(), profile_theft_dataset(), VIDYUT Data Profiling =========================================================, Demand-specific profiling: seasonality summary, peak analysis,     per-feeder s, Theft-specific profiling: class balance, consumption distribution     by label,, Generate a comprehensive statistical profile of a DataFrame.      Returns

### Community 29 - "Community 29"
Cohesion: 0.25
Nodes (7): clean_demand_df(), VIDYUT Data Cleaning ==========================================================, Remove duplicate rows by timestamp, keeping specified occurrence., Upsample or downsample demand data to 15-minute intervals.     Uses mean for do, Clean a raw demand time-series DataFrame.      Steps:     1. Parse and sort t, remove_duplicate_timestamps(), resample_to_15min()

### Community 30 - "Community 30"
Cohesion: 0.33
Nodes (4): _deserialize(), Retrieve a value by key. Returns `default` on miss or error., Check whether a key exists (without fetching value)., Bulk-fetch multiple keys. Returns {key: value} for hits only.

### Community 31 - "Community 31"
Cohesion: 0.29
Nodes (4): VIDYUT Data Loader ============================================================, Loads, cleans, and prepares theft detection data.      Attributes     -------, Stratified train/test split of aggregate features., TheftDataLoader

### Community 32 - "Community 32"
Cohesion: 0.53
Nodes (5): _load_or_synth_demand(), _load_or_synth_ring(), _load_or_synth_theft(), main(), =========================================================================== VID

### Community 33 - "Community 33"
Cohesion: 0.4
Nodes (3): CacheMetrics, Reset metrics counters (e.g., after deployment)., Tracks cache performance counters.

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (3): Close Redis connection pool gracefully., Drop the singleton (mainly for tests)., reset_cache()

### Community 36 - "Community 36"
Cohesion: 0.33
Nodes (5): DemandSchema, VIDYUT Feature Configuration ==================================================, Column schema expected by the demand pipeline., Column schema expected by the theft detection pipeline., TheftSchema

### Community 37 - "Community 37"
Cohesion: 0.4
Nodes (5): generate_feeder_demand(), generate_multi_feeder_demand(), VIDYUT Synthetic Data Generator ===============================================, Generate demand data for multiple synthetic Bangalore feeders.      Returns, Generate synthetic 15-min demand data for a single feeder.      Returns     -

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (5): clean_sgcc_df(), Clean the SGCC dataset.      Steps:     1. Isolate day_* columns     2. Clip, Execute full theft data pipeline., augment_theft_patterns(), Augment a cleaned SGCC DataFrame with synthetic anomaly patterns     for techni

### Community 39 - "Community 39"
Cohesion: 0.4
Nodes (4): _ColouredFormatter, get_logger(), VIDYUT Logging Utility ========================================================, Return a named logger with console + optional rotating file handler.      Para

## Knowledge Gaps
- **258 isolated node(s):** `Insert the HOW IT WORKS feature explanation block into dashboard/app.py. Run: py`, `=========================================================================== VID`, `=========================================================================== VID`, `Startup/shutdown hooks.`, `Application factory — used by Uvicorn entrypoint and tests.` (+253 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `lifespan()` connect `Community 0` to `Community 24`, `Community 3`, `Community 5`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Why does `CacheKeyBuilder` connect `Community 1` to `Community 33`, `Community 2`, `Community 10`, `Community 24`, `Community 25`?**
  _High betweenness centrality (0.165) - this node is a cross-community bridge._
- **Why does `TheftBatchPredictor` connect `Community 2` to `Community 0`, `Community 4`, `Community 6`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `CacheKeyBuilder` (e.g. with `ConsumerSequence` and `AnomalyDetectRequest`) actually correct?**
  _`CacheKeyBuilder` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `RedisCache` (e.g. with `CacheKeyBuilder` and `CacheNamespace`) actually correct?**
  _`RedisCache` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `ModelRegistry` (e.g. with `ModelLoader` and `train_engine()`) actually correct?**
  _`ModelRegistry` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Insert the HOW IT WORKS feature explanation block into dashboard/app.py. Run: py`, `=========================================================================== VID`, `=========================================================================== VID` to the rest of the system?**
  _258 weakly-connected nodes found - possible documentation gaps or missing edges._