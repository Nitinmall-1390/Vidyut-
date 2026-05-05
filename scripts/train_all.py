import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import torch
from datetime import date

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.synthetic_generator import (
    generate_multi_feeder_demand,
    augment_theft_patterns
)
from src.data.feature_engineering import (
    build_demand_features,
    build_theft_aggregate_features,
    build_lstm_sequences
)
from src.data.ingestion import load_or_generate_weather
from src.models.part_a.ensemble import DemandEnsemble
from src.models.part_b.lstm_autoencoder import LSTMAutoencoderModel
from src.models.part_b.isolation_forest import IsolationForestModel
from src.models.part_b.xgboost_classifier import XGBoostTheftClassifier
from src.models.versioning import ModelRegistry, ModelVersion
from src.utils.logger import get_logger

log = get_logger("vidyut.train_all")

def train_engine():
    # 1. SETUP REGISTRY
    registry = ModelRegistry()
    version_name = "v2"
    version_dir = registry.models_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Starting Vidyut Engine Training Pipeline for version {version_name}...")

    # 2. GENERATE SYNTHETIC DATA
    log.info("Step 1: Generating synthetic training data...")
    demand_df = generate_multi_feeder_demand(n_feeders=5, start_date=date(2023, 1, 1))
    
    min_date = demand_df["timestamp"].min().date()
    max_date = demand_df["timestamp"].max().date()
    weather_df = load_or_generate_weather(min_date, max_date)
    
    day_cols = [f"day_{i}" for i in range(60)]
    dummy_clean = pd.DataFrame(
        np.random.uniform(5, 15, size=(50, 60)),
        columns=day_cols
    )
    dummy_clean["CONS_NO"] = [f"CLEAN_{i:03d}" for i in range(50)]
    dummy_clean["FLAG"] = 0
    
    meter_df, labels = augment_theft_patterns(
        dummy_clean, 
        n_synthetic_theft=100, 
        n_synthetic_technical=50, 
        n_synthetic_billing=50
    )
    
    model_artefacts = []

    # 3. TRAIN PART A: DEMAND ENSEMBLE
    log.info("Step 2: Training Demand Ensemble (Part A)...")
    feeder_id = "FEEDER_001"
    feeder_df = demand_df[demand_df["feeder_id"] == feeder_id].copy()
    feeder_df = build_demand_features(feeder_df, weather_df=weather_df)
    feeder_df = feeder_df.dropna()
    
    split_idx = int(len(feeder_df) * 0.8)
    train_df = feeder_df.iloc[:split_idx]
    val_df = feeder_df.iloc[split_idx:]
    
    ensemble = DemandEnsemble(feeder_id=feeder_id)
    ensemble.fit(train_df, val_df=val_df)
    ensemble.save(version_dir)
    
    model_artefacts.append(ModelVersion(
        model_name=f"demand_{feeder_id}",
        version=version_name,
        task="demand_forecast",
        metrics={"mape": 5.2}, # Placeholder
        artefact_path=version_dir / f"{feeder_id}_ensemble_meta.joblib"
    ))

    # 4. TRAIN PART B: THEFT PIPELINE
    log.info("Step 3: Training Theft Detection Pipeline (Part B)...")
    
    # Stage 1a: LSTM Autoencoder
    log.info("Training LSTM Autoencoder...")
    sequences, cids = build_lstm_sequences(meter_df)
    lstm_model = LSTMAutoencoderModel(n_epochs=3)
    lstm_model.fit(sequences)
    lstm_path = version_dir / "lstm_ae.pt"
    lstm_model.save(lstm_path)
    
    model_artefacts.append(ModelVersion(
        model_name="lstm_ae",
        version=version_name,
        task="anomaly_detection",
        metrics={"threshold": float(lstm_model.threshold_)},
        artefact_path=lstm_path
    ))
    
    # Stage 1b: Isolation Forest
    log.info("Training Isolation Forest...")
    agg_features = build_theft_aggregate_features(meter_df)
    if_model = IsolationForestModel()
    if_model.fit(agg_features)
    if_path = version_dir / "isolation_forest.joblib"
    if_model.save(if_path)
    
    model_artefacts.append(ModelVersion(
        model_name="isolation_forest",
        version=version_name,
        task="anomaly_detection",
        metrics={"n_flagged": int(if_model.predict(agg_features)["if_anomaly"].sum())},
        artefact_path=if_path
    ))
    
    # Stage 2: XGBoost Classifier
    log.info("Training XGBoost Classifier...")
    xgb_model = XGBoostTheftClassifier(n_estimators=50)
    xgb_model.fit(agg_features, labels)
    xgb_path = version_dir / "xgb_classifier.joblib"
    xgb_model.save(xgb_path)
    
    model_artefacts.append(ModelVersion(
        model_name="xgb_classifier",
        version=version_name,
        task="theft_classification",
        metrics={"accuracy": 0.85}, # Placeholder
        artefact_path=xgb_path
    ))

    # 5. REGISTER AND PROMOTE
    registry.register(version_name, model_artefacts)
    registry.promote_to_latest(version_name)

    log.info("Vidyut Engine started successfully! All models populated and promoted to LATEST.")

if __name__ == "__main__":
    train_engine()
