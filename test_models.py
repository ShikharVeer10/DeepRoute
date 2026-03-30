import json
import time
import numpy as np
from pathlib import Path

from app.data_pipeline.synthetic_data import save_datasets
from app.models.ml_models.train_rf import train_random_forest
from app.models.ml_models.train_gbm import train_gbm
from app.models.ml_models.train_xgb import train_xgboost
from app.models.dl_models.train_lstm import train_lstm
from app.models.dl_models.train_transformer import train_transformer
from app.features.feature_builder import build_features
from app.models.inference import predict
from app.schemas import ModelType

def evaluate_models():
    save_datasets()
    print("Datasets generated. Training models...")
    
    results = {}
    
    print("\n--- Training RF ---")
    rf_m = train_random_forest()
    results["Random Forest"] = {"mae": rf_m.get("test_mae", 0), "r2": rf_m.get("test_r2", 0)}
    
    print("\n--- Training GBM ---")
    gbm_m = train_gbm()
    results["GBM"] = {"mae": gbm_m.get("test_mae", 0), "r2": gbm_m.get("test_r2", 0)}
    
    print("\n--- Training XGBoost ---")
    xgb_m = train_xgboost()
    results["XGBoost"] = {"mae": xgb_m.get("test_mae", 0), "r2": xgb_m.get("test_r2", 0)}
    
    print("\n--- Training LSTM ---")
    lstm_m = train_lstm()
    results["LSTM"] = {"val_loss": lstm_m.get("best_val_loss", 0), "mae": lstm_m.get("test_mae", lstm_m.get("best_val_loss", 0))}
    
    print("\n--- Training Transformer ---")
    tx_m = train_transformer()
    results["Transformer"] = {"val_loss": tx_m.get("best_val_loss", 0), "mae": tx_m.get("test_mae", tx_m.get("best_val_loss", 0))}
    
    print("\n--- Evaluating Latency ---")
    # Generate random features for inference
    features = build_features(origin_lat=17.3, origin_lon=78.4, departure_time="2026-03-30T10:00:00")
    
    models = [
        ("Random Forest", ModelType.RANDOM_FOREST),
        ("GBM", ModelType.GRADIENT_BOOSTING),
        ("XGBoost", ModelType.XGBOOST),
        ("LSTM", ModelType.LSTM),
        ("Transformer", ModelType.TRANSFORMER)
    ]
    
    for name, mtype in models:
        try:
            # warmup
            predict(features, mtype)
            
            start = time.time()
            for _ in range(50):
                predict(features, mtype)
            end = time.time()
            
            latency_ms = ((end - start) / 50) * 1000
            results[name]["latency_ms"] = round(latency_ms, 2)
        except Exception as e:
            results[name]["latency_err"] = str(e)

    import pandas as pd
    df = pd.DataFrame(results).T
    print("\n================ BENCHMARK RESULTS ================")
    print(df.to_string())
    print("===================================================")

if __name__ == "__main__":
    evaluate_models()
