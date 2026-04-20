import json
import time
import numpy as np
from pathlib import Path

from app.data_pipeline.synthetic_data import save_datasets
from app.models.ml_models.train_xgb import train_xgboost
from app.features.feature_builder import build_features
from app.models.inference import predict
from app.schemas import ModelType

def evaluate_models():
    save_datasets()
    print("Datasets generated. Training XGBoost model...")
    
    results = {}
    
    print("\n--- Training XGBoost ---")
    xgb_m = train_xgboost()
    results["XGBoost"] = {"mae": xgb_m.get("test_mae", 0), "r2": xgb_m.get("test_r2", 0)}
    
    print("\n--- Evaluating Latency ---")
    features = build_features(origin_lat=17.3, origin_lon=78.4, departure_time="2026-03-30T10:00:00")
    
    try:
        # warmup
        predict(features, ModelType.XGBOOST)
        
        start = time.time()
        for _ in range(50):
            predict(features, ModelType.XGBOOST)
        end = time.time()
        
        latency_ms = ((end - start) / 50) * 1000
        results["XGBoost"]["latency_ms"] = round(latency_ms, 2)
    except Exception as e:
        results["XGBoost"]["latency_err"] = str(e)

    import pandas as pd
    df = pd.DataFrame(results).T
    print("\n================ BENCHMARK RESULTS ================")
    print(df.to_string())
    print("===================================================")

if __name__ == "__main__":
    evaluate_models()
