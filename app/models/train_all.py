"""
Master training script — trains all ML and DL models and registers them.
"""

import json
from pathlib import Path
from datetime import datetime

from app.data_pipeline.synthetic_data import save_datasets
from app.models.ml_models.train_rf import train_random_forest
from app.models.ml_models.train_gbm import train_gbm
from app.models.ml_models.train_xgb import train_xgboost
from app.models.dl_models.train_lstm import train_lstm
from app.models.dl_models.train_transformer import train_transformer
try:
    from app.models.dl_models.train_gnn import train_gnn
    from app.models.dl_models.train_hybrid import train_hybrid
except ImportError:
    train_gnn = None
    train_hybrid = None

from app.models.model_registry import register_model
from app.schemas import ModelType


_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_peak_hour", "is_weekend",
    "length_m", "speed_limit_kph", "num_lanes", "elevation_change_m",
    "congestion_index", "weather_severity",
    "incident_proximity", "event_proximity", "road_risk_score",
]


def train_all() -> dict:
    """Train every model and return combined metrics."""
    print("=" * 60)
    print("  DeepRoute — Full Training Pipeline")
    print("=" * 60)

    print("\n[1/8] Generating synthetic datasets...")
    save_datasets()

    results = {}

    # ── ML Models ─────────────────────────────────────────────────────────────

    # print("\n[2/8] Training Random Forest...")
    # rf_metrics = train_random_forest()
    # results["random_forest"] = rf_metrics
    # register_model("random_forest", ModelType.RANDOM_FOREST, "1.0.0",
    #                 {"mae": rf_metrics["test_mae"], "r2": rf_metrics["test_r2"]},
    #                 rf_metrics["model_path"], _FEATURE_NAMES)

    # print("\n[3/8] Training Gradient Boosting...")
    # gbm_metrics = train_gbm()
    # results["gbm"] = gbm_metrics
    # register_model("gbm", ModelType.GRADIENT_BOOSTING, "1.0.0",
    #                 {"mae": gbm_metrics["test_mae"], "r2": gbm_metrics["test_r2"]},
    #                 gbm_metrics["model_path"], _FEATURE_NAMES)

    print("\n[4/8] Training XGBoost...")
    xgb_metrics = train_xgboost()
    results["xgboost"] = xgb_metrics
    register_model("xgboost", ModelType.XGBOOST, "1.0.0",
                    {"mae": xgb_metrics["test_mae"], "r2": xgb_metrics["test_r2"]},
                    xgb_metrics["model_path"], _FEATURE_NAMES)

    # ── DL Models ─────────────────────────────────────────────────────────────

    print("\n[5/8] Training LSTM...")
    lstm_metrics = train_lstm()
    results["lstm"] = lstm_metrics
    register_model("lstm", ModelType.LSTM, "1.0.0",
                    {"val_loss": lstm_metrics["best_val_loss"]},
                    lstm_metrics["model_path"], _FEATURE_NAMES)

    # print("\n[6/8] Training Transformer...")
    # tx_metrics = train_transformer()
    # results["transformer"] = tx_metrics
    # register_model("transformer", ModelType.TRANSFORMER, "1.0.0",
    #                 {"val_loss": tx_metrics["best_val_loss"]},
    #                 tx_metrics["model_path"], _FEATURE_NAMES)

    print("\n[7/8] Skipping GNN & Hybrid (not required for active components)")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = Path("data/models/training_report.json")
    report_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print("  Training complete! All models saved to data/models/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train_all()
