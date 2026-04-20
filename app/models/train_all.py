"""
Training script — trains the XGBoost model and registers it.
"""

import json
from pathlib import Path
from datetime import datetime

from app.data_pipeline.synthetic_data import save_datasets
from app.models.ml_models.train_xgb import train_xgboost
from app.models.model_registry import register_model
from app.schemas import ModelType


_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_peak_hour", "is_weekend",
    "is_festival", "festival_severity",
    "is_monsoon_season", "monsoon_severity",
    "is_school_hours", "is_market_day",
    "length_m", "speed_limit_kph", "num_lanes", "elevation_change_m",
    "congestion_index", "weather_severity",
    "incident_proximity", "event_proximity", "road_risk_score",
    "road_closure_active", "roadworks_active", "accident_active",
    "historical_speed_kph", "historical_congestion", "speed_reliability",
]


def train_all() -> dict:
    """Train the XGBoost model and return metrics."""
    print("=" * 60)
    print("  DeepRoute — Training Pipeline")
    print("=" * 60)

    print("\n[1/3] Generating synthetic datasets...")
    save_datasets()

    results = {}

    print("\n[2/3] Training XGBoost...")
    xgb_metrics = train_xgboost()
    results["xgboost"] = xgb_metrics
    register_model("xgboost", ModelType.XGBOOST, "1.0.0",
                    {"mae": xgb_metrics["test_mae"], "r2": xgb_metrics["test_r2"]},
                    xgb_metrics["model_path"], _FEATURE_NAMES)

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = Path("data/models/training_report.json")
    report_path.write_text(json.dumps(results, indent=2, default=str))

    print("\n" + "=" * 60)
    print("  Training complete! Model saved to data/models/")
    print("=" * 60)

    return results


if __name__ == "__main__":
    train_all()
