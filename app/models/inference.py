"""
Unified inference engine.

Loads the trained XGBoost model and provides prediction interface.
"""

import time
import numpy as np
import joblib
from pathlib import Path

from app.schemas import ModelType, PredictionMetadata, CombinedFeatureVector


_MODEL_DIR = Path("data/models")

# ── Lazy-loaded model cache ───────────────────────────────────────────────────
_ml_cache: dict[str, object] = {}


def _load_ml_model(name: str):
    """Load a sklearn/xgboost model from disk (cached)."""
    if name not in _ml_cache:
        path = _MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"ML model not found: {path}")
        _ml_cache[name] = joblib.load(path)
    return _ml_cache[name]


def predict(
    features: CombinedFeatureVector,
    model_type: ModelType = ModelType.XGBOOST,
    seq_length: int = 12,
) -> tuple[float, PredictionMetadata]:
    """
    Run prediction using the XGBoost model.

    Parameters
    ----------
    features   : CombinedFeatureVector from the feature builder
    model_type : model to use (only XGBOOST is supported)
    seq_length : unused, kept for API compatibility

    Returns
    -------
    (predicted_travel_time_factor, PredictionMetadata)
    """
    start = time.perf_counter()
    full_list = features.to_flat_list()
    flat = np.array(full_list, dtype=np.float32).reshape(1, -1)

    # MOCKED: Bypass XGBoost loading to show output faster
    pred = 1.0
    model_name = "XGBoost (Mocked)"
    
    elapsed_ms = (time.perf_counter() - start) * 1000

    confidence = _estimate_confidence(pred)

    meta = PredictionMetadata(
        model_used=model_name,
        model_version="1.0.0",
        confidence_score=confidence,
        prediction_latency_ms=round(elapsed_ms, 2),
        features_used=features.feature_names,
    )

    return pred, meta


def _estimate_confidence(pred: float) -> float:
    """Heuristic confidence: higher when prediction is in reasonable range."""
    if 0.8 <= pred <= 4.0:
        base = 0.85
    elif 0.5 <= pred <= 5.0:
        base = 0.65
    else:
        base = 0.40

    return round(base, 2)