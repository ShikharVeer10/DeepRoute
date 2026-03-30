"""
Unified inference engine.

Loads trained ML/DL models and provides a consistent prediction interface.
Supports ensemble prediction (weighted average of all available models).
Simplified: GNN and Hybrid models removed for reliability.
"""

import time
import numpy as np
import joblib
import torch
from pathlib import Path

from app.schemas import ModelType, PredictionMetadata, CombinedFeatureVector
from app.models.dl_models.lstm_model import TrafficLSTM
from app.models.dl_models.transformer_model import TrafficTransformer


_MODEL_DIR = Path("data/models")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Lazy-loaded model cache ───────────────────────────────────────────────────
_ml_cache: dict[str, object] = {}
_dl_cache: dict[str, torch.nn.Module] = {}


def _load_ml_model(name: str):
    """Load a sklearn/xgboost model from disk (cached)."""
    if name not in _ml_cache:
        path = _MODEL_DIR / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"ML model not found: {path}")
        _ml_cache[name] = joblib.load(path)
    return _ml_cache[name]


def _load_dl_model(name: str, model_cls, **kwargs) -> torch.nn.Module:
    """Load a PyTorch model from disk (cached)."""
    if name not in _dl_cache:
        path = _MODEL_DIR / f"{name}.pth"
        if not path.exists():
            raise FileNotFoundError(f"DL model not found: {path}")
        model = model_cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location=_DEVICE, weights_only=True))
        model.to(_DEVICE)
        model.eval()
        _dl_cache[name] = model
    return _dl_cache[name]


def predict(
    features: CombinedFeatureVector,
    model_type: ModelType = ModelType.ENSEMBLE,
    seq_length: int = 12,
) -> tuple[float, PredictionMetadata]:
    """
    Run prediction using the specified model type.

    Parameters
    ----------
    features   : CombinedFeatureVector from the feature builder
    model_type : which model to use
    seq_length : sequence length for DL models (repeats the feature vector)

    Returns
    -------
    (predicted_travel_time_factor, PredictionMetadata)
    """
    start = time.perf_counter()
    full_list = features.to_flat_list()
    # The models were trained on 15 features. Extract exactly those indices.
    idx = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    compat_list = [full_list[i] for i in idx]
    flat = np.array(compat_list, dtype=np.float32).reshape(1, -1)
    n_features = flat.shape[1]

    if model_type == ModelType.RANDOM_FOREST:
        model = _load_ml_model("random_forest")
        pred = float(model.predict(flat)[0])
        model_name = "RandomForest"

    elif model_type == ModelType.GRADIENT_BOOSTING:
        model = _load_ml_model("gbm")
        pred = float(model.predict(flat)[0])
        model_name = "GradientBoosting"

    elif model_type == ModelType.XGBOOST:
        model = _load_ml_model("xgboost")
        pred = float(model.predict(flat)[0])
        model_name = "XGBoost"

    elif model_type == ModelType.LSTM:
        model = _load_dl_model("lstm", TrafficLSTM, input_size=n_features)
        seq = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).repeat(1, seq_length, 1).to(_DEVICE)
        with torch.no_grad():
            pred = float(model(seq).item())
        model_name = "TrafficLSTM"

    elif model_type == ModelType.TRANSFORMER:
        model = _load_dl_model("transformer", TrafficTransformer, input_size=n_features)
        seq = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).repeat(1, seq_length, 1).to(_DEVICE)
        with torch.no_grad():
            pred = float(model(seq).item())
        model_name = "TrafficTransformer"

    elif model_type in (ModelType.GNN, ModelType.GNN_LSTM):
        # Fall back to ensemble for GNN-type models (they're complex and not trained)
        pred, model_name = _ensemble_predict(flat, n_features, seq_length)

    elif model_type == ModelType.ENSEMBLE:
        pred, model_name = _ensemble_predict(flat, n_features, seq_length)

    else:
        pred, model_name = _ensemble_predict(flat, n_features, seq_length)

    elapsed_ms = (time.perf_counter() - start) * 1000

    confidence = _estimate_confidence(pred, model_type)

    meta = PredictionMetadata(
        model_used=model_name,
        model_version="1.0.0",
        confidence_score=confidence,
        prediction_latency_ms=round(elapsed_ms, 2),
        features_used=[features.feature_names[i] for i in idx],
    )

    return pred, meta


def _ensemble_predict(
    flat: np.ndarray,
    n_features: int,
    seq_length: int,
) -> tuple[float, str]:
    """
    Weighted-average ensemble of all available models.
    Gracefully skips models that aren't trained yet.
    """
    predictions = []
    weights = []

    # ── ML models ─────────────────────────────────────────────────────────────
    ml_models = [
        ("random_forest", 1.0),
        ("gbm", 1.2),
        ("xgboost", 1.3),
    ]
    for name, weight in ml_models:
        try:
            model = _load_ml_model(name)
            predictions.append(float(model.predict(flat)[0]))
            weights.append(weight)
        except FileNotFoundError:
            pass

    # ── DL sequential models (LSTM, Transformer) ─────────────────────────────
    seq_configs = [
        ("lstm", TrafficLSTM, {"input_size": n_features}, 1.5),
        ("transformer", TrafficTransformer, {"input_size": n_features}, 1.4),
    ]
    for name, cls, kwargs, weight in seq_configs:
        try:
            model = _load_dl_model(name, cls, **kwargs)
            seq = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).repeat(1, seq_length, 1).to(_DEVICE)
            with torch.no_grad():
                p = float(model(seq).item())
            predictions.append(p)
            weights.append(weight)
        except FileNotFoundError:
            pass

    if not predictions:
        return 1.5, "Fallback"

    weights_arr = np.array(weights)
    weights_arr /= weights_arr.sum()
    ensemble_pred = float(np.dot(predictions, weights_arr))

    return ensemble_pred, f"Ensemble({len(predictions)} models)"


def _estimate_confidence(pred: float, model_type: ModelType) -> float:
    """Heuristic confidence: higher when prediction is in reasonable range."""
    if 0.8 <= pred <= 4.0:
        base = 0.85
    elif 0.5 <= pred <= 5.0:
        base = 0.65
    else:
        base = 0.40

    if model_type == ModelType.ENSEMBLE:
        base = min(1.0, base + 0.10)
    elif model_type == ModelType.TRANSFORMER:
        base = min(1.0, base + 0.05)

    return round(base, 2)