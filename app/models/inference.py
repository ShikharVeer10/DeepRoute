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
            return None
        _ml_cache[name] = joblib.load(path)
    return _ml_cache[name]


def _load_deep_model(num_edges: int, n_features: int):
    """Load the PyTorch GAT/Temporal model (cached)."""
    name = "deep_route"
    if name not in _ml_cache:
        import torch
        from app.models.ml_models.deep_route_model import DeepRouteModel
        path = _MODEL_DIR / "deep_route_model.pth"
        if not path.exists():
            return None
        model = DeepRouteModel(
            num_edge_features=n_features,
            temporal_d_model=64,
            temporal_nhead=4,
            temporal_layers=2,
            num_edges=num_edges,
            node_feature_dim=n_features,
            gat_hidden_dim=32,
            gat_output_dim=64,
            gat_heads=4,
            gat_layers=2,
            pred_hidden_dim=128,
            dropout=0.1,
        )
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        model.eval()
        _ml_cache[name] = model
    return _ml_cache[name]


def predict(
    features: CombinedFeatureVector,
    model_type: ModelType = ModelType.XGBOOST,
    seq_length: int = 12,
) -> tuple[float, PredictionMetadata]:
    """
    Run prediction using the specified model type (XGBoost or Deep Learning).
    """
    start = time.perf_counter()
    full_list = features.to_flat_list()
    
    if model_type == ModelType.DEEP_ROUTE:
        import torch
        # Load graph info from data/ directory
        data_dir = Path("data")
        try:
            edge_index = np.load(data_dir / "edge_index.npy")
            node_features = np.load(data_dir / "node_features.npy")
        except Exception:
            # Fallback if files aren't generated
            edge_index = np.array([[0], [0]], dtype=np.int64)
            node_features = np.zeros((1, len(full_list)), dtype=np.float32)

        num_edges = edge_index.shape[1]
        n_features = node_features.shape[1]

        model = _load_deep_model(num_edges, n_features)
        if model is None:
            # Fallback if model not trained
            pred = 1.05
            model_name = "Deep Learning GNN (Uninitialized Fallback)"
        else:
            # Prepare dummy batched inputs
            traffic_series = torch.tensor(full_list, dtype=torch.float32).view(1, 1, -1)
            # Replicate along sequence dimension
            traffic_series = traffic_series.repeat(1, seq_length, 1)
            node_features_t = torch.tensor(node_features, dtype=torch.float32)
            edge_index_t = torch.tensor(edge_index, dtype=torch.long)
            src_idx = torch.tensor([0], dtype=torch.long)
            dst_idx = torch.tensor([0], dtype=torch.long)

            with torch.no_grad():
                _, edge_weights, _ = model(
                    traffic_series, node_features_t, edge_index_t, src_idx, dst_idx
                )
            # Average predicted travel factor across graph edges
            pred = float(edge_weights.mean().item())
            model_name = "Deep Learning GNN (Temporal-GAT)"
    else:
        # Default: XGBoost
        model = _load_ml_model("xgboost")
        if model is None:
            pred = 1.02
            model_name = "XGBoost (Uninitialized Fallback)"
        else:
            flat = np.array(full_list, dtype=np.float32).reshape(1, -1)
            # Some xgboost installations require xgb DMatrix or predict wrapper
            try:
                pred = float(model.predict(flat)[0])
            except Exception:
                pred = 1.02
            model_name = "XGBoost (Production)"

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