"""
Modular Prediction Interface for DeepRoute.
"""

from typing import Tuple
from app.models.inference import predict
from app.schemas import CombinedFeatureVector, ModelType, PredictionMetadata


def predict_eta_factor(
    features: CombinedFeatureVector,
    model_type: ModelType = ModelType.XGBOOST,
) -> Tuple[float, PredictionMetadata]:
    """
    Modular wrapper around inference.predict engine.
    """
    return predict(features, model_type=model_type)
