"""Models package."""

from .inference import predict
from .model_registry import register_model, get_model_info, list_models, set_default_model
from .train_all import train_all
