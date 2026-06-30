"""
Model Registry — manages trained model metadata, loading, and versioning.

DeepRoute uses a single production model: xgboost.
Legacy registry files may contain historical entries from prior experiments;
those entries are filtered out at load time to keep runtime behavior stable.
"""

import json
from pathlib import Path
from datetime import datetime
from app.schemas import ModelInfo, ModelType, ModelRegistryResponse


_REGISTRY_PATH = Path("data/models/registry.json")
_SUPPORTED_MODEL_NAMES = {"xgboost", "deep_route"}


def _default_registry() -> dict:
    return {"models": {}, "default": "xgboost"}


def _load_registry() -> dict:
    if not _REGISTRY_PATH.exists():
        return _default_registry()

    reg = json.loads(_REGISTRY_PATH.read_text())

    # Keep only supported production models.
    models = reg.get("models", {})
    filtered_models = {
        name: payload
        for name, payload in models.items()
        if name in _SUPPORTED_MODEL_NAMES
    }

    # Ensure the default is valid and deterministic.
    reg["models"] = filtered_models
    if reg.get("default") not in _SUPPORTED_MODEL_NAMES:
        reg["default"] = "xgboost"
    return reg



def _save_registry(reg: dict) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def register_model(
    name: str,
    model_type: ModelType,
    version: str,
    metrics: dict[str, float],
    file_path: str,
    input_features: list[str] | None = None,
) -> ModelInfo:
    """Register a trained model in the registry."""
    if name not in _SUPPORTED_MODEL_NAMES or model_type not in (ModelType.XGBOOST, ModelType.DEEP_ROUTE):
        raise ValueError(f"DeepRoute supports only {_SUPPORTED_MODEL_NAMES} in production")

    reg = _load_registry()

    info = ModelInfo(
        name=name,
        version=version,
        model_type=model_type,
        metrics=metrics,
        trained_at=datetime.now().isoformat(),
        input_features=input_features or [],
        file_path=file_path,
    )
    reg["models"][name] = info.model_dump()
    _save_registry(reg)
    return info



def get_model_info(name: str) -> ModelInfo | None:
    if name not in _SUPPORTED_MODEL_NAMES:
        return None
    reg = _load_registry()
    data = reg["models"].get(name)
    return ModelInfo(**data) if data else None


def list_models() -> ModelRegistryResponse:
    reg = _load_registry()
    models = [ModelInfo(**v) for v in reg["models"].values()]
    return ModelRegistryResponse(
        models=models, default_model=reg.get("default", "xgboost")
    )


def set_default_model(name: str) -> None:
    if name not in _SUPPORTED_MODEL_NAMES:
        raise ValueError(f"DeepRoute supports only {_SUPPORTED_MODEL_NAMES} in production")

    reg = _load_registry()
    if name not in reg["models"]:
        raise ValueError(f"Model '{name}' not found in registry")
    reg["default"] = name
    _save_registry(reg)

