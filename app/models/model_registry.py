"""
Model Registry — manages trained model metadata, loading, and versioning.
"""

import json
from pathlib import Path
from datetime import datetime
from app.schemas import ModelInfo, ModelType, ModelRegistryResponse


_REGISTRY_PATH = Path("data/models/registry.json")


def _load_registry() -> dict:
    if _REGISTRY_PATH.exists():
        return json.loads(_REGISTRY_PATH.read_text())
    return {"models": {}, "default": "gbm"}


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
    reg = _load_registry()
    data = reg["models"].get(name)
    return ModelInfo(**data) if data else None


def list_models() -> ModelRegistryResponse:
    reg = _load_registry()
    models = [ModelInfo(**v) for v in reg["models"].values()]
    return ModelRegistryResponse(models=models, default_model=reg.get("default", "gbm"))


def set_default_model(name: str) -> None:
    reg = _load_registry()
    if name not in reg["models"]:
        raise ValueError(f"Model '{name}' not found in registry")
    reg["default"] = name
    _save_registry(reg)
