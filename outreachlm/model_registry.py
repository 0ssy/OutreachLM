from typing import Any, Callable

import torch.nn as nn

from outreachlm.model import OutreachModel
from outreachlm.v4_model import OutreachV4Model


ModelFactory = Callable[[dict[str, Any]], nn.Module]
MODEL_REGISTRY: dict[str, ModelFactory] = {}


def register_model(model_type: str, factory: ModelFactory) -> None:
    if not model_type:
        raise ValueError("model_type cannot be empty.")
    if not callable(factory):
        raise TypeError("factory must be callable.")
    if model_type in MODEL_REGISTRY:
        raise ValueError(f"Model type already registered: {model_type}")
    MODEL_REGISTRY[model_type] = factory


def _create_legacy_v1(model_config: dict[str, Any]) -> nn.Module:
    return OutreachModel(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config.get("num_layers", 1),
        num_heads=model_config.get("num_heads", 4),
    )


def _create_v4(model_config: dict[str, Any]) -> nn.Module:
    return OutreachV4Model(
        vocab_size=model_config["vocab_size"],
        context_length=model_config.get("context_length", 256),
        embedding_dim=model_config.get("embedding_dim", 256),
        num_layers=model_config.get("num_layers", 4),
        num_heads=model_config.get("num_heads", 8),
        ffn_dim=model_config.get("ffn_dim", 684),
    )


def create_model(model_type: str, model_config: dict[str, Any]) -> nn.Module:
    factory = MODEL_REGISTRY.get(model_type)
    if factory is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_type '{model_type}'. Registered model types: {available}"
        )
    return factory(model_config)


register_model("legacy_v1", _create_legacy_v1)
register_model("v4", _create_v4)
