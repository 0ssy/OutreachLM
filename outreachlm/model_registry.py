from collections.abc import Callable
from typing import Any

import torch.nn as nn

from outreachlm.model import OutreachModel
from outreachlm.v4_model import OutreachV4Model


ModelFactory = Callable[[dict[str, Any]], nn.Module]


def _build_legacy_v1(model_config: dict[str, Any]) -> nn.Module:
    return OutreachModel(
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config.get("num_layers", 1),
        num_heads=model_config.get("num_heads", 4),
    )


def _build_v4(model_config: dict[str, Any]) -> nn.Module:
    return OutreachV4Model(
        vocab_size=model_config["vocab_size"],
        context_length=model_config.get("context_length", 256),
        embedding_dim=model_config.get("embedding_dim", 256),
        num_layers=model_config.get("num_layers", 4),
        num_heads=model_config.get("num_heads", 8),
        ffn_dim=model_config.get("ffn_dim", 684),
    )


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "legacy_v1": _build_legacy_v1,
    "v4": _build_v4,
}


def create_model(model_type: str, model_config: dict[str, Any]) -> nn.Module:
    try:
        factory = MODEL_REGISTRY[model_type]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_type={model_type!r}. Available model types: {available}"
        ) from exc
    return factory(model_config)


def available_model_types() -> tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))
