from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REQUIRED_FIELDS = (
    "artifact_version",
    "model_type",
    "model_config",
    "tokenizer_config",
    "training_config",
    "state_dict",
)


@dataclass
class ModelArtifact:
    artifact_version: int
    model_type: str
    model_config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    training_config: dict[str, Any]
    state_dict: dict[str, Any]


def _validate_payload(payload: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in payload:
            raise ValueError(f"Artifact payload missing required field: {field}")


def save_artifact(artifact: ModelArtifact, path: str | Path) -> None:
    path = Path(path)
    payload = {
        "artifact_version": artifact.artifact_version,
        "model_type": artifact.model_type,
        "model_config": artifact.model_config,
        "tokenizer_config": artifact.tokenizer_config,
        "training_config": artifact.training_config,
        "state_dict": artifact.state_dict,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_artifact(path: str | Path) -> ModelArtifact:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Artifact payload must be a dictionary.")

    _validate_payload(payload)

    return ModelArtifact(
        artifact_version=payload["artifact_version"],
        model_type=payload["model_type"],
        model_config=payload["model_config"],
        tokenizer_config=payload["tokenizer_config"],
        training_config=payload["training_config"],
        state_dict=payload["state_dict"],
    )
