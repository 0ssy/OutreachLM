from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

from outreachlm.evaluation_profiles import EvaluationProfile
from outreachlm.training_config import TrainingConfig


@dataclass(frozen=True)
class RuntimeConfig:
    device: str | None = None
    backend: str = "gloo"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("runtime_config.world_size must be > 0.")
        if self.rank < 0:
            raise ValueError("runtime_config.rank must be >= 0.")
        if self.local_rank < 0:
            raise ValueError("runtime_config.local_rank must be >= 0.")
        if not self.backend:
            raise ValueError("runtime_config.backend must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "backend": self.backend,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeConfig":
        device = payload.get("device")
        if device is not None and not isinstance(device, str):
            raise ValueError("runtime_config.device must be a string or null.")
        return cls(
            device=device,
            backend=payload.get("backend", "gloo"),
            world_size=payload.get("world_size", 1),
            rank=payload.get("rank", 0),
            local_rank=payload.get("local_rank", 0),
        )


@dataclass(frozen=True)
class ExperimentMetadata:
    name: str = "experiment"
    output_dir: str = "experiments"
    git_commit: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metadata.name must not be empty.")
        if not self.output_dir:
            raise ValueError("metadata.output_dir must not be empty.")
        if self.git_commit is not None and not isinstance(self.git_commit, str):
            raise ValueError("metadata.git_commit must be a string or null.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_dir": self.output_dir,
            "git_commit": self.git_commit,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentMetadata":
        return cls(
            name=payload.get("name", "experiment"),
            output_dir=payload.get("output_dir", "experiments"),
            git_commit=payload.get("git_commit"),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    model_type: str = "legacy_v1"
    model_config: dict[str, Any] = field(default_factory=dict)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer_config: dict[str, Any] = field(default_factory=dict)
    evaluation_profile: EvaluationProfile = field(default_factory=EvaluationProfile)
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    paths: dict[str, str] = field(default_factory=dict)
    script_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be >= 0.")
        if not self.model_type:
            raise ValueError("model_type must not be empty.")
        if not isinstance(self.model_config, dict):
            raise ValueError("model_config must be a dictionary.")
        if not isinstance(self.tokenizer_config, dict):
            raise ValueError("tokenizer_config must be a dictionary.")
        if not isinstance(self.paths, dict):
            raise ValueError("paths must be a dictionary.")
        if not isinstance(self.script_args, dict):
            raise ValueError("script_args must be a dictionary.")
        for key, value in self.paths.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("paths must contain string keys and string values.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "model_type": self.model_type,
            "model_config": self.model_config,
            "training_config": self.training_config.to_dict(),
            "tokenizer_config": self.tokenizer_config,
            "evaluation_profile": self.evaluation_profile.to_dict(),
            "runtime_config": self.runtime_config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "paths": self.paths,
            "script_args": self.script_args,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(payload, dict):
            raise ValueError("Experiment config payload must be a dictionary.")
        return cls(
            seed=payload.get("seed", 42),
            model_type=payload.get("model_type", "legacy_v1"),
            model_config=dict(payload.get("model_config", {})),
            training_config=TrainingConfig.from_dict(payload.get("training_config", {})),
            tokenizer_config=dict(payload.get("tokenizer_config", {})),
            evaluation_profile=EvaluationProfile.from_dict(payload.get("evaluation_profile", {})),
            runtime_config=RuntimeConfig.from_dict(payload.get("runtime_config", {})),
            metadata=ExperimentMetadata.from_dict(payload.get("metadata", {})),
            paths=dict(payload.get("paths", {})),
            script_args=dict(payload.get("script_args", {})),
        )


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, ensure_ascii=False, indent=2)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return ExperimentConfig.from_dict(payload)


def to_train_cli_defaults(config: ExperimentConfig) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "seed": config.seed,
        "corpus": Path(config.paths["corpus"]) if "corpus" in config.paths else None,
        "model": Path(config.paths["model"]) if "model" in config.paths else None,
        "tokenizer": Path(config.paths["tokenizer"]) if "tokenizer" in config.paths else None,
        "context_length": config.model_config.get("context_length"),
        "embedding_dim": config.model_config.get("embedding_dim"),
        "num_layers": config.model_config.get("num_layers"),
        "num_heads": config.model_config.get("num_heads"),
        "batch_size": config.training_config.batch_size,
        "learning_rate": config.training_config.learning_rate,
        "steps": config.training_config.steps,
        "warmup_steps": config.training_config.warmup_steps,
        "min_learning_rate_ratio": config.training_config.min_learning_rate_ratio,
        "validation_interval": config.training_config.eval_interval,
    }
    for key, value in config.script_args.items():
        if value is not None:
            defaults[key] = value
    return {key: value for key, value in defaults.items() if value is not None}


def to_train_v4_cli_defaults(config: ExperimentConfig) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "seed": config.seed,
        "corpus": Path(config.paths["corpus"]) if "corpus" in config.paths else None,
        "output_dir": Path(config.metadata.output_dir) if config.metadata.output_dir else None,
        "context_length": config.model_config.get("context_length"),
        "embedding_dim": config.model_config.get("embedding_dim"),
        "num_layers": config.model_config.get("num_layers"),
        "num_heads": config.model_config.get("num_heads"),
        "ffn_dim": config.model_config.get("ffn_dim"),
        "steps": config.training_config.steps,
        "eval_interval": config.training_config.eval_interval,
        "checkpoint_interval": config.training_config.checkpoint_interval,
        "batch_size": config.training_config.batch_size,
        "learning_rate": config.training_config.learning_rate,
        "warmup_steps": config.training_config.warmup_steps,
        "min_learning_rate_ratio": config.training_config.min_learning_rate_ratio,
        "label_smoothing": config.training_config.label_smoothing,
    }
    for key, value in config.script_args.items():
        if value is not None:
            defaults[key] = value
    return {key: value for key, value in defaults.items() if value is not None}
