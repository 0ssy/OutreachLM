from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    steps: int = 4500
    batch_size: int = 8
    learning_rate: float = 5e-4
    warmup_steps: int = 250
    min_learning_rate_ratio: float = 0.1
    eval_interval: int = 250
    checkpoint_interval: int = 500
    label_smoothing: float = 0.05

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0.")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be > 0.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0.")
        if self.min_learning_rate_ratio <= 0:
            raise ValueError("min_learning_rate_ratio must be > 0.")
        if self.min_learning_rate_ratio > 1:
            raise ValueError("min_learning_rate_ratio must be <= 1.")
        if self.label_smoothing < 0:
            raise ValueError("label_smoothing must be >= 0.")
        if self.label_smoothing >= 1:
            raise ValueError("label_smoothing must be < 1.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "min_learning_rate_ratio": self.min_learning_rate_ratio,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "label_smoothing": self.label_smoothing,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        return cls(
            seed=payload.get("seed", 42),
            steps=payload.get("steps", 4500),
            batch_size=payload.get("batch_size", 8),
            learning_rate=payload.get("learning_rate", 5e-4),
            warmup_steps=payload.get("warmup_steps", 250),
            min_learning_rate_ratio=payload.get("min_learning_rate_ratio", 0.1),
            eval_interval=payload.get("eval_interval", 250),
            checkpoint_interval=payload.get("checkpoint_interval", 500),
            label_smoothing=payload.get("label_smoothing", 0.05),
        )
