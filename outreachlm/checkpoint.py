from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_VERSION = 3
_SUPPORTED_LOAD_VERSIONS = (None, 1, 2, CHECKPOINT_VERSION)

REQUIRED_CHECKPOINT_FIELDS = (
    "checkpoint_version",
    "model_state",
    "optimizer_state",
    "trainer_state",
    "rng_state",
    "config",
    "metadata",
)


@dataclass
class TrainingCheckpoint:
    checkpoint_version: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    trainer_state: dict[str, Any]
    rng_state: dict[str, Any]
    config: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "model_state": self.model_state,
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "scaler_state": self.scaler_state,
            "trainer_state": self.trainer_state,
            "rng_state": self.rng_state,
            "config": self.config,
            "metadata": self.metadata,
        }


def _current_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(rng_state: dict[str, Any]) -> None:
    torch_state = rng_state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = rng_state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _validate_payload(payload: dict[str, Any]) -> None:
    for field in REQUIRED_CHECKPOINT_FIELDS:
        if field not in payload:
            raise ValueError(f"Checkpoint payload missing required field: {field}")


def _normalize_legacy_payload(payload: dict[str, Any], version: int | None) -> dict[str, Any]:
    train_loss = payload.get(
        "train_loss",
        payload.get(
            "average_train_loss",
            payload.get("last_loss", payload.get("loss", float("nan"))),
        ),
    )
    trainer_state = {
        "step": payload["step"],
        "train_loss": train_loss,
        "best_validation_loss": payload.get("best_validation_loss", float("inf")),
    }
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state": payload["model_state_dict"],
        "optimizer_state": payload["optimizer_state_dict"],
        "scheduler_state": payload.get("scheduler_state_dict"),
        "scaler_state": payload.get("scaler_state_dict"),
        "trainer_state": trainer_state,
        "rng_state": payload.get("rng_state", {}),
        "config": payload.get("config", {}),
        "metadata": {
            "format": "training_checkpoint",
            "source_version": version,
            "normalized_legacy": True,
        },
    }


def build_config(
    *,
    context_length,
    embedding_dim,
    batch_size,
    learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
    validation_split,
    seed,
    corpus_path,
    vocab_size,
    num_layers,
    num_heads,
    eval_loader_config: dict[str, Any] | None = None,
):
    payload = {
        "context_length": context_length,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "min_learning_rate_ratio": min_learning_rate_ratio,
        "validation_split": validation_split,
        "seed": seed,
        "corpus_path": str(Path(corpus_path).resolve()),
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }
    if eval_loader_config is not None:
        payload["eval_loader_config"] = eval_loader_config
    return payload


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    train_loss,
    best_validation_loss,
    config,
    *,
    scheduler_state: dict[str, Any] | None = None,
    scaler_state: dict[str, Any] | None = None,
    trainer_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    runtime: Any | None = None,
):
    path = Path(path)
    is_distributed = bool(runtime is not None and runtime.info.is_distributed)
    is_main_process = bool(runtime is None or runtime.info.is_main_process)
    if is_main_process:
        path.parent.mkdir(parents=True, exist_ok=True)

    resolved_trainer_state = {
        "step": step,
        "train_loss": train_loss,
        "best_validation_loss": best_validation_loss,
    }
    if trainer_state is not None:
        resolved_trainer_state.update(trainer_state)

    resolved_metadata = {
        "format": "training_checkpoint",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata is not None:
        resolved_metadata.update(metadata)

    checkpoint = TrainingCheckpoint(
        checkpoint_version=CHECKPOINT_VERSION,
        model_state=(model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        trainer_state=resolved_trainer_state,
        rng_state=_current_rng_state(),
        config=config,
        metadata=resolved_metadata,
    )
    if is_main_process:
        torch.save(checkpoint.to_dict(), path)
    if is_distributed:
        runtime.barrier()


def load_checkpoint(
    path,
    model,
    optimizer,
    device,
    *,
    scheduler: Any | None = None,
    runtime: Any | None = None,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    is_distributed = bool(runtime is not None and runtime.info.is_distributed)
    if is_distributed:
        runtime.barrier()

    payload = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")

    version = payload.get("checkpoint_version")
    if version not in _SUPPORTED_LOAD_VERSIONS:
        raise RuntimeError(f"Unsupported checkpoint version: {version}")

    if version == CHECKPOINT_VERSION:
        _validate_payload(payload)
        normalized = payload
    else:
        normalized = _normalize_legacy_payload(payload, version)

    try:
        model.load_state_dict(normalized["model_state"])
    except RuntimeError:
        if hasattr(model, "module"):
            model.module.load_state_dict(normalized["model_state"])
        else:
            raise
    optimizer.load_state_dict(normalized["optimizer_state"])
    if scheduler is not None and normalized.get("scheduler_state") is not None:
        scheduler.load_state_dict(normalized["scheduler_state"])
    if runtime is not None:
        runtime.load_scaler_state(normalized.get("scaler_state"))
    _restore_rng_state(normalized["rng_state"])
    if is_distributed:
        runtime.barrier()

    trainer_state = normalized["trainer_state"]
    return {
        "step": trainer_state["step"],
        "train_loss": trainer_state.get("train_loss", float("nan")),
        "best_validation_loss": trainer_state.get("best_validation_loss", float("inf")),
        "config": normalized.get("config", {}),
        "trainer_state": trainer_state,
        "metadata": normalized.get("metadata", {}),
        "checkpoint_version": normalized.get("checkpoint_version"),
        "scheduler_state": normalized.get("scheduler_state"),
        "scaler_state": normalized.get("scaler_state"),
        "rng_state": normalized.get("rng_state", {}),
        "is_legacy": version != CHECKPOINT_VERSION,
    }


def validate_config(
    checkpoint_config,
    current_config,
):
    mismatches = []

    for key, current_value in current_config.items():
        if key not in checkpoint_config:
            mismatches.append(f"{key}: missing from checkpoint")
            continue

        saved_value = checkpoint_config[key]

        if saved_value != current_value:
            mismatches.append(
                f"{key}: "
                f"checkpoint={saved_value!r}, "
                f"current={current_value!r}"
            )

    if mismatches:
        message = (
            "\nCheckpoint configuration "
            "does not match current "
            "training configuration.\n\n"
            + "\n".join(
                f"- {item}"
                for item in mismatches
            )
        )

        raise RuntimeError(message)
