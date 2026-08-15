from pathlib import Path

import torch

CHECKPOINT_VERSION = 2


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
):
    return {
        "context_length": context_length,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "min_learning_rate_ratio": min_learning_rate_ratio,
        "validation_split": validation_split,
        "seed": seed,
        "corpus_path": str(
            Path(corpus_path).resolve()
        ),
        "vocab_size": vocab_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    train_loss,
    best_validation_loss,
    config,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,

        "step": step,

        "train_loss": train_loss,

        "best_validation_loss": (
            best_validation_loss
        ),

        "model_state_dict": (
            model.state_dict()
        ),

        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        "config": config,
    }

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer,
    device,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    version = checkpoint.get(
        "checkpoint_version"
    )

    if version not in (None, 1, CHECKPOINT_VERSION):
        raise RuntimeError(
            "Unsupported checkpoint version: "
            f"{version}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    if "train_loss" in checkpoint:
        train_loss = checkpoint["train_loss"]
    elif "average_train_loss" in checkpoint:
        train_loss = checkpoint["average_train_loss"]
    elif "last_loss" in checkpoint:
        train_loss = checkpoint["last_loss"]
    elif "loss" in checkpoint:
        train_loss = checkpoint["loss"]
    else:
        train_loss = float("nan")

    return {
        "step": checkpoint["step"],

        "train_loss": train_loss,

        "best_validation_loss": checkpoint.get(
            "best_validation_loss",
            float("inf")
        ),

        "config": checkpoint.get("config", {}),
        "is_legacy": version != CHECKPOINT_VERSION,
    }


def validate_config(
    checkpoint_config,
    current_config,
):
    mismatches = []

    for key, current_value in (
        current_config.items()
    ):
        if key not in checkpoint_config:
            mismatches.append(
                f"{key}: missing from checkpoint"
            )
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
