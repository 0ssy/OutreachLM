from pathlib import Path

import pytest

from outreachlm.experiment_config import (
    ExperimentConfig,
    ExperimentMetadata,
    RuntimeConfig,
    load_experiment_config,
    save_experiment_config,
    to_train_cli_defaults,
    to_train_v4_cli_defaults,
)
from outreachlm.training_config import TrainingConfig


def test_experiment_config_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(
        seed=1337,
        model_type="v4",
        model_config={
            "context_length": 128,
            "embedding_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "ffn_dim": 512,
        },
        training_config=TrainingConfig(
            seed=1337,
            steps=2000,
            batch_size=16,
            learning_rate=1e-3,
            warmup_steps=100,
            min_learning_rate_ratio=0.2,
            eval_interval=100,
            checkpoint_interval=200,
            label_smoothing=0.1,
        ),
        tokenizer_config={"pad_token": "<PAD>", "unk_token": "<UNK>", "tokens": ["a", "b"]},
        runtime_config=RuntimeConfig(device="cpu"),
        metadata=ExperimentMetadata(
            name="b2.1-roundtrip",
            output_dir="experiments/b21",
            git_commit="abc123",
        ),
        paths={
            "corpus": "corpus/fineweb",
            "model": "experiments/model.pt",
            "tokenizer": "experiments/tokenizer.json",
        },
        script_args={"log_interval": 25, "recovery_start_index": 40},
    )
    path = tmp_path / "experiment.json"
    save_experiment_config(config, path)

    loaded = load_experiment_config(path)

    assert loaded == config
    assert loaded.to_dict() == config.to_dict()


def test_experiment_config_rejects_invalid_seed() -> None:
    with pytest.raises(ValueError, match="seed must be >= 0"):
        ExperimentConfig(seed=-1)


def test_runtime_config_rejects_non_string_device() -> None:
    with pytest.raises(ValueError, match="runtime_config.device must be a string or null"):
        RuntimeConfig.from_dict({"device": 123})


def test_train_defaults_mapping_uses_typed_sections() -> None:
    config = ExperimentConfig(
        seed=7,
        model_config={
            "context_length": 64,
            "embedding_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
        },
        training_config=TrainingConfig(
            seed=7,
            steps=99,
            batch_size=3,
            learning_rate=0.01,
            warmup_steps=5,
            min_learning_rate_ratio=0.3,
            eval_interval=11,
            checkpoint_interval=22,
            label_smoothing=0.2,
        ),
        paths={
            "corpus": "corpus/x",
            "model": "m.pt",
            "tokenizer": "t.json",
        },
        script_args={
            "validation_split": 0.2,
            "log_interval": 9,
            "eval_num_workers": 2,
            "eval_prefetch_factor": 4,
            "eval_persistent_workers": True,
        },
    )

    defaults = to_train_cli_defaults(config)
    assert defaults["seed"] == 7
    assert defaults["context_length"] == 64
    assert defaults["steps"] == 99
    assert defaults["validation_interval"] == 11
    assert defaults["validation_split"] == 0.2
    assert defaults["log_interval"] == 9
    assert defaults["eval_num_workers"] == 2
    assert defaults["eval_prefetch_factor"] == 4
    assert defaults["eval_persistent_workers"] is True
    assert defaults["corpus"] == Path("corpus/x")


def test_train_v4_defaults_mapping_uses_typed_sections() -> None:
    config = ExperimentConfig(
        seed=9,
        model_config={
            "context_length": 256,
            "embedding_dim": 128,
            "num_layers": 3,
            "num_heads": 4,
            "ffn_dim": 400,
        },
        training_config=TrainingConfig(
            seed=9,
            steps=500,
            batch_size=6,
            learning_rate=5e-4,
            warmup_steps=10,
            min_learning_rate_ratio=0.1,
            eval_interval=50,
            checkpoint_interval=60,
            label_smoothing=0.05,
        ),
        metadata=ExperimentMetadata(name="v4", output_dir="experiments/v4"),
        paths={"corpus": "corpus/y"},
        script_args={"recovery_start_index": 39, "log_interval": 13},
    )

    defaults = to_train_v4_cli_defaults(config)
    assert defaults["seed"] == 9
    assert defaults["context_length"] == 256
    assert defaults["steps"] == 500
    assert defaults["eval_interval"] == 50
    assert defaults["checkpoint_interval"] == 60
    assert defaults["output_dir"] == Path("experiments/v4")
    assert defaults["recovery_start_index"] == 39
    assert defaults["log_interval"] == 13
