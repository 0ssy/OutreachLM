import json
from pathlib import Path

from outreachlm.train import parse_args as parse_train_args
from outreachlm.train_v4 import parse_args as parse_train_v4_args


def test_train_cli_uses_experiment_config_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "train-config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 123,
                "model_config": {
                    "context_length": 64,
                    "embedding_dim": 48,
                    "num_layers": 2,
                    "num_heads": 4,
                },
                "training_config": {
                    "steps": 321,
                    "batch_size": 7,
                    "learning_rate": 0.002,
                    "warmup_steps": 8,
                    "min_learning_rate_ratio": 0.2,
                    "eval_interval": 17,
                },
                "paths": {
                    "corpus": "corpus/a",
                    "model": "out/model.pt",
                    "tokenizer": "out/tokenizer.json",
                },
                "script_args": {
                    "validation_split": 0.2,
                    "log_interval": 6,
                    "eval_num_workers": 1,
                    "eval_prefetch_factor": 3,
                    "eval_pin_memory": True,
                },
            }
        ),
        encoding="utf-8",
    )

    args = parse_train_args(["--config", str(config_path)])

    assert args.seed == 123
    assert args.context_length == 64
    assert args.embedding_dim == 48
    assert args.num_layers == 2
    assert args.num_heads == 4
    assert args.steps == 321
    assert args.batch_size == 7
    assert args.learning_rate == 0.002
    assert args.warmup_steps == 8
    assert args.min_learning_rate_ratio == 0.2
    assert args.validation_interval == 17
    assert args.validation_split == 0.2
    assert args.log_interval == 6
    assert args.eval_num_workers == 1
    assert args.eval_prefetch_factor == 3
    assert args.eval_pin_memory is True
    assert args.corpus == Path("corpus/a")
    assert args.model == Path("out/model.pt")
    assert args.tokenizer == Path("out/tokenizer.json")


def test_train_cli_explicit_flags_override_config(tmp_path: Path) -> None:
    config_path = tmp_path / "train-config.json"
    config_path.write_text(
        json.dumps(
            {
                "training_config": {"steps": 321},
                "script_args": {"log_interval": 6},
            }
        ),
        encoding="utf-8",
    )

    args = parse_train_args(
        [
            "--config",
            str(config_path),
            "--steps",
            "5",
            "--log-interval",
            "2",
        ]
    )

    assert args.steps == 5
    assert args.log_interval == 2


def test_train_v4_cli_uses_experiment_config_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "train-v4-config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 99,
                "model_config": {
                    "context_length": 128,
                    "embedding_dim": 96,
                    "num_layers": 3,
                    "num_heads": 8,
                    "ffn_dim": 320,
                },
                "training_config": {
                    "steps": 100,
                    "eval_interval": 10,
                    "checkpoint_interval": 20,
                    "batch_size": 4,
                    "learning_rate": 0.001,
                    "warmup_steps": 4,
                    "min_learning_rate_ratio": 0.3,
                    "label_smoothing": 0.1,
                },
                "metadata": {"output_dir": "experiments/v4-from-config"},
                "paths": {"corpus": "corpus/b"},
                "script_args": {
                    "recovery_start_index": 41,
                    "rollout_calibration_start_index": 42,
                    "log_interval": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    args = parse_train_v4_args(["--config", str(config_path)])

    assert args.seed == 99
    assert args.context_length == 128
    assert args.embedding_dim == 96
    assert args.num_layers == 3
    assert args.num_heads == 8
    assert args.ffn_dim == 320
    assert args.steps == 100
    assert args.eval_interval == 10
    assert args.checkpoint_interval == 20
    assert args.batch_size == 4
    assert args.learning_rate == 0.001
    assert args.warmup_steps == 4
    assert args.min_learning_rate_ratio == 0.3
    assert args.label_smoothing == 0.1
    assert args.output_dir == Path("experiments/v4-from-config")
    assert args.corpus == Path("corpus/b")
    assert args.recovery_start_index == 41
    assert args.rollout_calibration_start_index == 42
    assert args.log_interval == 3


def test_train_v4_cli_explicit_flags_override_config(tmp_path: Path) -> None:
    config_path = tmp_path / "train-v4-config.json"
    config_path.write_text(
        json.dumps(
            {
                "training_config": {"steps": 100},
                "script_args": {"log_interval": 3},
            }
        ),
        encoding="utf-8",
    )

    args = parse_train_v4_args(
        [
            "--config",
            str(config_path),
            "--steps",
            "9",
            "--log-interval",
            "1",
        ]
    )

    assert args.steps == 9
    assert args.log_interval == 1
