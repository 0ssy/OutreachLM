import pytest

from outreachlm.training_config import TrainingConfig


def test_training_config_defaults_match_train_v4():
    cfg = TrainingConfig()
    assert cfg.seed == 42
    assert cfg.steps == 4500
    assert cfg.batch_size == 8
    assert cfg.learning_rate == 5e-4
    assert cfg.warmup_steps == 250
    assert cfg.min_learning_rate_ratio == 0.1
    assert cfg.eval_interval == 250
    assert cfg.checkpoint_interval == 500
    assert cfg.label_smoothing == 0.05


@pytest.mark.parametrize(
    "payload",
    [
        {"steps": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"warmup_steps": -1},
        {"eval_interval": 0},
        {"checkpoint_interval": 0},
        {"min_learning_rate_ratio": 0.0},
        {"min_learning_rate_ratio": 1.01},
        {"label_smoothing": -0.01},
        {"label_smoothing": 1.0},
    ],
)
def test_training_config_validation(payload):
    kwargs = TrainingConfig().to_dict()
    kwargs.update(payload)
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_training_config_round_trip_dict():
    original = TrainingConfig(
        seed=0,
        steps=1234,
        batch_size=16,
        learning_rate=1e-3,
        warmup_steps=100,
        min_learning_rate_ratio=0.25,
        eval_interval=100,
        checkpoint_interval=200,
        label_smoothing=0.1,
    )
    restored = TrainingConfig.from_dict(original.to_dict())
    assert restored == original
