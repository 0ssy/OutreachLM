import pytest

from outreachlm.model_config import LegacyV1Config, V4Config
from outreachlm.model_registry import create_model


def test_legacy_v1_config_defaults_match_current_behavior():
    cfg = LegacyV1Config()
    assert cfg.vocab_size == 490
    assert cfg.context_length == 32
    assert cfg.embedding_dim == 64
    assert cfg.num_layers == 1
    assert cfg.num_heads == 4


def test_v4_config_defaults_match_current_behavior():
    cfg = V4Config()
    assert cfg.vocab_size == 490
    assert cfg.context_length == 256
    assert cfg.embedding_dim == 256
    assert cfg.num_layers == 4
    assert cfg.num_heads == 8
    assert cfg.ffn_dim == 684


@pytest.mark.parametrize(
    "payload",
    [
        {"vocab_size": 0},
        {"context_length": 0},
        {"embedding_dim": 0},
        {"num_layers": 0},
        {"num_heads": 0},
    ],
)
def test_legacy_v1_config_validation(payload):
    kwargs = {
        "vocab_size": 490,
        "context_length": 32,
        "embedding_dim": 64,
        "num_layers": 1,
        "num_heads": 4,
    }
    kwargs.update(payload)
    with pytest.raises(ValueError):
        LegacyV1Config(**kwargs)


@pytest.mark.parametrize(
    "payload",
    [
        {"vocab_size": 0},
        {"context_length": 0},
        {"embedding_dim": 0},
        {"num_layers": 0},
        {"num_heads": 0},
        {"ffn_dim": 0},
        {"embedding_dim": 255, "num_heads": 8},
    ],
)
def test_v4_config_validation(payload):
    kwargs = {
        "vocab_size": 490,
        "context_length": 256,
        "embedding_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_dim": 684,
    }
    kwargs.update(payload)
    with pytest.raises(ValueError):
        V4Config(**kwargs)


def test_legacy_v1_config_round_trip_dict():
    original = LegacyV1Config(
        vocab_size=490,
        context_length=64,
        embedding_dim=96,
        num_layers=2,
        num_heads=4,
    )
    restored = LegacyV1Config.from_dict(original.to_dict())
    assert restored == original


def test_v4_config_round_trip_dict():
    original = V4Config(
        vocab_size=490,
        context_length=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=704,
    )
    restored = V4Config.from_dict(original.to_dict())
    assert restored == original


def test_registry_accepts_typed_v4_config_and_matches_dict_path():
    cfg = V4Config(
        vocab_size=490,
        context_length=256,
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_dim=684,
    )
    from_typed = create_model(cfg)
    from_dict = create_model("v4", cfg.to_dict())

    assert list(from_typed.state_dict().keys()) == list(from_dict.state_dict().keys())
    assert sum(p.numel() for p in from_typed.parameters()) == sum(
        p.numel() for p in from_dict.parameters()
    )


def test_registry_accepts_typed_legacy_config_and_matches_dict_path():
    cfg = LegacyV1Config(
        vocab_size=490,
        context_length=64,
        embedding_dim=64,
        num_layers=1,
        num_heads=4,
    )
    from_typed = create_model(cfg)
    from_dict = create_model("legacy_v1", cfg.to_dict())

    assert list(from_typed.state_dict().keys()) == list(from_dict.state_dict().keys())
    assert sum(p.numel() for p in from_typed.parameters()) == sum(
        p.numel() for p in from_dict.parameters()
    )
