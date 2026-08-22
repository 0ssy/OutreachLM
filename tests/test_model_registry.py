import torch.nn as nn
import pytest

from outreachlm.model import OutreachModel
from outreachlm.model_registry import available_model_types, create_model
from outreachlm.v4_model import OutreachV4Model


def test_available_model_types():
    assert available_model_types() == ("legacy_v1", "v4")


def test_create_legacy_v1():
    config = {
        "vocab_size": 490,
        "context_length": 64,
        "embedding_dim": 64,
        "num_layers": 1,
        "num_heads": 4,
    }
    model = create_model("legacy_v1", config)
    assert isinstance(model, OutreachModel)
    assert isinstance(model, nn.Module)
    assert model.vocab_size == 490
    assert model.context_length == 64
    assert model.embedding_dim == 64


def test_create_v4():
    config = {
        "vocab_size": 490,
        "context_length": 256,
        "embedding_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_dim": 684,
    }
    model = create_model("v4", config)
    assert isinstance(model, OutreachV4Model)
    assert isinstance(model, nn.Module)
    assert model.vocab_size == 490
    assert model.context_length == 256
    assert model.embedding_dim == 256
    assert model.num_layers == 4
    assert model.num_heads == 8
    assert model.ffn_dim == 684


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unknown model_type"):
        create_model("does_not_exist", {})


def test_registry_v4_constructor_matches_direct_constructor_structure():
    config = {
        "vocab_size": 490,
        "context_length": 256,
        "embedding_dim": 256,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_dim": 684,
    }
    direct = OutreachV4Model(**config)
    via_registry = create_model("v4", config)

    direct_state = direct.state_dict()
    registry_state = via_registry.state_dict()
    assert list(direct_state.keys()) == list(registry_state.keys())
    for name in direct_state:
        assert direct_state[name].shape == registry_state[name].shape
    assert sum(p.numel() for p in direct.parameters()) == sum(
        p.numel() for p in via_registry.parameters()
    )


def test_registry_legacy_constructor_matches_direct_constructor_structure():
    config = {
        "vocab_size": 490,
        "context_length": 64,
        "embedding_dim": 64,
        "num_layers": 1,
        "num_heads": 4,
    }
    direct = OutreachModel(**config)
    via_registry = create_model("legacy_v1", config)

    direct_state = direct.state_dict()
    registry_state = via_registry.state_dict()
    assert list(direct_state.keys()) == list(registry_state.keys())
    for name in direct_state:
        assert direct_state[name].shape == registry_state[name].shape
    assert sum(p.numel() for p in direct.parameters()) == sum(
        p.numel() for p in via_registry.parameters()
    )
