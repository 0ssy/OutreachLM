import pytest
import torch

from outreachlm.model_artifacts import ModelArtifact, load_artifact, save_artifact


def test_model_artifact_round_trip(tmp_path):
    artifact_path = tmp_path / "artifact.pt"
    expected = ModelArtifact(
        artifact_version=1,
        model_type="test_model",
        model_config={"context_length": 32, "embedding_dim": 64},
        tokenizer_config={"tokens": ["<PAD>", "<UNK>", "a"], "pad_token": "<PAD>", "unk_token": "<UNK>"},
        training_config={"seed": 42, "steps": 1000},
        state_dict={"test.weight": torch.tensor([1.0, 2.0, 3.0])},
    )

    save_artifact(expected, artifact_path)
    actual = load_artifact(artifact_path)

    assert actual.artifact_version == expected.artifact_version
    assert actual.model_type == expected.model_type
    assert actual.model_config == expected.model_config
    assert actual.tokenizer_config == expected.tokenizer_config
    assert actual.training_config == expected.training_config
    assert torch.equal(actual.state_dict["test.weight"], expected.state_dict["test.weight"])


def test_model_artifact_missing_required_field_raises(tmp_path):
    artifact_path = tmp_path / "broken_artifact.pt"
    payload = {
        "artifact_version": 1,
        "model_type": "test_model",
        "model_config": {"context_length": 32},
        "tokenizer_config": {"tokens": ["<PAD>"]},
        "state_dict": {"test.weight": torch.tensor([1.0, 2.0, 3.0])},
    }
    torch.save(payload, artifact_path)

    with pytest.raises(ValueError, match="training_config"):
        load_artifact(artifact_path)
