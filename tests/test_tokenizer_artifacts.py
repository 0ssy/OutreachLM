import pytest

from outreachlm.tokenizer import CharacterTokenizer
from outreachlm.tokenizer_artifacts import (
    TokenizerArtifact,
    load_tokenizer_artifact,
    save_tokenizer_artifact,
)


def _sample_tokens():
    return ["<PAD>", "<UNK>", "a", "b", "c"]


def test_tokenizer_artifact_construction():
    artifact = TokenizerArtifact(
        tokens=_sample_tokens(),
        pad_token="<PAD>",
        unk_token="<UNK>",
    )
    assert artifact.pad_token == "<PAD>"
    assert artifact.unk_token == "<UNK>"


def test_vocab_size_is_derived_correctly():
    artifact = TokenizerArtifact(
        tokens=_sample_tokens(),
        pad_token="<PAD>",
        unk_token="<UNK>",
    )
    assert artifact.vocab_size == 5


def test_special_token_configuration():
    artifact = TokenizerArtifact(
        tokens=_sample_tokens(),
        pad_token="<PAD>",
        unk_token="<UNK>",
    )
    assert artifact.pad_token in artifact.tokens
    assert artifact.unk_token in artifact.tokens


def test_save_load_round_trip(tmp_path):
    path = tmp_path / "tokenizer.json"
    expected = TokenizerArtifact(
        tokens=_sample_tokens(),
        pad_token="<PAD>",
        unk_token="<UNK>",
    )
    save_tokenizer_artifact(expected, path)
    actual = load_tokenizer_artifact(path)
    assert actual == expected


def test_missing_required_fields_raise():
    with pytest.raises(ValueError, match="missing required field: unk_token"):
        TokenizerArtifact.from_dict(
            {
                "tokens": _sample_tokens(),
                "pad_token": "<PAD>",
            }
        )


@pytest.mark.parametrize(
    "tokens",
    [
        [],
        ["", "<UNK>"],
    ],
)
def test_invalid_vocabulary_raises(tokens):
    with pytest.raises(ValueError):
        TokenizerArtifact(
            tokens=tokens,
            pad_token="<PAD>",
            unk_token="<UNK>",
        )


def test_duplicate_tokens_raise():
    with pytest.raises(ValueError, match="tokens must be unique"):
        TokenizerArtifact(
            tokens=["<PAD>", "<UNK>", "a", "a"],
            pad_token="<PAD>",
            unk_token="<UNK>",
        )


def test_serialization_deserialization_round_trip():
    original = TokenizerArtifact(
        tokens=_sample_tokens(),
        pad_token="<PAD>",
        unk_token="<UNK>",
    )
    restored = TokenizerArtifact.from_dict(original.to_dict())
    assert restored == original


def test_compatibility_with_current_tokenizer_representation():
    payload = {
        "vocab_size": 5,
        "tokens": _sample_tokens(),
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
    }
    artifact = TokenizerArtifact.from_dict(payload)
    tokenizer = artifact.to_tokenizer()
    assert isinstance(tokenizer, CharacterTokenizer)
    assert tokenizer.vocab_size == 5
    assert tokenizer.pad_token == "<PAD>"
    assert tokenizer.unk_token == "<UNK>"


def test_vocab_size_mismatch_raises():
    with pytest.raises(ValueError, match="vocab_size mismatch"):
        TokenizerArtifact.from_dict(
            {
                "vocab_size": 4,
                "tokens": _sample_tokens(),
                "pad_token": "<PAD>",
                "unk_token": "<UNK>",
            }
        )
