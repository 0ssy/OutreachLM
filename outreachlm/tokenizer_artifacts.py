from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

from outreachlm.tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class TokenizerArtifact:
    tokens: list[str]
    pad_token: str
    unk_token: str
    artifact_version: int = 1

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("tokens must not be empty.")
        if not self.pad_token:
            raise ValueError("pad_token must not be empty.")
        if not self.unk_token:
            raise ValueError("unk_token must not be empty.")

        if any(not isinstance(token, str) or token == "" for token in self.tokens):
            raise ValueError("every token must be a non-empty string.")

        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokens must be unique.")

        if self.pad_token not in self.tokens:
            raise ValueError("pad_token must be present in tokens.")
        if self.unk_token not in self.tokens:
            raise ValueError("unk_token must be present in tokens.")

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "vocab_size": self.vocab_size,
            "tokens": self.tokens,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TokenizerArtifact":
        required = ("tokens", "pad_token", "unk_token")
        for field in required:
            if field not in payload:
                raise ValueError(f"Tokenizer artifact missing required field: {field}")

        artifact = cls(
            tokens=list(payload["tokens"]),
            pad_token=payload["pad_token"],
            unk_token=payload["unk_token"],
            artifact_version=payload.get("artifact_version", 1),
        )

        serialized_vocab_size = payload.get("vocab_size")
        if serialized_vocab_size is not None and serialized_vocab_size != artifact.vocab_size:
            raise ValueError(
                f"Tokenizer artifact vocab_size mismatch: serialized={serialized_vocab_size}, "
                f"derived={artifact.vocab_size}"
            )
        return artifact

    @classmethod
    def from_tokenizer(cls, tokenizer: CharacterTokenizer) -> "TokenizerArtifact":
        return cls(
            tokens=list(tokenizer.tokens),
            pad_token=tokenizer.pad_token,
            unk_token=tokenizer.unk_token,
        )

    def to_tokenizer(self) -> CharacterTokenizer:
        tokenizer = CharacterTokenizer.__new__(CharacterTokenizer)
        tokenizer.pad_token = self.pad_token
        tokenizer.unk_token = self.unk_token
        tokenizer.tokens = list(self.tokens)
        tokenizer.token_to_id = {
            token: index for index, token in enumerate(tokenizer.tokens)
        }
        tokenizer.id_to_token = {
            index: token for token, index in tokenizer.token_to_id.items()
        }
        return tokenizer


def save_tokenizer_artifact(artifact: TokenizerArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(artifact.to_dict(), file, ensure_ascii=False, indent=2)


def load_tokenizer_artifact(path: str | Path) -> TokenizerArtifact:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Tokenizer artifact payload must be a dictionary.")
    return TokenizerArtifact.from_dict(payload)
