from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegacyV1Config:
    vocab_size: int = 490
    context_length: int = 32
    embedding_dim: int = 64
    num_layers: int = 1
    num_heads: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LegacyV1Config":
        return cls(
            vocab_size=payload["vocab_size"],
            context_length=payload["context_length"],
            embedding_dim=payload["embedding_dim"],
            num_layers=payload.get("num_layers", 1),
            num_heads=payload.get("num_heads", 4),
        )


@dataclass(frozen=True)
class V4Config:
    vocab_size: int = 490
    context_length: int = 256
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 684

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be > 0.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be > 0.")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "V4Config":
        return cls(
            vocab_size=payload["vocab_size"],
            context_length=payload.get("context_length", 256),
            embedding_dim=payload.get("embedding_dim", 256),
            num_layers=payload.get("num_layers", 4),
            num_heads=payload.get("num_heads", 8),
            ffn_dim=payload.get("ffn_dim", 684),
        )
