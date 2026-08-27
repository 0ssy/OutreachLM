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


@dataclass(frozen=True)
class DenseTransformerConfig:
    vocab_size: int = 490
    context_length: int = 256
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    kv_heads: int | None = None
    attention_head_dim: int | None = None
    ffn_dim: int = 684
    attention_backend: str = "sdpa"
    rope_base: float = 10000.0
    normalization: str = "rmsnorm"
    positional_encoding: str = "rope"
    ffn_variant: str = "swiglu"
    attention_dropout: float = 0.0
    dropout: float = 0.0
    use_bias: bool = True
    tie_embeddings: bool = True
    moe_enabled: bool = False
    num_experts: int = 4
    top_k: int = 2
    expert_ffn_dim: int | None = None
    capacity_factor: float = 1.25
    router_bias: bool = True
    load_balancing_weight: float = 0.0
    moe_fallback: str = "drop"

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
        resolved_kv_heads = self.num_heads if self.kv_heads is None else self.kv_heads
        if resolved_kv_heads <= 0:
            raise ValueError("kv_heads must be > 0 when provided.")
        if self.num_heads % resolved_kv_heads != 0:
            raise ValueError("num_heads must be divisible by kv_heads.")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be > 0.")
        if self.attention_head_dim is None:
            if self.embedding_dim % self.num_heads != 0:
                raise ValueError("embedding_dim must be divisible by num_heads.")
        else:
            if self.attention_head_dim <= 0:
                raise ValueError("attention_head_dim must be > 0 when provided.")
            if self.embedding_dim != self.num_heads * self.attention_head_dim:
                raise ValueError(
                    "embedding_dim must equal num_heads * attention_head_dim when attention_head_dim is set."
                )
        if self.attention_backend not in {"sdpa"}:
            raise ValueError("attention_backend must be one of: sdpa.")
        if self.rope_base <= 0.0:
            raise ValueError("rope_base must be > 0.")
        if self.normalization not in {"rmsnorm", "layernorm"}:
            raise ValueError("normalization must be one of: rmsnorm, layernorm.")
        if self.positional_encoding not in {"rope", "none"}:
            raise ValueError("positional_encoding must be one of: rope, none.")
        if self.ffn_variant not in {"swiglu", "standard", "gated"}:
            raise ValueError("ffn_variant must be one of: swiglu, standard, gated.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be > 0.")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must be <= num_experts.")
        if self.expert_ffn_dim is not None and self.expert_ffn_dim <= 0:
            raise ValueError("expert_ffn_dim must be > 0 when provided.")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0.")
        if self.load_balancing_weight < 0:
            raise ValueError("load_balancing_weight must be >= 0.")
        if self.moe_fallback not in {"drop", "dense"}:
            raise ValueError("moe_fallback must be one of: drop, dense.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "kv_heads": self.kv_heads,
            "attention_head_dim": self.attention_head_dim,
            "ffn_dim": self.ffn_dim,
            "attention_backend": self.attention_backend,
            "rope_base": self.rope_base,
            "normalization": self.normalization,
            "positional_encoding": self.positional_encoding,
            "ffn_variant": self.ffn_variant,
            "attention_dropout": self.attention_dropout,
            "dropout": self.dropout,
            "use_bias": self.use_bias,
            "tie_embeddings": self.tie_embeddings,
            "moe_enabled": self.moe_enabled,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_ffn_dim": self.expert_ffn_dim,
            "capacity_factor": self.capacity_factor,
            "router_bias": self.router_bias,
            "load_balancing_weight": self.load_balancing_weight,
            "moe_fallback": self.moe_fallback,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DenseTransformerConfig":
        return cls(
            vocab_size=payload["vocab_size"],
            context_length=payload.get("context_length", 256),
            embedding_dim=payload.get("embedding_dim", 256),
            num_layers=payload.get("num_layers", 4),
            num_heads=payload.get("num_heads", 8),
            kv_heads=payload.get("kv_heads"),
            attention_head_dim=payload.get("attention_head_dim"),
            ffn_dim=payload.get("ffn_dim", 684),
            attention_backend=payload.get("attention_backend", "sdpa"),
            rope_base=payload.get("rope_base", 10000.0),
            normalization=payload.get("normalization", "rmsnorm"),
            positional_encoding=payload.get("positional_encoding", "rope"),
            ffn_variant=payload.get("ffn_variant", "swiglu"),
            attention_dropout=payload.get("attention_dropout", 0.0),
            dropout=payload.get("dropout", 0.0),
            use_bias=payload.get("use_bias", True),
            tie_embeddings=payload.get("tie_embeddings", True),
            moe_enabled=payload.get("moe_enabled", False),
            num_experts=payload.get("num_experts", 4),
            top_k=payload.get("top_k", 2),
            expert_ffn_dim=payload.get("expert_ffn_dim"),
            capacity_factor=payload.get("capacity_factor", 1.25),
            router_bias=payload.get("router_bias", True),
            load_balancing_weight=payload.get("load_balancing_weight", 0.0),
            moe_fallback=payload.get("moe_fallback", "drop"),
        )
