import torch
import torch.nn as nn
import torch.nn.functional as F

from outreachlm.model_config import DenseTransformerConfig
from outreachlm.moe import MoELayer, MoEForwardStats


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, self.inv_freq)
        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.repeat_interleave(cos, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.repeat_interleave(sin, 2, dim=-1).unsqueeze(0).unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class HeadRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight.unsqueeze(0).unsqueeze(2)


def _build_norm(kind: str, dim: int) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim)
    if kind == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported normalization: {kind}")


class ScalableSelfAttention(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.kv_heads = config.num_heads if config.kv_heads is None else config.kv_heads
        self.head_dim = (
            config.embedding_dim // config.num_heads
            if config.attention_head_dim is None
            else config.attention_head_dim
        )
        self.attention_dropout = config.attention_dropout
        self.positional_encoding = config.positional_encoding
        self.attention_backend = config.attention_backend
        self.use_fused_qkv = self.kv_heads == self.num_heads

        if self.use_fused_qkv:
            self.qkv = nn.Linear(
                config.embedding_dim,
                config.embedding_dim * 3,
                bias=config.use_bias,
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv = None
            self.q_proj = nn.Linear(
                config.embedding_dim,
                self.num_heads * self.head_dim,
                bias=config.use_bias,
            )
            self.k_proj = nn.Linear(
                config.embedding_dim,
                self.kv_heads * self.head_dim,
                bias=config.use_bias,
            )
            self.v_proj = nn.Linear(
                config.embedding_dim,
                self.kv_heads * self.head_dim,
                bias=config.use_bias,
            )
        self.out = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
            bias=config.use_bias,
        )

        self.rope = (
            RotaryEmbedding(self.head_dim, base=config.rope_base)
            if config.positional_encoding == "rope"
            else None
        )
        self.q_norm = HeadRMSNorm(self.num_heads, self.head_dim)
        self.k_norm = HeadRMSNorm(self.kv_heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        if self.use_fused_qkv:
            assert self.qkv is not None
            qkv = self.qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
        else:
            assert self.q_proj is not None
            assert self.k_proj is not None
            assert self.v_proj is not None
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        q = q.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.rope is not None:
            cos, sin = self.rope.get_cos_sin(sequence_length, x.device, x.dtype)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.kv_heads != self.num_heads:
            group_size = self.num_heads // self.kv_heads
            k = k.repeat_interleave(group_size, dim=1)
            v = v.repeat_interleave(group_size, dim=1)

        if self.attention_backend != "sdpa":
            raise RuntimeError(f"Unsupported attention backend: {self.attention_backend}")
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.embedding_dim)
        return self.out(attn)


class ScalableFFN(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.variant = config.ffn_variant
        self.dropout = nn.Dropout(config.dropout)

        if self.variant == "swiglu":
            self.w1 = nn.Linear(config.embedding_dim, config.ffn_dim, bias=config.use_bias)
            self.w2 = nn.Linear(config.embedding_dim, config.ffn_dim, bias=config.use_bias)
            self.w3 = nn.Linear(config.ffn_dim, config.embedding_dim, bias=config.use_bias)
        elif self.variant == "gated":
            self.w_gate = nn.Linear(config.embedding_dim, config.ffn_dim, bias=config.use_bias)
            self.w_value = nn.Linear(config.embedding_dim, config.ffn_dim, bias=config.use_bias)
            self.w_out = nn.Linear(config.ffn_dim, config.embedding_dim, bias=config.use_bias)
        elif self.variant == "standard":
            self.w_in = nn.Linear(config.embedding_dim, config.ffn_dim, bias=config.use_bias)
            self.w_out = nn.Linear(config.ffn_dim, config.embedding_dim, bias=config.use_bias)
        else:
            raise ValueError(f"Unsupported ffn_variant: {self.variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "swiglu":
            gate = F.silu(self.w1(x))
            value = self.w2(x)
            return self.w3(self.dropout(gate * value))
        if self.variant == "gated":
            gate = torch.sigmoid(self.w_gate(x))
            value = F.gelu(self.w_value(x))
            return self.w_out(self.dropout(gate * value))
        return self.w_out(self.dropout(F.gelu(self.w_in(x))))


class ScalableTransformerBlock(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.config = config
        self.attn_norm = _build_norm(config.normalization, config.embedding_dim)
        self.attn = ScalableSelfAttention(config)
        self.ffn_norm = _build_norm(config.normalization, config.embedding_dim)
        self.ffn = ScalableFFN(config)
        self.moe = MoELayer(config) if config.moe_enabled else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, MoEForwardStats | None]:
        x = x + self.attn(self.attn_norm(x))
        if self.moe is None:
            x = x + self.ffn(self.ffn_norm(x))
            return x, x.new_zeros(()), None

        moe_output, load_balancing_loss, stats = self.moe(self.ffn_norm(x))
        x = x + moe_output
        return x, load_balancing_loss, stats


class ScalableTransformerModel(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.ffn_dim = config.ffn_dim
        self.head_dim = (
            config.embedding_dim // config.num_heads
            if config.attention_head_dim is None
            else config.attention_head_dim
        )

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([ScalableTransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = _build_norm(config.normalization, config.embedding_dim)
        self.output_head = None if config.tie_embeddings else nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=config.use_bias,
        )
        self.last_moe_load_balancing_loss = torch.tensor(0.0)
        self.last_moe_stats: list[MoEForwardStats] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        x = self.token_embedding(input_ids)
        total_moe_loss = x.new_zeros(())
        moe_stats: list[MoEForwardStats] = []
        for block in self.blocks:
            x, moe_loss, stats = block(x)
            total_moe_loss = total_moe_loss + moe_loss
            if stats is not None:
                moe_stats.append(stats)
        x = self.final_norm(x)
        self.last_moe_load_balancing_loss = total_moe_loss
        self.last_moe_stats = moe_stats
        if self.output_head is None:
            return F.linear(x, self.token_embedding.weight)
        return self.output_head(x)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def combine_with_moe_loss(self, language_loss: torch.Tensor) -> torch.Tensor:
        if not self.config.moe_enabled:
            return language_loss
        return language_loss + (self.config.load_balancing_weight * self.last_moe_load_balancing_loss)
