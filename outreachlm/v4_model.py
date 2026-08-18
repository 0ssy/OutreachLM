import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        inv_freq = 1.0 / (
            base ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32
                ) / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len, device, dtype):
        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype
        )
        angles = torch.einsum(
            "i,j->ij",
            positions,
            self.inv_freq
        )
        cos = torch.cos(angles).to(dtype=dtype)
        sin = torch.sin(angles).to(dtype=dtype)
        return cos, sin


def rotate_half(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack(
        (-x_odd, x_even),
        dim=-1
    ).flatten(-2)


def apply_rope(x, cos, sin):
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class HeadRMSNorm(nn.Module):
    def __init__(self, num_heads, head_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(num_heads, head_dim)
        )

    def forward(self, x):
        # x: [B, H, T, D]
        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return x * rms * self.weight.unsqueeze(0).unsqueeze(2)


class V4SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=True
        )
        self.out = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True
        )
        self.rope = RotaryEmbedding(self.head_dim)
        self.q_norm = HeadRMSNorm(
            num_heads=num_heads,
            head_dim=self.head_dim
        )
        self.k_norm = HeadRMSNorm(
            num_heads=num_heads,
            head_dim=self.head_dim
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(
            seq_len=sequence_length,
            device=x.device,
            dtype=x.dtype
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        attn = attn.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )
        return self.out(attn)


class V4SwiGLU(nn.Module):
    def __init__(self, embedding_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(
            embedding_dim,
            ffn_dim,
            bias=True
        )
        self.w2 = nn.Linear(
            embedding_dim,
            ffn_dim,
            bias=True
        )
        self.w3 = nn.Linear(
            ffn_dim,
            embedding_dim,
            bias=True
        )

    def forward(self, x):
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.w3(gate * value)


class V4Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim):
        super().__init__()
        self.attn_norm = RMSNorm(embedding_dim)
        self.attn = V4SelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )
        self.ffn_norm = RMSNorm(embedding_dim)
        self.ffn = V4SwiGLU(
            embedding_dim=embedding_dim,
            ffn_dim=ffn_dim
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class OutreachV4Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length=256,
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_dim=684,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.ffn_dim = ffn_dim

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.blocks = nn.ModuleList(
            [
                V4Block(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = RMSNorm(embedding_dim)

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # Tied LM head.
        logits = F.linear(
            x,
            self.token_embedding.weight
        )
        return logits

    @property
    def parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )
