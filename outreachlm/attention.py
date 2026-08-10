import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear projections for Q, K, V
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Final projection after combining all heads
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        # x:
        # [batch_size, sequence_length, embedding_dim]

        batch_size, sequence_length, embedding_dim = x.shape

        # --------------------------------------------------
        # Create Q, K, V
        # --------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Current shape:
        #
        # [B, T, D]
        #
        # where:
        # B = batch size
        # T = sequence length
        # D = embedding dimension

        # --------------------------------------------------
        # Split embedding dimension into attention heads
        # --------------------------------------------------

        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        # Change dimensions to:
        #
        # [B, H, T, Hd]
        #
        # H  = number of heads
        # Hd = dimensions per head

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # --------------------------------------------------
        # Attention scores
        # --------------------------------------------------

        # Q:
        # [B, H, T, Hd]
        #
        # K.transpose:
        # [B, H, Hd, T]
        #
        # Result:
        # [B, H, T, T]

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # --------------------------------------------------
        # Scale attention scores
        # --------------------------------------------------

        scores = scores / math.sqrt(self.head_dim)

        # --------------------------------------------------
        # Causal mask
        # --------------------------------------------------

        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        # Add dimensions so the same mask is applied
        # to every batch and every attention head.
        #
        # [T, T]
        # becomes
        # [1, 1, T, T]

        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

        # --------------------------------------------------
        # Convert scores to probabilities
        # --------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # --------------------------------------------------
        # Weighted combination of values
        # --------------------------------------------------

        attention_output = torch.matmul(
            attention_weights,
            V
        )

        # Shape:
        #
        # [B, H, T, Hd]

        # --------------------------------------------------
        # Combine attention heads
        # --------------------------------------------------

        attention_output = attention_output.transpose(
            1,
            2
        )

        # [B, T, H, Hd]

        attention_output = attention_output.contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # [B, T, D]

        # --------------------------------------------------
        # Final projection
        # --------------------------------------------------

        output = self.output(
            attention_output
        )

        return output, attention_weights