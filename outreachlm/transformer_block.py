import torch
import torch.nn as nn

from outreachlm.attention import CausalSelfAttention
from outreachlm.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.attention = CausalSelfAttention(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        # Self-attention
        attention_output, attention_weights = self.attention(
            self.norm1(x)
        )

        # Residual connection
        x = x + attention_output

        # Feed-forward network
        feed_forward_output = self.feed_forward(
            self.norm2(x)
        )

        # Residual connection
        x = x + feed_forward_output

        return x