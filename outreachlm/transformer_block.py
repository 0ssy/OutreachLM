import torch
import torch.nn as nn

from outreachlm.attention import CausalSelfAttention
from outreachlm.feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads=4):
        super().__init__()

        self.attention = CausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
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

    def forward(
        self,
        x,
        return_attention=False
    ):

        # ==================================================
        # SELF ATTENTION
        # ==================================================

        normalized_x = self.norm1(x)

        attention_output, attention_weights = self.attention(
            normalized_x
        )

        # Residual connection

        x = x + attention_output

        # ==================================================
        # FEED FORWARD
        # ==================================================

        normalized_x = self.norm2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Residual connection

        x = x + feed_forward_output

        # ==================================================
        # OUTPUT
        # ==================================================

        if return_attention:
            return x, attention_weights

        return x