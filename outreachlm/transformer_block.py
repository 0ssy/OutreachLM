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

        # --------------------------------------------------
        # SELF-ATTENTION
        # --------------------------------------------------

        normalized_x = self.norm1(x)

        attention_output, attention_weights = self.attention(
            normalized_x
        )

        # --------------------------------------------------
        # RESIDUAL CONNECTION
        # --------------------------------------------------

        x = x + attention_output

        # --------------------------------------------------
        # FEED-FORWARD NETWORK
        # --------------------------------------------------

        feed_forward_output = self.feed_forward(
            self.norm2(x)
        )

        # --------------------------------------------------
        # RESIDUAL CONNECTION
        # --------------------------------------------------

        x = x + feed_forward_output

        # Return both the transformed representation
        # and the attention information.
        return x, attention_weights