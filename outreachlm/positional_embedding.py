import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        context_length,
        embedding_dim
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, x):

        # x shape:
        # (batch_size, sequence_length, embedding_dim)

        batch_size, sequence_length, embedding_dim = x.shape

        # Position IDs:
        # [0, 1, 2, ..., sequence_length - 1]

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Convert positions into learned vectors

        position_vectors = self.position_embedding(
            positions
        )

        # position_vectors shape:
        # (sequence_length, embedding_dim)

        # Add a batch dimension so broadcasting works

        position_vectors = position_vectors.unsqueeze(0)

        # Shape:
        # (1, sequence_length, embedding_dim)

        return x + position_vectors