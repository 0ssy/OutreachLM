import torch
import torch.nn as nn

from outreachlm.transformer_block import TransformerBlock


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):

        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):

    def __init__(self, context_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, positions):

        return self.embedding(positions)


class OutreachModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim
    ):
        super().__init__()

        self.context_length = context_length

        # Token identity
        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        # Token position
        self.position_embedding = PositionalEmbedding(
            context_length,
            embedding_dim
        )

        # Transformer processing
        self.transformer = TransformerBlock(
            embedding_dim
        )

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        # -----------------------------
        # Token embeddings
        # -----------------------------

        token_vectors = self.token_embedding(
            input_ids
        )

        # -----------------------------
        # Position indices
        # -----------------------------

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # -----------------------------
        # Positional embeddings
        # -----------------------------

        position_vectors = self.position_embedding(
            positions.unsqueeze(0)
        )

        # Add position to token representation
        x = token_vectors + position_vectors

        # -----------------------------
        # Transformer processing
        # -----------------------------

        x = self.transformer(x)

        logits = self.output_head(x)

        return logits