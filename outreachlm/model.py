import torch
import torch.nn as nn

from outreachlm.transformer_block import TransformerBlock


class TokenEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

    def forward(self, token_ids):

        return self.embedding(
            token_ids
        )


class PositionalEmbedding(nn.Module):

    def __init__(
        self,
        context_length,
        embedding_dim
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            context_length,
            embedding_dim
        )

    def forward(self, positions):

        return self.embedding(
            positions
        )


class OutreachModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_layers=1,
        num_heads=4
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # ==================================================
        # TOKEN EMBEDDING
        # ==================================================

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        # ==================================================
        # POSITION EMBEDDING
        # ==================================================

        self.position_embedding = PositionalEmbedding(
            context_length,
            embedding_dim
        )

        # ==================================================
        # TRANSFORMER BLOCKS
        # ==================================================

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads
                )
                for _ in range(num_layers)
            ]
        )

        # ==================================================
        # FINAL NORMALIZATION
        # ==================================================

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        # ==================================================
        # LANGUAGE MODEL HEAD
        # ==================================================

        self.output_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(
        self,
        input_ids,
        return_attention=False
    ):

        batch_size, sequence_length = input_ids.shape

        # ==================================================
        # CONTEXT CHECK
        # ==================================================

        if sequence_length > self.context_length:

            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds context length "
                f"{self.context_length}"
            )

        # ==================================================
        # TOKEN EMBEDDINGS
        # ==================================================

        token_vectors = self.token_embedding(
            input_ids
        )

        # Shape:

        # [B, T, D]

        # ==================================================
        # POSITION INDICES
        # ==================================================

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # Shape:

        # [T]

        # ==================================================
        # POSITION EMBEDDINGS
        # ==================================================

        position_vectors = self.position_embedding(
            positions
        )

        # Shape:

        # [T, D]

        # ==================================================
        # COMBINE TOKEN + POSITION
        # ==================================================

        x = token_vectors + position_vectors

        # Shape:

        # [B, T, D]

        # ==================================================
        # TRANSFORMER
        # ==================================================

        attention_history = []

        for block in self.transformer_blocks:

            if return_attention:

                x, attention_weights = block(
                    x,
                    return_attention=True
                )

                attention_history.append(
                    attention_weights
                )

            else:

                x = block(x)

        # ==================================================
        # FINAL NORMALIZATION
        # ==================================================

        x = self.final_norm(x)

        # ==================================================
        # OUTPUT LOGITS
        # ==================================================

        logits = self.output_head(x)

        # Shape:

        # [B, T, vocab_size]

        # ==================================================
        # OUTPUT
        # ==================================================

        if return_attention:

            return logits, attention_history

        return logits