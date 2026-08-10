import torch
import torch.nn as nn

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
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, context_length):
        super().__init__()
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

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length
            )
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (K.size(-1) ** 0.5)

        sequence_length = x.size(1)

        causal_mask = self.mask[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )
        output = attention_weights @ V

        return output, attention_weights