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
        