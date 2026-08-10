import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim
        #Linear projections for the query, key and value
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

        #Final Projection
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )
    
    def forward(self, x):
        # X shape: (batch_size, context_length, embedding_dim)

        batch_size, sequence_length, embedding_dim = x.shape

        #Create Q , K , V matrices
        Q = self.query(x)  # Shape: (batch_size, context_length, embedding_dim)
        K = self.key(x)  # Shape: (batch_size, context_length,    embedding_dim)
        V = self.value(x)  # Shape: (batch_size, context_length, embedding_dim)

        #Attention scores
        # Q @ K^T
        #[B, T, D] @ [B, D, T] -> [B, T, T]

        scores = torch.matmul(Q, K.transpose(-2, -1))

        #Scale the scores
        scores = scores / math.sqrt(embedding_dim)

        #Casual Mask
        #Create a lower triangular matrix to mask future tokens
        #1 0 0 0
        #1 1 0 0
        #1 1 1 0
        #1 1 1 1
        # Each position can only attend to itself
        #and previous positions, not future ones

        casual_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )
        #Turn future positions into infinity
        scores = scores.masked_fill(
            casual_mask == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        #weighted combination of values
        attention_output = torch.matmul(
            attention_weights,
            V
        )

        #Final linear projection
        output = self.output(
            attention_output
        )

        return output
        
