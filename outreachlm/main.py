from outreachlm.model import (
    TokenEmbedding,
    PositionalEmbedding,
    SelfAttention
)
import torch


vocab_size = 100
context_length = 5
embedding_dim = 16


token_embedding = TokenEmbedding(
    vocab_size,
    embedding_dim
)

position_embedding = PositionalEmbedding(
    context_length,
    embedding_dim
)


token_ids = torch.tensor([
    [1, 2, 3, 4, 5]
])


positions = torch.arange(
    context_length
)


token_vectors = token_embedding(token_ids)

position_vectors = position_embedding(positions)


print("\n==================================================")
print("POSITIONAL EMBEDDING TEST")
print("==================================================")

print("Token IDs:")
print(token_ids)

print("\nPositions:")
print(positions)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nPosition embedding shape:")
print(position_vectors.shape)

combined = token_vectors + position_vectors
print("\nCombined shape:")
print(combined.shape)

attention = SelfAttention(
    embedding_dim=16,
    context_length=5
)

attention_output, attention_weights = attention(
    combined
)

print("\n==================================================")
print("SELF-ATTENTION TEST")
print("==================================================")

print("Input shape:")
print(combined.shape)

print("\nAttention output shape:")
print(attention_output.shape)
print("\nAttention weights shape:")
print(attention_weights.shape)

future_attention = torch.triu(
    attention_weights,
    diagonal=1
)

print("\nFuture attention:")
print(future_attention)

print(
    "\nCausal constraint satisfied:",
    torch.all(future_attention == 0).item()
)