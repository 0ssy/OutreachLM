import torch

from outreachlm.model import TokenEmbedding
from outreachlm.attention import CausalSelfAttention


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 100
EMBEDDING_DIM = 16
CONTEXT_LENGTH = 5


# ============================================================
# CREATE TOKEN EMBEDDING
# ============================================================

embedding = TokenEmbedding(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM
)


# ============================================================
# CREATE ATTENTION
# ============================================================

attention = CausalSelfAttention(
    embedding_dim=EMBEDDING_DIM
)


# ============================================================
# INPUT
# ============================================================

input_ids = torch.tensor([
    [1, 2, 3, 4, 5]
])


print("=" * 50)
print("CAUSAL ATTENTION TEST")
print("=" * 50)

print("\nInput:")
print(input_ids)


# ============================================================
# EMBEDDING
# ============================================================

x = embedding(input_ids)

print("\nEmbedding shape:")
print(x.shape)


# ============================================================
# ATTENTION
# ============================================================

output = attention(x)

print("\nAttention output shape:")
print(output.shape)

print("\nAttention output:")
print(output)