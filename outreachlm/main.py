import torch

from outreachlm.model import TokenEmbedding
from outreachlm.attention import CausalSelfAttention


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 500
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16


# ============================================================
# DATASET TEST
# ============================================================

print("=" * 50)
print("DATASET TEST")
print("=" * 50)

tokens = list(range(20))

print(f"Total tokens: {len(tokens)}")
print(f"Context length: {CONTEXT_LENGTH}")

dataset_length = len(tokens) - CONTEXT_LENGTH

print(f"Dataset length: {dataset_length}")
print()


for i in range(min(3, dataset_length)):

    input_sequence = tokens[i:i + CONTEXT_LENGTH]

    target_sequence = tokens[i + 1:i + CONTEXT_LENGTH + 1]

    print(f"Example {i}")
    print(f"Input : {input_sequence}")
    print(f"Target: {target_sequence}")
    print()


# ============================================================
# EMBEDDING TEST
# ============================================================

print("=" * 50)
print("EMBEDDING TEST")
print("=" * 50)

# Example token IDs
input_ids = torch.tensor(
    [[1, 2, 3, 4, 5]],
    dtype=torch.long
)

print("Input:")
print(input_ids)

print("\nInput shape:")
print(input_ids.shape)


# Create embedding layer
embedding = TokenEmbedding(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM
)


# Run input through embedding layer
embedding_output = embedding(input_ids)


print("\nOutput shape:")
print(embedding_output.shape)

print("\nEmbedding output:")
print(embedding_output)


# ============================================================
# CAUSAL SELF-ATTENTION TEST
# ============================================================

print()
print("=" * 50)
print("CAUSAL SELF-ATTENTION TEST")
print("=" * 50)


# Create attention layer
attention = CausalSelfAttention(
    embedding_dim=EMBEDDING_DIM
)


# Run embeddings through causal attention
attention_output, attention_weights = attention(
    embedding_output
)


print("Input shape:")
print(embedding_output.shape)


print("\nAttention output shape:")
print(attention_output.shape)


print("\nAttention output:")
print(attention_output)


# ============================================================
# ATTENTION WEIGHTS
# ============================================================

print()
print("=" * 50)
print("ATTENTION WEIGHTS")
print("=" * 50)

print(attention_weights)


# ============================================================
# VERIFY CAUSAL MASK
# ============================================================

print()
print("=" * 50)
print("CAUSAL MASK VERIFICATION")
print("=" * 50)


# Everything above the diagonal should be zero.

upper_triangle = torch.triu(
    attention_weights,
    diagonal=1
)


print("Future-token attention values:")
print(upper_triangle)


# Check whether future-token attention is zero
causal_attention_is_valid = torch.allclose(
    upper_triangle,
    torch.zeros_like(upper_triangle),
    atol=1e-6
)


print()


if causal_attention_is_valid:

    print("✓ Causal mask is working.")

    print(
        "✓ No token is attending to a future token."
    )

else:

    print("✗ Causal mask verification failed.")

    print(
        "✗ Future-token attention was detected."
    )


# ============================================================
# FINAL SHAPE CHECK
# ============================================================

print()
print("=" * 50)
print("FINAL SHAPE CHECK")
print("=" * 50)

print(f"Input IDs       : {input_ids.shape}")
print(f"Embeddings      : {embedding_output.shape}")
print(f"Attention output: {attention_output.shape}")


if attention_output.shape == embedding_output.shape:

    print("\n✓ Attention preserves the embedding shape.")

else:

    print("\n✗ Unexpected attention output shape.")


print()
print("=" * 50)
print("ATTENTION STAGE COMPLETE")
print("=" * 50)

print("\n" + "=" * 50)
print("MULTI-HEAD ATTENTION TEST")
print("=" * 50)

attention = CausalSelfAttention(
    embedding_dim=16,
    num_heads=4
)

attention_output, attention_weights = attention(
    embedding_output
)

print("\nAttention output shape:")
print(attention_output.shape)

expected_shape = embedding_output.shape

if attention_output.shape == expected_shape:
    print("✓ Multi-head attention preserves the embedding shape.")
else:
    print("✗ Shape mismatch.")