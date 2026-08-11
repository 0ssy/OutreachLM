import torch

from outreachlm.model import OutreachModel


VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16


print("=" * 60)
print("OUTREACHLM EMBEDDING ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
)


# ------------------------------------------------------------
# Load trained weights
# ------------------------------------------------------------

model.load_state_dict(
    torch.load(
        "outreachlm_model.pt",
        map_location="cpu"
    )
)

model.eval()


print()
print("Model loaded successfully.")


# ------------------------------------------------------------
# Extract token embeddings
# ------------------------------------------------------------

embeddings = model.token_embedding.embedding.weight.detach()


print()
print("Token embedding shape:")
print(embeddings.shape)


# ------------------------------------------------------------
# Print embeddings
# ------------------------------------------------------------

print()
print("=" * 60)
print("TOKEN EMBEDDINGS")
print("=" * 60)


for token_id in range(VOCAB_SIZE):

    print()
    print(f"Token {token_id}:")
    print(embeddings[token_id])


# ------------------------------------------------------------
# Compare neighboring tokens
# ------------------------------------------------------------

print()
print("=" * 60)
print("NEIGHBORING TOKEN DISTANCES")
print("=" * 60)


for token_id in range(VOCAB_SIZE - 1):

    a = embeddings[token_id]
    b = embeddings[token_id + 1]

    distance = torch.norm(a - b).item()

    print(
        f"{token_id} -> {token_id + 1} | "
        f"Distance: {distance:.6f}"
    )


# ------------------------------------------------------------
# Compare distant tokens
# ------------------------------------------------------------

print()
print("=" * 60)
print("SELECTED TOKEN DISTANCES")
print("=" * 60)


pairs = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (0, 10),
    (5, 15),
]


for a_id, b_id in pairs:

    distance = torch.norm(
        embeddings[a_id] - embeddings[b_id]
    ).item()

    print(
        f"{a_id} <-> {b_id} | "
        f"Distance: {distance:.6f}"
    )


print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)