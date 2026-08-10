import torch

from outreachlm.model import OutreachModel


VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16


model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
)


input_ids = torch.tensor([
    [1, 2, 3, 4, 5]
])


logits = model(input_ids)


print("=" * 60)
print("OUTREACHLM MODEL TEST")
print("=" * 60)

print()
print("Input IDs:")
print(input_ids)

print()
print("Input shape:")
print(input_ids.shape)

print()
print("Logits shape:")
print(logits.shape)

print()
print("Logits:")
print(logits)

print()
print("Prediction shape:")
print(logits.shape)

print()
print("✓ Model produces vocabulary logits.")