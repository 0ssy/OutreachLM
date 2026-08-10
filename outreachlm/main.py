import torch
import torch.nn as nn

from outreachlm.model import OutreachModel


VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16


model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
)


# Training example
input_ids = torch.tensor([
    [0, 1, 2, 3, 4]
])

target_ids = torch.tensor([
    [1, 2, 3, 4, 5]
])


# Forward pass
logits = model(input_ids)


print("=" * 60)
print("OUTREACHLM LOSS TEST")
print("=" * 60)

print()
print("Input:")
print(input_ids)

print()
print("Target:")
print(target_ids)

print()
print("Logits shape:")
print(logits.shape)


# Cross-entropy loss
loss_function = nn.CrossEntropyLoss()


# CrossEntropyLoss expects:
#
# predictions:
# [batch, classes, sequence]
#
# targets:
# [batch, sequence]
#
# Our logits are:
# [batch, sequence, classes]
#
# Therefore we transpose dimensions 1 and 2.

loss = loss_function(
    logits.transpose(1, 2),
    target_ids
)


print()
print("Loss:")
print(loss.item())