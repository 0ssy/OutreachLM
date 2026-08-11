import torch
import torch.nn as nn

from outreachlm.model import OutreachModel


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16

LEARNING_RATE = 0.001
TRAINING_STEPS = 200


# ============================================================
# BUILD DATA
# ============================================================

# We deliberately DO NOT train on every sequence.
#
# Training:
#   0 -> 1
#   2 -> 3
#   4 -> 5
#   ...
#
# Validation:
#   1 -> 2
#   3 -> 4
#   5 -> 6
#   ...
#
# The model therefore has to generalize to transitions
# it never directly observed.

training_examples = []
validation_examples = []


for start in range(0, 15):

    input_sequence = [
        start,
        start + 1,
        start + 2,
        start + 3,
        start + 4
    ]

    target_sequence = [
        start + 1,
        start + 2,
        start + 3,
        start + 4,
        start + 5
    ]

    # Even starting positions -> training
    if start % 2 == 0:
        training_examples.append(
            (
                input_sequence,
                target_sequence
            )
        )

    # Odd starting positions -> validation
    else:
        validation_examples.append(
            (
                input_sequence,
                target_sequence
            )
        )


# ============================================================
# DISPLAY DATASET
# ============================================================

print("=" * 60)
print("OUTREACHLM CONTROLLED GENERALIZATION EXPERIMENT")
print("=" * 60)

print()
print("Training examples:")
print()

for i, (x, y) in enumerate(training_examples):

    print(
        f"{i:2d} | "
        f"{x} -> {y}"
    )


print()
print("Validation examples:")
print()

for i, (x, y) in enumerate(validation_examples):

    print(
        f"{i:2d} | "
        f"{x} -> {y}"
    )


# ============================================================
# MODEL
# ============================================================

model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
)


# ============================================================
# LOSS
# ============================================================

loss_function = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)

model.train()


for step in range(
    1,
    TRAINING_STEPS + 1
):

    total_loss = 0.0

    for input_sequence, target_sequence in training_examples:

        input_ids = torch.tensor(
            [input_sequence],
            dtype=torch.long
        )

        target_ids = torch.tensor(
            [target_sequence],
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            input_ids
        )

        # ----------------------------------------------------
        # Cross entropy
        # ----------------------------------------------------

        loss = loss_function(
            logits.transpose(1, 2),
            target_ids
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model
        # ----------------------------------------------------

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(training_examples)
    )

    if (
        step == 1
        or step % 20 == 0
    ):

        print(
            f"Step {step:3d} | "
            f"Training Loss: {average_loss:.6f}"
        )


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("CONTROLLED GENERALIZATION")
print("=" * 60)

model.eval()

total_correct = 0
total_predictions = 0

validation_loss = 0.0


with torch.no_grad():

    for input_sequence, target_sequence in validation_examples:

        input_ids = torch.tensor(
            [input_sequence],
            dtype=torch.long
        )

        target_ids = torch.tensor(
            [target_sequence],
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(
            input_ids
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = loss_function(
            logits.transpose(1, 2),
            target_ids
        )

        validation_loss += loss.item()

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        correct = (
            predictions == target_ids
        ).sum().item()

        total_correct += correct
        total_predictions += target_ids.numel()

        print()
        print("Input     :", input_sequence)
        print("Target    :", target_sequence)
        print(
            "Prediction:",
            predictions[0].tolist()
        )


# ============================================================
# RESULTS
# ============================================================

average_validation_loss = (
    validation_loss /
    len(validation_examples)
)

accuracy = (
    total_correct /
    total_predictions
) * 100


print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print()
print(
    "Validation Loss:",
    f"{average_validation_loss:.6f}"
)

print(
    "Validation Accuracy:",
    f"{accuracy:.2f}%"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()

if accuracy == 100.0:

    print(
        "✓ The model generalized perfectly "
        "to unseen starting positions."
    )

elif accuracy >= 50.0:

    print(
        "⚠ The model shows partial "
        "generalization."
    )

else:

    print(
        "✗ The model did not generalize "
        "well to the unseen transitions."
    )

print()
print("=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)