import torch
import torch.nn as nn

from outreachlm.model import OutreachModel
from outreachlm.datasets import LanguageModelDataset


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16

LEARNING_RATE = 0.001
TRAINING_STEPS = 100


# ============================================================
# DATASET
# ============================================================

token_ids = torch.arange(VOCAB_SIZE)

dataset = LanguageModelDataset(
    token_ids=token_ids,
    context_length=CONTEXT_LENGTH
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
# LOSS FUNCTION
# ============================================================

loss_function = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING INFORMATION
# ============================================================

print("=" * 60)
print("OUTREACHLM TRAINING")
print("=" * 60)

print()
print("Training examples:", len(dataset))
print("Context length:", CONTEXT_LENGTH)
print("Vocabulary size:", VOCAB_SIZE)
print("Embedding dimension:", EMBEDDING_DIM)
print("Learning rate:", LEARNING_RATE)
print("Training steps:", TRAINING_STEPS)

print()


# ============================================================
# SHOW FIRST TRAINING EXAMPLE
# ============================================================

first_input, first_target = dataset[0]

print("First training example:")
print()
print("Input :", first_input.tolist())
print("Target:", first_target.tolist())

print()


# ============================================================
# TRAINING LOOP
# ============================================================

model.train()

for step in range(TRAINING_STEPS):

    total_loss = 0.0

    # --------------------------------------------------------
    # Train on every example in the dataset
    # --------------------------------------------------------

    for index in range(len(dataset)):

        # ----------------------------------------------------
        # Get training example
        # ----------------------------------------------------

        input_ids, target_ids = dataset[index]

        # ----------------------------------------------------
        # Add batch dimension
        #
        # Before:
        #
        # [sequence]
        #
        # After:
        #
        # [batch, sequence]
        # ----------------------------------------------------

        input_ids = input_ids.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(input_ids)

        # ----------------------------------------------------
        # Logits shape:
        #
        # [batch, sequence, vocabulary]
        #
        # Example:
        #
        # [1, 5, 20]
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Cross-entropy loss
        #
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
        # Therefore transpose dimensions 1 and 2.
        # ----------------------------------------------------

        loss = loss_function(
            logits.transpose(1, 2),
            target_ids
        )

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model parameters
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Accumulate loss
        # ----------------------------------------------------

        total_loss += loss.item()

    # --------------------------------------------------------
    # Calculate average loss for this step
    # --------------------------------------------------------

    average_loss = total_loss / len(dataset)

    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if step == 0 or (step + 1) % 10 == 0:

        print(
            f"Step {step + 1:3d} | "
            f"Loss: {average_loss:.6f}"
        )


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 60)
print("OUTREACHLM EVALUATION")
print("=" * 60)


model.eval()


# ============================================================
# TEST INPUT
# ============================================================

test_input = torch.tensor([
    [10, 11, 12, 13, 14]
])


print()
print("Test input:")
print(test_input)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    logits = model(test_input)


print()
print("Logits shape:")
print(logits.shape)


# ============================================================
# FINAL POSITION
# ============================================================

# The final position is the model's prediction
# for the token that should come after the input.

final_logits = logits[:, -1, :]


# ============================================================
# CONVERT LOGITS TO PROBABILITIES
# ============================================================

probabilities = torch.softmax(
    final_logits,
    dim=-1
)


# ============================================================
# MOST LIKELY TOKEN
# ============================================================

predicted_token = torch.argmax(
    probabilities,
    dim=-1
)


print()
print("Predicted next token:")
print(predicted_token)


# ============================================================
# TOP 5 PREDICTIONS
# ============================================================

top_probabilities, top_tokens = torch.topk(
    probabilities,
    k=5,
    dim=-1
)


print()
print("Top 5 predictions:")
print()

for token, probability in zip(
    top_tokens[0],
    top_probabilities[0]
):

    print(
        f"Token {token.item():2d} "
        f"| Probability: {probability.item():.6f}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("NEXT-TOKEN PREDICTION")
print("=" * 60)

print()

print(
    "Input:",
    test_input[0].tolist()
)

print(
    "Expected next token:",
    15
)

print(
    "Model prediction:",
    predicted_token.item()
)

print()

if predicted_token.item() == 15:

    print(
        "✓ Model predicted the correct next token."
    )

else:

    print(
        "✗ Model did not predict the correct next token."
    )