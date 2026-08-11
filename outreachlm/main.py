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
TRAINING_STEPS = 100

MODEL_PATH = "outreachlm_model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

examples = [
    ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
    ([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]),
    ([2, 3, 4, 5, 6], [3, 4, 5, 6, 7]),
    ([3, 4, 5, 6, 7], [4, 5, 6, 7, 8]),
    ([4, 5, 6, 7, 8], [5, 6, 7, 8, 9]),
    ([5, 6, 7, 8, 9], [6, 7, 8, 9, 10]),
    ([6, 7, 8, 9, 10], [7, 8, 9, 10, 11]),
    ([7, 8, 9, 10, 11], [8, 9, 10, 11, 12]),
    ([8, 9, 10, 11, 12], [9, 10, 11, 12, 13]),
    ([9, 10, 11, 12, 13], [10, 11, 12, 13, 14]),
    ([10, 11, 12, 13, 14], [11, 12, 13, 14, 15]),
    ([11, 12, 13, 14, 15], [12, 13, 14, 15, 16]),
    ([12, 13, 14, 15, 16], [13, 14, 15, 16, 17]),
    ([13, 14, 15, 16, 17], [14, 15, 16, 17, 18]),
    ([14, 15, 16, 17, 18], [15, 16, 17, 18, 19]),
]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

training_examples = examples[:-1]
validation_examples = examples[-1:]


# ============================================================
# DISPLAY CONFIGURATION
# ============================================================

print()
print("=" * 60)
print("OUTREACHLM")
print("=" * 60)

print()

print(f"Total examples: {len(examples)}")
print(f"Training examples: {len(training_examples)}")
print(f"Validation examples: {len(validation_examples)}")
print(f"Context length: {CONTEXT_LENGTH}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Training steps: {TRAINING_STEPS}")
print(f"Device: {DEVICE}")

print()

print("Validation example:")
print()

validation_input = validation_examples[0][0]
validation_target = validation_examples[0][1]

print(f"Input : {validation_input}")
print(f"Target: {validation_target}")

print()


# ============================================================
# MODEL
# ============================================================

model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
).to(DEVICE)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LOSS FUNCTION
# ============================================================

loss_function = nn.CrossEntropyLoss()


# ============================================================
# TRAINING
# ============================================================

print("=" * 60)
print("TRAINING")
print("=" * 60)

model.train()

for step in range(1, TRAINING_STEPS + 1):

    total_loss = 0.0

    for input_sequence, target_sequence in training_examples:

        # ----------------------------------------
        # Convert data to tensors
        # ----------------------------------------

        input_ids = torch.tensor(
            input_sequence,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        targets = torch.tensor(
            target_sequence,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        # ----------------------------------------
        # Forward pass
        #
        # OutreachModel now returns:
        #
        # logits
        # attention_weights
        # ----------------------------------------

        logits, _ = model(input_ids)

        # ----------------------------------------
        # Calculate loss
        # ----------------------------------------

        loss = loss_function(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        # ----------------------------------------
        # Backpropagation
        # ----------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(training_examples)
    )

    # ----------------------------------------
    # Training progress
    # ----------------------------------------

    if (
        step == 1
        or step % 10 == 0
    ):
        print(
            f"Step {step:3d} | "
            f"Training Loss: {average_loss:.6f}"
        )


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": VOCAB_SIZE,
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
    },
    MODEL_PATH
)

print()
print("✓ Model saved to:")
print(MODEL_PATH)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)

model.eval()

validation_losses = []
correct_predictions = 0
total_predictions = 0

with torch.no_grad():

    for input_sequence, target_sequence in validation_examples:

        input_ids = torch.tensor(
            input_sequence,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        targets = torch.tensor(
            target_sequence,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        # ----------------------------------------
        # Forward pass
        # ----------------------------------------

        logits, attention_weights = model(
            input_ids
        )

        # ----------------------------------------
        # Validation loss
        # ----------------------------------------

        loss = loss_function(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1)
        )

        validation_losses.append(
            loss.item()
        )

        # ----------------------------------------
        # Predictions
        # ----------------------------------------

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        prediction_list = (
            predictions[0]
            .cpu()
            .tolist()
        )

        print()
        print(f"Input     : {input_sequence}")
        print(f"Target    : {target_sequence}")
        print(f"Prediction: {prediction_list}")

        # ----------------------------------------
        # Accuracy
        # ----------------------------------------

        correct_predictions += (
            predictions == targets
        ).sum().item()

        total_predictions += targets.numel()


# ============================================================
# VALIDATION METRICS
# ============================================================

validation_loss = (
    sum(validation_losses)
    / len(validation_losses)
)

validation_accuracy = (
    correct_predictions
    / total_predictions
    * 100
)

print()

print(
    f"Validation Loss: "
    f"{validation_loss:.6f}"
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy:.2f}%"
)


# ============================================================
# AUTOREGRESSIVE GENERATION
# ============================================================

print()
print("=" * 60)
print("AUTOREGRESSIVE GENERATION")
print("=" * 60)

prompt = [0, 1, 2, 3, 4]

print()
print("Prompt:")
print(prompt)

generated = prompt.copy()

model.eval()

with torch.no_grad():

    for step in range(10):

        # ----------------------------------------
        # Keep only the latest context window
        # ----------------------------------------

        context = generated[
            -CONTEXT_LENGTH:
        ]

        input_ids = torch.tensor(
            context,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        # ----------------------------------------
        # Model prediction
        # ----------------------------------------

        logits, attention_weights = model(
            input_ids
        )

        # ----------------------------------------
        # Use final position
        # ----------------------------------------

        next_token_logits = logits[
            0,
            -1,
            :
        ]

        # ----------------------------------------
        # Convert to probabilities
        # ----------------------------------------

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        # ----------------------------------------
        # Greedy decoding
        # ----------------------------------------

        next_token = torch.argmax(
            probabilities
        ).item()

        confidence = probabilities[
            next_token
        ].item()

        # ----------------------------------------
        # Add token to sequence
        # ----------------------------------------

        generated.append(
            next_token
        )

        print(
            f"Step {step + 1:2d} | "
            f"Context: {context} | "
            f"Next token: {next_token} | "
            f"Confidence: {confidence:.6f}"
        )


# ============================================================
# FINAL GENERATION
# ============================================================

print()
print("Final generated sequence:")
print(generated)

print()
print("=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)