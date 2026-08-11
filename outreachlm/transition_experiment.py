import torch
import torch.nn as nn

from outreachlm.model import OutreachModel


# ============================================================
# OUTREACHLM TRANSITION LEARNING EXPERIMENT
# ============================================================

VOCAB_SIZE = 20
CONTEXT_LENGTH = 1
EMBEDDING_DIM = 16

LEARNING_RATE = 0.001
TRAINING_STEPS = 500

MODEL_PATH = "outreachlm_transition_model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATA
# ============================================================

# Train on transitions 0 -> 1 through 17 -> 18.
#
# The final transition:
#
# 18 -> 19
#
# is completely held out.

training_inputs = torch.tensor(
    [
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [16],
        [17],
    ],
    dtype=torch.long
)

training_targets = torch.tensor(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ],
    dtype=torch.long
)


# Completely unseen transition.
validation_input = torch.tensor(
    [[18]],
    dtype=torch.long
)

validation_target = torch.tensor(
    [19],
    dtype=torch.long
)


# ============================================================
# INFORMATION
# ============================================================

print()
print("=" * 60)
print("OUTREACHLM TRANSITION LEARNING")
print("=" * 60)

print()
print("Goal:")
print("Learn the transition:")
print()
print("    x -> x + 1")
print()

print(f"Training transitions: {len(training_inputs)}")
print("Validation transitions: 1")
print("Held-out transition: 18 -> 19")
print()

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Context length: {CONTEXT_LENGTH}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Training steps: {TRAINING_STEPS}")
print(f"Device: {DEVICE}")


# ============================================================
# MOVE DATA TO DEVICE
# ============================================================

training_inputs = training_inputs.to(DEVICE)
training_targets = training_targets.to(DEVICE)

validation_input = validation_input.to(DEVICE)
validation_target = validation_target.to(DEVICE)


# ============================================================
# MODEL
# ============================================================

model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
).to(DEVICE)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
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

    optimizer.zero_grad()

    logits, _ = model(
        training_inputs
    )

    # logits:
    #
    # [batch, sequence, vocabulary]
    #
    # Since sequence length = 1:
    #
    # [18, 1, 20]

    logits = logits[:, -1, :]

    loss = criterion(
        logits,
        training_targets
    )

    loss.backward()

    optimizer.step()

    if (
        step == 1
        or step % 50 == 0
    ):

        print(
            f"Step {step:3d} | "
            f"Training Loss: {loss.item():.6f}"
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
print("[OK] Model saved to:")
print(MODEL_PATH)


# ============================================================
# TRAINING TRANSITION TEST
# ============================================================

print()
print("=" * 60)
print("TRAINING TRANSITION TEST")
print("=" * 60)

model.eval()

correct = 0

with torch.no_grad():

    logits, _ = model(
        training_inputs
    )

    logits = logits[:, -1, :]

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1
    )

    for i in range(
        len(training_inputs)
    ):

        source = training_inputs[
            i,
            0
        ].item()

        target = training_targets[
            i
        ].item()

        prediction = predictions[
            i
        ].item()

        confidence = probabilities[
            i,
            prediction
        ].item()

        is_correct = (
            prediction == target
        )

        if is_correct:
            correct += 1

        symbol = (
            "[OK]"
            if is_correct
            else "[X]"
        )

        print(
            f"{source:2d} -> "
            f"Predicted: {prediction:2d} | "
            f"Expected: {target:2d} | "
            f"Confidence: {confidence:.6f} | "
            f"{symbol}"
        )


training_accuracy = (
    correct
    / len(training_inputs)
    * 100
)

print()
print(
    f"Training Accuracy: "
    f"{training_accuracy:.2f}%"
)


# ============================================================
# HELD-OUT TRANSITION
# ============================================================

print()
print("=" * 60)
print("HELD-OUT TRANSITION")
print("=" * 60)

with torch.no_grad():

    logits, _ = model(
        validation_input
    )

    logits = logits[:, -1, :]

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    prediction = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        prediction
    ].item()


expected = validation_target.item()

print()
print(
    f"Input     : {validation_input[0].tolist()}"
)

print(
    f"Target    : {expected}"
)

print(
    f"Prediction: {prediction}"
)

print(
    f"Confidence: {confidence:.6f}"
)

if prediction == expected:

    print()
    print(
        "[OK] The model correctly "
        "predicted the unseen transition."
    )

else:

    print()
    print(
        "[X] The model failed to "
        "generalize to the unseen transition."
    )


# ============================================================
# ALL TRANSITIONS
# ============================================================

print()
print("=" * 60)
print("TRANSITION MAP")
print("=" * 60)

print()
print(
    "Testing every transition "
    "from 0 -> 1 through 18 -> 19."
)

print()

all_correct = 0

with torch.no_grad():

    for source in range(
        VOCAB_SIZE - 1
    ):

        input_tensor = torch.tensor(
            [[source]],
            dtype=torch.long,
            device=DEVICE
        )

        logits, _ = model(
            input_tensor
        )

        logits = logits[:, -1, :]

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities,
            dim=-1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

        expected = source + 1

        is_correct = (
            prediction == expected
        )

        if is_correct:
            all_correct += 1

        symbol = (
            "[OK]"
            if is_correct
            else "[X]"
        )

        print(
            f"{source:2d} -> "
            f"{prediction:2d} | "
            f"Expected: {expected:2d} | "
            f"Confidence: {confidence:.6f} | "
            f"{symbol}"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

accuracy = (
    all_correct
    / (VOCAB_SIZE - 1)
    * 100
)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print()
print(
    f"Transition Accuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"Correct transitions: "
    f"{all_correct}/{VOCAB_SIZE - 1}"
)

print()

if accuracy == 100.0:

    print(
        "[OK] OutreachLM learned every "
        "tested transition."
    )

elif accuracy >= 90.0:

    print(
        "[OK] Strong transition learning."
    )

elif accuracy >= 70.0:

    print(
        "[~] Partial transition learning."
    )

else:

    print(
        "[X] Transition learning is weak."
    )


print()
print("=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)