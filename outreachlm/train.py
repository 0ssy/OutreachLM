import torch

from outreachlm.model import OutreachModel
from outreachlm.training_engine import TrainingEngine


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 20

CONTEXT_LENGTH = 5

EMBEDDING_DIM = 16

NUM_LAYERS = 1

NUM_HEADS = 4

LEARNING_RATE = 0.001

TRAINING_STEPS = 500

CHECKPOINT_PATH = (
    "outreachlm_training_checkpoint.pt"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DATASET
# ============================================================

inputs = torch.tensor(
    [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11, 12],
        [9, 10, 11, 12, 13],
        [10, 11, 12, 13, 14],
        [11, 12, 13, 14, 15],
        [12, 13, 14, 15, 16],
        [13, 14, 15, 16, 17]
    ],
    dtype=torch.long
)


targets = torch.tensor(
    [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8],
        [5, 6, 7, 8, 9],
        [6, 7, 8, 9, 10],
        [7, 8, 9, 10, 11],
        [8, 9, 10, 11, 12],
        [9, 10, 11, 12, 13],
        [10, 11, 12, 13, 14],
        [11, 12, 13, 14, 15],
        [12, 13, 14, 15, 16],
        [13, 14, 15, 16, 17],
        [14, 15, 16, 17, 18]
    ],
    dtype=torch.long
)


# ============================================================
# MODEL
# ============================================================

model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS
)


model = model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING ENGINE
# ============================================================

engine = TrainingEngine(
    model=model,
    optimizer=optimizer,
    device=device,
    checkpoint_path=CHECKPOINT_PATH
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 60)
print("OUTREACHLM TRAINING ENGINE")
print("=" * 60)

print()

print(
    f"Training examples: "
    f"{len(inputs)}"
)

print(
    f"Context length: "
    f"{CONTEXT_LENGTH}"
)

print(
    f"Vocabulary size: "
    f"{VOCAB_SIZE}"
)

print(
    f"Embedding dimension: "
    f"{EMBEDDING_DIM}"
)

print(
    f"Transformer layers: "
    f"{NUM_LAYERS}"
)

print(
    f"Attention heads: "
    f"{NUM_HEADS}"
)

print(
    f"Learning rate: "
    f"{LEARNING_RATE}"
)

print(
    f"Training steps: "
    f"{TRAINING_STEPS}"
)

print(
    f"Device: "
    f"{device}"
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 60)
print("TRAINING")
print("=" * 60)


for step in range(
    1,
    TRAINING_STEPS + 1
):

    result = engine.train_step(
        inputs,
        targets
    )

    if (
        step == 1
        or step % 50 == 0
    ):

        print(
            f"Step {step:4d} | "
            f"Loss: {result['loss']:.6f} | "
            f"Gradient Norm: "
            f"{result['gradient_norm']:.6f}"
        )


# ============================================================
# SAVE
# ============================================================

engine.save_checkpoint()


print()
print("✓ Training checkpoint saved to:")
print(CHECKPOINT_PATH)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("VALIDATION")
print("=" * 60)


validation_input = torch.tensor(
    [[14, 15, 16, 17, 18]],
    dtype=torch.long
)


validation_target = torch.tensor(
    [[15, 16, 17, 18, 19]],
    dtype=torch.long
)


validation = engine.validate(
    validation_input,
    validation_target
)


with torch.no_grad():

    logits = model(
        validation_input.to(device)
    )

    predictions = torch.argmax(
        logits,
        dim=-1
    )


print()

print(
    "Input     :",
    validation_input[0].tolist()
)

print(
    "Target    :",
    validation_target[0].tolist()
)

print(
    "Prediction:",
    predictions[0].cpu().tolist()
)

print(
    f"Validation Loss: "
    f"{validation['loss']:.6f}"
)

print(
    f"Validation Accuracy: "
    f"{validation['accuracy'] * 100:.2f}%"
)


# ============================================================
# GRADIENT SUMMARY
# ============================================================

print()
print("=" * 60)
print("TRAINING ENGINE SUMMARY")
print("=" * 60)

print()

print(
    f"Final training loss: "
    f"{engine.history['training_loss'][-1]:.6f}"
)

print(
    f"Final gradient norm: "
    f"{engine.history['gradient_norm'][-1]:.6f}"
)

print(
    f"Validation loss: "
    f"{validation['loss']:.6f}"
)

print(
    f"Validation accuracy: "
    f"{validation['accuracy'] * 100:.2f}%"
)

print()
print("EXPERIMENT COMPLETE")