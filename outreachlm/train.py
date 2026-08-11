import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from outreachlm.model import OutreachModel


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
TRAIN_STEPS = 1000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "outreachlm_model.pt"


# ============================================================
# DATASET
# ============================================================

class LanguageModelDataset(Dataset):

    def __init__(self, sequences, context_length):
        self.inputs = []
        self.targets = []

        for sequence in sequences:

            if len(sequence) <= context_length:
                continue

            for start in range(
                len(sequence) - context_length
            ):

                input_sequence = sequence[
                    start:start + context_length
                ]

                target_sequence = sequence[
                    start + 1:start + context_length + 1
                ]

                self.inputs.append(input_sequence)
                self.targets.append(target_sequence)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):

        return (
            torch.tensor(
                self.inputs[index],
                dtype=torch.long
            ),
            torch.tensor(
                self.targets[index],
                dtype=torch.long
            )
        )


# ============================================================
# TRAINING DATA
# ============================================================

def create_training_sequences():

    return [
        list(range(0, 10)),
        list(range(2, 12)),
        list(range(4, 14)),
        list(range(6, 16)),
        list(range(8, 18)),
        list(range(10, 20)),
    ]


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = OutreachModel(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM
    )

    return model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

def calculate_loss(logits, targets):

    batch_size, sequence_length, vocab_size = logits.shape

    logits = logits.reshape(
        batch_size * sequence_length,
        vocab_size
    )

    targets = targets.reshape(
        batch_size * sequence_length
    )

    loss = nn.functional.cross_entropy(
        logits,
        targets
    )

    return loss


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader):

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    for inputs, targets in dataloader:

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(inputs)

        loss = calculate_loss(
            logits,
            targets
        )

        token_count = targets.numel()

        total_loss += (
            loss.item() * token_count
        )

        total_tokens += token_count

        predictions = torch.argmax(
            logits,
            dim=-1
        )

        correct_tokens += (
            predictions == targets
        ).sum().item()

    average_loss = (
        total_loss / total_tokens
    )

    accuracy = (
        correct_tokens / total_tokens
    )

    perplexity = math.exp(
        min(average_loss, 20)
    )

    return (
        average_loss,
        accuracy,
        perplexity
    )


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():

    print("=" * 60)
    print("OUTREACHLM TRAINING ENGINE")
    print("=" * 60)

    print()
    print(f"Device:           {DEVICE}")
    print(f"Vocabulary size:  {VOCAB_SIZE}")
    print(f"Context length:   {CONTEXT_LENGTH}")
    print(f"Embedding dim:    {EMBEDDING_DIM}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Training steps:   {TRAIN_STEPS}")

    # --------------------------------------------------------
    # Create data
    # --------------------------------------------------------

    sequences = create_training_sequences()

    dataset = LanguageModelDataset(
        sequences,
        CONTEXT_LENGTH
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    print()
    print(f"Training samples: {len(dataset)}")
    print()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    model.train()

    step = 0

    while step < TRAIN_STEPS:

        for inputs, targets in dataloader:

            if step >= TRAIN_STEPS:
                break

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(inputs)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = calculate_loss(
                logits,
                targets
            )

            # ------------------------------------------------
            # Clear old gradients
            # ------------------------------------------------

            optimizer.zero_grad()

            # ------------------------------------------------
            # Backward pass
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # ------------------------------------------------
            # Gradient descent / AdamW update
            # ------------------------------------------------

            optimizer.step()

            step += 1

            if (
                step == 1
                or step % 100 == 0
            ):

                print(
                    f"Step {step:4d} | "
                    f"Loss: {loss.item():.6f}"
                )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "embedding_dim": EMBEDDING_DIM
        },
        MODEL_PATH
    )

    print()
    print("✓ Model saved to:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    evaluation_loss, accuracy, perplexity = evaluate(
        model,
        dataloader
    )

    print(
        f"Loss:       {evaluation_loss:.6f}"
    )

    print(
        f"Accuracy:   {accuracy * 100:.2f}%"
    )

    print(
        f"Perplexity: {perplexity:.4f}"
    )


if __name__ == "__main__":
    main()