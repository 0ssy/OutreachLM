import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from outreachlm.corpus import TextCorpus
from outreachlm.tokenizer import CharacterTokenizer
from outreachlm.datasets import LanguageModelDataset
from outreachlm.model import OutreachModel


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONTEXT_LENGTH = 32
EMBEDDING_DIM = 64

BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAINING_STEPS = 1000

PROJECT_DIR = Path(__file__).resolve().parent

CORPUS_PATH = PROJECT_DIR / "data1" / "train.txt"
MODEL_PATH = PROJECT_DIR / "outreachlm_model.pt"


# ============================================================
# CORPUS
# ============================================================

def load_corpus():

    corpus = TextCorpus(
        CORPUS_PATH
    )

    text = corpus.load()

    print(
        f"Corpus characters: {len(text)}"
    )

    return text


# ============================================================
# TOKENIZER
# ============================================================

def create_tokenizer(text):

    tokenizer = CharacterTokenizer(
        text
    )

    print(
        f"Vocabulary size:  {tokenizer.vocab_size}"
    )

    return tokenizer


# ============================================================
# DATASET
# ============================================================

def create_dataset(
    tokenizer,
    text
):

    token_ids = tokenizer.encode(
        text
    )

    dataset = LanguageModelDataset(
        token_ids=token_ids,
        context_length=CONTEXT_LENGTH
    )

    print(
        f"Training samples: {len(dataset)}"
    )

    return dataset


# ============================================================
# MODEL
# ============================================================

def create_model(
    vocab_size
):

    model = OutreachModel(
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_dim=EMBEDDING_DIM
    )

    return model.to(DEVICE)


# ============================================================
# TRAINING
# ============================================================

def train():

    print("=" * 60)
    print("OUTREACHLM TRAINING ENGINE")
    print("=" * 60)

    print(
        f"\nDevice:           {DEVICE}"
    )

    print(
        f"Context length:   {CONTEXT_LENGTH}"
    )

    print(
        f"Embedding dim:    {EMBEDDING_DIM}"
    )

    print(
        f"Batch size:       {BATCH_SIZE}"
    )

    print(
        f"Learning rate:    {LEARNING_RATE}"
    )

    print(
        f"Training steps:   {TRAINING_STEPS}"
    )

    # --------------------------------------------------------
    # Load text
    # --------------------------------------------------------

    text = load_corpus()

    # --------------------------------------------------------
    # Build tokenizer
    # --------------------------------------------------------

    tokenizer = create_tokenizer(
        text
    )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    dataset = create_dataset(
        tokenizer,
        text
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        tokenizer.vocab_size
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    step = 0

    while step < TRAINING_STEPS:

        for input_ids, target_ids in dataloader:

            if step >= TRAINING_STEPS:
                break

            input_ids = input_ids.to(
                DEVICE
            )

            target_ids = target_ids.to(
                DEVICE
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(
                input_ids
            )

            # logits:
            #
            # [batch, sequence, vocabulary]
            #
            # target:
            #
            # [batch, sequence]

            batch_size = logits.shape[0]
            sequence_length = logits.shape[1]
            vocab_size = logits.shape[2]

            # ------------------------------------------------
            # Flatten predictions and targets
            # ------------------------------------------------

            logits_for_loss = logits.reshape(
                batch_size * sequence_length,
                vocab_size
            )

            targets_for_loss = target_ids.reshape(
                batch_size * sequence_length
            )

            # ------------------------------------------------
            # Cross entropy
            # ------------------------------------------------

            loss = loss_function(
                logits_for_loss,
                targets_for_loss
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            # ------------------------------------------------
            # Parameter update
            # ------------------------------------------------

            optimizer.step()

            step += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if step == 1 or step % 100 == 0:

                print(
                    f"Step {step:4d} | "
                    f"Loss: {loss.item():.6f}"
                )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\n✓ Model saved to:")
    print(MODEL_PATH)

    return model, tokenizer


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    tokenizer
):

    model.eval()

    text = load_corpus()

    dataset = create_dataset(
        tokenizer,
        text
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0

    total_correct = 0
    total_tokens = 0

    batches = 0

    for input_ids, target_ids in dataloader:

        input_ids = input_ids.to(
            DEVICE
        )

        target_ids = target_ids.to(
            DEVICE
        )

        logits = model(
            input_ids
        )

        batch_size = logits.shape[0]
        sequence_length = logits.shape[1]
        vocab_size = logits.shape[2]

        logits_for_loss = logits.reshape(
            batch_size * sequence_length,
            vocab_size
        )

        targets_for_loss = target_ids.reshape(
            batch_size * sequence_length
        )

        loss = loss_function(
            logits_for_loss,
            targets_for_loss
        )

        total_loss += loss.item()

        batches += 1

        predictions = logits.argmax(
            dim=-1
        )

        total_correct += (
            predictions == target_ids
        ).sum().item()

        total_tokens += target_ids.numel()

    average_loss = (
        total_loss / batches
    )

    accuracy = (
        total_correct / total_tokens
    )

    perplexity = math.exp(
        average_loss
    )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    print(
        f"\nLoss:       {average_loss:.6f}"
    )

    print(
        f"Accuracy:   {accuracy * 100:.2f}%"
    )

    print(
        f"Perplexity: {perplexity:.4f}"
    )

    model.train()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    model, tokenizer = train()

    evaluate(
        model,
        tokenizer
    )