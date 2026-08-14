import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.corpus import Corpus
from outreachlm.tokenizer import CharacterTokenizer
from outreachlm.datasets import LanguageModelDataset
from outreachlm.model import OutreachModel

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = PROJECT_DIR.parent / "corpus" / "fineweb"
DEFAULT_MODEL_PATH = PROJECT_DIR / "outreachlm_model.pt"
DEFAULT_TOKENIZER_PATH = PROJECT_DIR / "outreachlm_tokenizer.json"

# ============================================================
# RUNTIME
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# COMPATIBILITY CONFIGURATION
#
# These names are kept because generate.py may import them.
# Actual values can be overridden from the command line.
# ============================================================

CONTEXT_LENGTH = 32
EMBEDDING_DIM = 64

BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAINING_STEPS = 1000

VALIDATION_SPLIT = 0.10

LOG_INTERVAL = 100
SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# CORPUS
# ============================================================


def load_corpus(corpus_path):
    corpus = Corpus(
        str(corpus_path)
    )

    text = corpus.load()

    if not text.strip():
        raise RuntimeError(
            "Corpus is empty."
        )

    print(
        f"Corpus characters: {len(text)}"
    )

    return text

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================


def split_corpus(
    text,
    validation_split
):
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            "validation_split must be between 0 and 1."
        )

    split_index = int(
        len(text) * (1.0 - validation_split)
    )

    if split_index <= 0:
        raise ValueError(
            "Training portion of corpus is empty."
        )

    if split_index >= len(text):
        raise ValueError(
            "Validation portion of corpus is empty."
        )

    training_text = text[:split_index]
    validation_text = text[split_index:]

    print(
        f"Training characters:   {len(training_text)}"
    )

    print(
        f"Validation characters: {len(validation_text)}"
    )

    return training_text, validation_text

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
    text,
    context_length
):
    token_ids = tokenizer.encode(
        text
    )

    dataset = LanguageModelDataset(
        token_ids=token_ids,
        context_length=context_length
    )

    return dataset

# ============================================================
# MODEL
# ============================================================


def create_model(
    vocab_size,
    context_length,
    embedding_dim
):
    model = OutreachModel(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim
    )

    return model.to(DEVICE)

# ============================================================
# RANDOM MINIBATCH
# ============================================================


def get_random_batch(
    dataset,
    batch_size,
    device
):
    dataset_size = len(dataset)

    indices = torch.randint(
        low=0,
        high=dataset_size,
        size=(batch_size,)
    )

    inputs = []
    targets = []

    for index in indices.tolist():

        input_ids, target_ids = dataset[index]

        inputs.append(input_ids)
        targets.append(target_ids)

    input_batch = torch.stack(
        inputs
    ).to(device)

    target_batch = torch.stack(
        targets
    ).to(device)

    return input_batch, target_batch

# ============================================================
# LOSS
# ============================================================


def calculate_loss(
    logits,
    target_ids,
    loss_function
):
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

    return loss_function(
        logits_for_loss,
        targets_for_loss
    )

# ============================================================
# TRAINING
# ============================================================


def train(
    corpus_path=DEFAULT_CORPUS_PATH,
    model_path=DEFAULT_MODEL_PATH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    training_steps=TRAINING_STEPS,
    validation_split=VALIDATION_SPLIT,
    log_interval=LOG_INTERVAL,
    seed=SEED
):

    print("=" * 60)
    print("OUTREACHLM TRAINING ENGINE")
    print("=" * 60)

    print(
        f"\nDevice:             {DEVICE}"
    )

    print(
        f"Context length:     {context_length}"
    )

    print(
        f"Embedding dim:      {embedding_dim}"
    )

    print(
        f"Batch size:         {batch_size}"
    )

    print(
        f"Learning rate:      {learning_rate}"
    )

    print(
        f"Training steps:     {training_steps}"
    )

    print(
        f"Validation split:   {validation_split:.2%}"
    )

    print()

    set_seed(seed)

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    text = load_corpus(
        corpus_path
    )

    # --------------------------------------------------------
    # Split corpus
    # --------------------------------------------------------

    training_text, validation_text = split_corpus(
        text,
        validation_split
    )

    # --------------------------------------------------------
    # Build tokenizer ONLY from training data
    # --------------------------------------------------------

    tokenizer = create_tokenizer(
        training_text
    )

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    training_dataset = create_dataset(
        tokenizer,
        training_text,
        context_length
    )

    validation_dataset = create_dataset(
        tokenizer,
        validation_text,
        context_length
    )

    print(
        f"Training samples:   {len(training_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    if len(training_dataset) == 0:
        raise RuntimeError(
            "Training dataset contains no samples."
        )

    if len(validation_dataset) == 0:
        raise RuntimeError(
            "Validation dataset contains no samples."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim
    )

    # --------------------------------------------------------
    # AdamW
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------------
    # Cross entropy
    # --------------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    model.train()

    for step in range(
        1,
        training_steps + 1
    ):

        # ----------------------------------------------------
        # Random minibatch
        # ----------------------------------------------------

        input_ids, target_ids = get_random_batch(
            training_dataset,
            batch_size,
            DEVICE
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        model_output = model(
            input_ids
        )

        if isinstance(model_output, tuple):
            logits = model_output[0]
        else:
            logits = model_output

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = calculate_loss(
            logits,
            target_ids,
            loss_function
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            step == 1
            or step % log_interval == 0
            or step == training_steps
        ):

            print(
                f"Step {step:5d} | "
                f"Loss: {loss.item():.6f}"
            )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        model_path
    )

    print()
    print("[OK] Model saved to:")
    print(model_path)

    # --------------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------------

    save_tokenizer(
        tokenizer,
        tokenizer_path
    )

    print("[OK] Tokenizer saved to:")
    print(tokenizer_path)

    return (
        model,
        tokenizer,
        validation_dataset
    )

# ============================================================
# TOKENIZER CHECKPOINT
# ============================================================


def save_tokenizer(
    tokenizer,
    tokenizer_path
):
    tokenizer_data = {
        "vocab_size": tokenizer.vocab_size,
    }

    if hasattr(tokenizer, "stoi"):
        tokenizer_data["stoi"] = tokenizer.stoi

    if hasattr(tokenizer, "itos"):
        tokenizer_data["itos"] = tokenizer.itos

    tokenizer_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        tokenizer_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tokenizer_data,
            file,
            ensure_ascii=False,
            indent=2
        )

# ============================================================
# EVALUATION
# ============================================================


@torch.no_grad()
def evaluate(
    model,
    validation_dataset,
    batch_size
):

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0

    total_correct = 0
    total_tokens = 0

    batches = 0

    # --------------------------------------------------------
    # Evaluate the entire validation dataset
    # --------------------------------------------------------

    for start in range(
        0,
        len(validation_dataset),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(validation_dataset)
        )

        inputs = []
        targets = []

        for index in range(
            start,
            end
        ):

            input_ids, target_ids = (
                validation_dataset[index]
            )

            inputs.append(
                input_ids
            )

            targets.append(
                target_ids
            )

        input_batch = torch.stack(
            inputs
        ).to(DEVICE)

        target_batch = torch.stack(
            targets
        ).to(DEVICE)

        model_output = model(
            input_batch
        )

        if isinstance(model_output, tuple):
            logits = model_output[0]
        else:
            logits = model_output

        loss = calculate_loss(
            logits,
            target_batch,
            loss_function
        )

        batch_tokens = target_batch.numel()

        total_loss += (
            loss.item() * batch_tokens
        )

        predictions = logits.argmax(
            dim=-1
        )

        total_correct += (
            predictions == target_batch
        ).sum().item()

        total_tokens += batch_tokens

        batches += 1

    if total_tokens == 0:
        raise RuntimeError(
            "Validation produced zero tokens."
        )

    average_loss = (
        total_loss / total_tokens
    )

    accuracy = (
        total_correct / total_tokens
    )

    perplexity = math.exp(
        average_loss
    )

    print()
    print("=" * 60)
    print("VALIDATION")
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

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
    }

# ============================================================
# COMMAND LINE
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train OutreachLM."
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH
    )

    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH
    )

    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=TRAINING_STEPS
    )

    parser.add_argument(
        "--validation-split",
        type=float,
        default=VALIDATION_SPLIT
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=LOG_INTERVAL
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    args = parse_args()

    model, tokenizer, validation_dataset = train(
        corpus_path=args.corpus,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        context_length=args.context_length,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_steps=args.steps,
        validation_split=args.validation_split,
        log_interval=args.log_interval,
        seed=args.seed
    )

    evaluate(
        model,
        validation_dataset,
        args.batch_size
    )
