import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.corpus import Corpus
from outreachlm.data_loader_config import DataLoaderConfig, build_data_loader
from outreachlm.experiment_config import (
    load_experiment_config,
    to_train_cli_defaults,
)
from outreachlm.model_config import LegacyV1Config
from outreachlm.model_registry import create_model as create_registered_model
from outreachlm.runtime import SingleDeviceRuntime
from outreachlm.tokenizer import CharacterTokenizer
from outreachlm.tokenizer_artifacts import (
    TokenizerArtifact,
    save_tokenizer_artifact,
)
from outreachlm.datasets import LanguageModelDataset
from outreachlm.checkpoint import (
    build_config,
    save_checkpoint,
    load_checkpoint,
    validate_config,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = PROJECT_DIR.parent / "corpus" / "fineweb"
DEFAULT_MODEL_PATH = PROJECT_DIR / "outreachlm_model.pt"
DEFAULT_TOKENIZER_PATH = PROJECT_DIR / "outreachlm_tokenizer.json"
CHECKPOINT_PATH = PROJECT_DIR / "outreachlm_checkpoint.pt"
BEST_MODEL_PATH = PROJECT_DIR / "outreachlm_best.pt"
SAVE_INTERVAL = 1000
VALIDATION_INTERVAL = 5000

# Backward-compatible aliases for modules that import these names.
CORPUS_PATH = DEFAULT_CORPUS_PATH
MODEL_PATH = DEFAULT_MODEL_PATH
TOKENIZER_PATH = DEFAULT_TOKENIZER_PATH

# ============================================================
# RUNTIME
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
RUNTIME = SingleDeviceRuntime(DEVICE)

# ============================================================
# COMPATIBILITY CONFIGURATION
#
# These names are kept because generate.py may import them.
# Actual values can be overridden from the command line.
# ============================================================

CONTEXT_LENGTH = 32
EMBEDDING_DIM = 64
NUM_LAYERS = 1
NUM_HEADS = 4

BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAINING_STEPS = 1000

VALIDATION_SPLIT = 0.10

LOG_INTERVAL = 100
SEED = 42
WARMUP_STEPS = 500
MIN_LEARNING_RATE_RATIO = 0.1

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


def load_corpus(corpus_path=DEFAULT_CORPUS_PATH):
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
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS
):
    model_config = LegacyV1Config(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    model = create_registered_model(model_config)
    return RUNTIME.prepare_model(model)


def build_model_artifact(
    model,
    tokenizer,
    context_length,
    embedding_dim,
    num_layers,
    num_heads,
    training_config,
):
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": tokenizer.vocab_size,
            "context_length": context_length,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
        },
        "training_config": training_config,
        "tokenizer_config": {
            "tokens": tokenizer.tokens,
            "pad_token": tokenizer.pad_token,
            "unk_token": tokenizer.unk_token,
        },
    }

# ============================================================
# RANDOM MINIBATCH
# ============================================================


def get_random_batch(
    token_ids,
    context_length,
    batch_size,
    device
):
    max_start = (
        len(token_ids)
        - context_length
        - 1
    )

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,)
    )

    inputs = torch.stack([
        token_ids[
            start:
            start + context_length
        ]
        for start in starts
    ])

    targets = torch.stack([
        token_ids[
            start + 1:
            start + context_length + 1
        ]
        for start in starts
    ])

    return (
        inputs.to(device),
        targets.to(device)
    )

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


def get_learning_rate(
    step,
    max_steps,
    base_learning_rate,
    warmup_steps,
    min_learning_rate_ratio,
):
    if step < warmup_steps:

        return (
            base_learning_rate
            * (step + 1)
            / warmup_steps
        )

    if max_steps <= warmup_steps:
        return base_learning_rate

    progress = (
        step - warmup_steps
    ) / (
        max_steps - warmup_steps
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    cosine_decay = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    min_learning_rate = (
        base_learning_rate
        * min_learning_rate_ratio
    )

    return (
        min_learning_rate
        + (
            base_learning_rate
            - min_learning_rate
        )
        * cosine_decay
    )

def evaluate_validation(
    model,
    validation_dataset,
    device,
    batch_size,
    eval_loader_config: DataLoaderConfig | None = None,
):
    model.eval()

    resolved_loader_config = eval_loader_config or DataLoaderConfig(
        batch_size=batch_size,
        shuffle=False,
    )
    dataloader = build_data_loader(
        validation_dataset,
        resolved_loader_config,
    )

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for input_ids, target_ids in dataloader:

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            model_output = model(input_ids)

            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output

            batch_size_actual = logits.shape[0]
            sequence_length = logits.shape[1]
            vocab_size = logits.shape[2]

            logits_for_loss = logits.reshape(
                batch_size_actual * sequence_length,
                vocab_size
            )

            targets_for_loss = target_ids.reshape(
                batch_size_actual * sequence_length
            )

            loss = loss_function(
                logits_for_loss,
                targets_for_loss
            )

            total_loss += loss.item()
            batches += 1

    average_loss = total_loss / batches

    perplexity = math.exp(
        average_loss
    )

    model.train()

    return average_loss, perplexity

# ============================================================
# TRAINING
# ============================================================


def train(
    corpus_path=DEFAULT_CORPUS_PATH,
    model_path=DEFAULT_MODEL_PATH,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    best_model_path=BEST_MODEL_PATH,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    training_steps=TRAINING_STEPS,
    validation_split=VALIDATION_SPLIT,
    log_interval=LOG_INTERVAL,
    seed=SEED,
    warmup_steps=WARMUP_STEPS,
    min_learning_rate_ratio=MIN_LEARNING_RATE_RATIO,
    save_interval=SAVE_INTERVAL,
    validation_interval=VALIDATION_INTERVAL,
    eval_num_workers=0,
    eval_prefetch_factor=2,
    eval_persistent_workers=False,
    eval_pin_memory=False,
    eval_drop_last=False,
    resume=False
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
        f"Transformer layers: {num_layers}"
    )

    print(
        f"Attention heads:    {num_heads}"
    )

    print(
        f"Batch size:         {batch_size}"
    )

    print(
        f"Learning rate:      {learning_rate}"
    )

    print(
        f"Warmup steps:       {warmup_steps}"
    )

    print(
        f"Min LR ratio:       {min_learning_rate_ratio}"
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

    training_token_ids = torch.tensor(
        tokenizer.encode(training_text),
        dtype=torch.long
    )

    validation_token_ids = torch.tensor(
        tokenizer.encode(validation_text),
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Build validation dataset
    # --------------------------------------------------------

    validation_dataset = LanguageModelDataset(
        token_ids=validation_token_ids,
        context_length=context_length
    )

    training_samples = (
        len(training_token_ids)
        - context_length
    )

    print(
        f"Training samples:   {training_samples}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    if training_samples <= 0:
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
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # --------------------------------------------------------
    # AdamW
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )
    eval_loader_config = DataLoaderConfig(
        batch_size=batch_size,
        num_workers=eval_num_workers,
        prefetch_factor=eval_prefetch_factor,
        persistent_workers=eval_persistent_workers,
        pin_memory=eval_pin_memory,
        drop_last=eval_drop_last,
        shuffle=False,
    )

    run_config = build_config(
        context_length=context_length,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        min_learning_rate_ratio=min_learning_rate_ratio,
        validation_split=validation_split,
        seed=seed,
        corpus_path=corpus_path,
        vocab_size=tokenizer.vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        eval_loader_config=eval_loader_config.to_dict(),
    )

    start_step = 0
    best_validation_loss = float("inf")
    last_learning_rate = learning_rate

    if resume and checkpoint_path.exists():
        checkpoint_state = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            DEVICE
        )

        checkpoint_config = dict(
            checkpoint_state.get("config", {})
        )

        if checkpoint_state.get("is_legacy"):
            print(
                "[CHECKPOINT] Legacy checkpoint detected. "
                "Using compatibility mode for missing config fields."
            )

            for key, value in run_config.items():
                checkpoint_config.setdefault(
                    key,
                    value
                )

        validate_config(
            checkpoint_config,
            run_config
        )

        start_step = checkpoint_state["step"]
        best_validation_loss = checkpoint_state[
            "best_validation_loss"
        ]

        print(
            f"\n[CHECKPOINT] Resuming from step {start_step}"
        )

        print(
            f"Previous train loss: {checkpoint_state['train_loss']:.6f}"
        )

        resumed_lr = checkpoint_state.get("trainer_state", {}).get("last_learning_rate")
        if resumed_lr is not None:
            last_learning_rate = resumed_lr

    elif resume:
        raise RuntimeError(
            f"Checkpoint not found for --resume: {checkpoint_path}"
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
    interval_losses = []
    average_train_loss = float("nan")
    validation_loss = float("nan")

    for step in range(
        start_step + 1,
        training_steps + 1
    ):

        # ----------------------------------------------------
        # Random minibatch
        # ----------------------------------------------------

        input_ids, target_ids = get_random_batch(
            training_token_ids,
            context_length,
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

        current_learning_rate = get_learning_rate(
            step=step - 1,
            max_steps=training_steps,
            base_learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            min_learning_rate_ratio=min_learning_rate_ratio,
        )

        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_learning_rate

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------

        optimizer.step()
        last_learning_rate = current_learning_rate

        last_loss = loss.item()
        interval_losses.append(
            last_loss
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        if (
            step % log_interval == 0
            or step == training_steps
        ):
            average_train_loss = (
                sum(interval_losses)
                / len(interval_losses)
            )

            print(
                f"Step {step:5d} | "
                f"Train Loss: {average_train_loss:.6f} | "
                f"LR: {current_learning_rate:.8f}"
            )

            interval_losses.clear()

        if (
            step % validation_interval == 0
            or step == training_steps
        ):

            validation_loss, validation_perplexity = (
                evaluate_validation(
                    model,
                    validation_dataset,
                    DEVICE,
                    batch_size,
                    eval_loader_config=eval_loader_config,
                )
            )

            print(
                f"           "
                f"Validation Loss: {validation_loss:.6f} | "
                f"Perplexity: {validation_perplexity:.4f}"
            )

            if validation_loss < best_validation_loss:

                best_validation_loss = validation_loss

                best_model_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                best_model_artifact = build_model_artifact(
                    model=model,
                    tokenizer=tokenizer,
                    context_length=context_length,
                    embedding_dim=embedding_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    training_config=run_config,
                )

                torch.save(
                    best_model_artifact,
                    best_model_path
                )

                print(
                    "[BEST] Validation loss improved."
                )

                print(
                    best_model_path
                )

        if (
            step % save_interval == 0
            or step == training_steps
        ):
            if math.isnan(average_train_loss):
                average_train_loss = (
                    sum(interval_losses)
                    / len(interval_losses)
                )

            if math.isnan(validation_loss):
                validation_loss, _ = evaluate_validation(
                    model,
                    validation_dataset,
                    DEVICE,
                    batch_size,
                    eval_loader_config=eval_loader_config,
                )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step,
                average_train_loss,
                best_validation_loss,
                run_config,
                trainer_state={
                    "optimizer_step": step,
                    "micro_step": step,
                    "last_learning_rate": last_learning_rate,
                    "interval_loss_count": len(interval_losses),
                },
                metadata={
                    "entrypoint": "outreachlm.train",
                    "resumed": resume,
                },
            )

            print(
                f"\n[CHECKPOINT] Saved at step {step}:"
            )

            print(
                checkpoint_path
            )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_artifact = build_model_artifact(
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        training_config=run_config,
    )

    torch.save(
        model_artifact,
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
    tokenizer_artifact = TokenizerArtifact.from_tokenizer(tokenizer)
    save_tokenizer_artifact(
        tokenizer_artifact,
        tokenizer_path,
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


def _build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Train OutreachLM."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment config JSON. CLI flags override config values.",
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
        "--num-layers",
        type=int,
        default=NUM_LAYERS
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=NUM_HEADS
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

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=WARMUP_STEPS
    )

    parser.add_argument(
        "--min-learning-rate-ratio",
        type=float,
        default=MIN_LEARNING_RATE_RATIO
    )

    parser.add_argument(
        "--validation-interval",
        type=int,
        default=VALIDATION_INTERVAL
    )

    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--eval-prefetch-factor",
        type=int,
        default=2
    )

    parser.add_argument(
        "--eval-persistent-workers",
        action="store_true"
    )

    parser.add_argument(
        "--eval-pin-memory",
        action="store_true"
    )

    parser.add_argument(
        "--eval-drop-last",
        action="store_true"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    return parser


def parse_args(argv: list[str] | None = None):
    parser = _build_arg_parser()
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config is not None:
        experiment_config = load_experiment_config(pre_args.config)
        parser.set_defaults(**to_train_cli_defaults(experiment_config))
    return parser.parse_args(argv)

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
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        training_steps=args.steps,
        validation_split=args.validation_split,
        log_interval=args.log_interval,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        validation_interval=args.validation_interval,
        eval_num_workers=args.eval_num_workers,
        eval_prefetch_factor=args.eval_prefetch_factor,
        eval_persistent_workers=args.eval_persistent_workers,
        eval_pin_memory=args.eval_pin_memory,
        eval_drop_last=args.eval_drop_last,
        resume=args.resume
    )

    evaluate(
        model,
        validation_dataset,
        args.batch_size
    )
