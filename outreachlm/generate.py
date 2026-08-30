import json
import os

import torch

from outreachlm.train import (
    DEVICE,
    CONTEXT_LENGTH,
    EMBEDDING_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    CORPUS_PATH,
    VALIDATION_SPLIT,
    CHECKPOINT_PATH,
    BEST_MODEL_PATH,
    MODEL_PATH,
    TOKENIZER_PATH,
    load_corpus,
    split_corpus,
    create_tokenizer,
    create_model,
    save_tokenizer,
)
from outreachlm.tokenizer import CharacterTokenizer
from outreachlm.phase_h_runtime import BoundedStateRuntime


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

DEFAULT_PROMPT = "OutreachLM"

MAX_NEW_TOKENS = 100

# Temperature controls randomness.
#
# < 1.0  = more deterministic
# = 1.0   = normal sampling
# > 1.0  = more random
TEMPERATURE = 0.8

# Top-k limits sampling to the k most probable tokens.
TOP_K = 8
PHASE_H_ARTIFACT_ENV = "OUTREACHLM_PHASE_H_ARTIFACT"


# ============================================================
# LOAD MODEL
# ============================================================

def tokenizer_from_tokens_config(tokenizer_config):
    tokenizer = CharacterTokenizer.__new__(
        CharacterTokenizer
    )

    tokenizer.pad_token = tokenizer_config["pad_token"]
    tokenizer.unk_token = tokenizer_config["unk_token"]
    tokenizer.tokens = tokenizer_config["tokens"]

    tokenizer.token_to_id = {
        token: index
        for index, token in enumerate(tokenizer.tokens)
    }

    tokenizer.id_to_token = {
        index: token
        for token, index in tokenizer.token_to_id.items()
    }

    return tokenizer


def load_tokenizer_artifact(tokenizer_path):
    if not tokenizer_path.exists():
        return None

    with open(
        tokenizer_path,
        "r",
        encoding="utf-8"
    ) as file:
        tokenizer_data = json.load(file)

    if (
        "tokens" in tokenizer_data
        and "pad_token" in tokenizer_data
        and "unk_token" in tokenizer_data
    ):
        return tokenizer_from_tokens_config(
            tokenizer_data
        )

    return None


def upgrade_legacy_tokenizer_artifact(tokenizer_path):
    print(
        "[WARN] Tokenizer artifact is legacy; "
        "upgrading tokenizer JSON to full token mapping."
    )

    text = load_corpus(CORPUS_PATH)
    training_text, _ = split_corpus(
        text,
        VALIDATION_SPLIT
    )

    tokenizer = create_tokenizer(
        training_text
    )

    save_tokenizer(
        tokenizer,
        tokenizer_path
    )

    return tokenizer


def resolve_model_path():
    if BEST_MODEL_PATH.exists():
        return BEST_MODEL_PATH

    print(
        "[WARN] Best model artifact not found; "
        "falling back to outreachlm_model.pt."
    )
    return MODEL_PATH


def load_model_and_tokenizer():

    print("=" * 60)
    print("OUTREACHLM GENERATION")
    print("=" * 60)

    print()
    print(f"Device:           {DEVICE}")
    model_path = resolve_model_path()
    print(f"Model path:       {model_path}")

    tokenizer = load_tokenizer_artifact(
        TOKENIZER_PATH
    )

    if tokenizer is None:
        tokenizer = upgrade_legacy_tokenizer_artifact(
            TOKENIZER_PATH
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
        and "model_config" in checkpoint
    ):
        model_config = checkpoint["model_config"]

        print(
            f"Context length:   {model_config['context_length']}"
        )
        print(
            f"Embedding dim:    {model_config['embedding_dim']}"
        )

        if tokenizer is None:
            raise RuntimeError(
                f"Tokenizer artifact not available: {TOKENIZER_PATH}"
            )

        print(
            f"Vocabulary size:  {tokenizer.vocab_size}"
        )

        model = create_model(
            vocab_size=model_config["vocab_size"],
            context_length=model_config["context_length"],
            embedding_dim=model_config["embedding_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
        )

        if tokenizer.vocab_size != model_config["vocab_size"]:
            raise RuntimeError(
                "Tokenizer/model vocab mismatch: "
                f"tokenizer={tokenizer.vocab_size}, "
                f"model={model_config['vocab_size']}"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        print(
            "[WARN] Model artifact is legacy format; "
            "inferring model config from checkpoint/state dict."
        )

        if tokenizer is None:
            raise RuntimeError(
                f"Tokenizer artifact not available: {TOKENIZER_PATH}"
            )

        state_dict = checkpoint

        token_embedding = state_dict[
            "token_embedding.embedding.weight"
        ]
        inferred_vocab_size = token_embedding.shape[0]
        inferred_embedding_dim = token_embedding.shape[1]

        inferred_context_length = CONTEXT_LENGTH
        inferred_num_layers = NUM_LAYERS
        inferred_num_heads = NUM_HEADS

        if CHECKPOINT_PATH.exists():
            train_checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=DEVICE,
                weights_only=False
            )
            saved_config = train_checkpoint.get(
                "config",
                {}
            )
            inferred_context_length = saved_config.get(
                "context_length",
                inferred_context_length
            )
            inferred_num_layers = saved_config.get(
                "num_layers",
                inferred_num_layers
            )
            inferred_num_heads = saved_config.get(
                "num_heads",
                inferred_num_heads
            )

        if tokenizer.vocab_size != inferred_vocab_size:
            raise RuntimeError(
                "Tokenizer/model vocab mismatch: "
                f"tokenizer={tokenizer.vocab_size}, "
                f"model={inferred_vocab_size}"
            )

        print(
            f"Context length:   {inferred_context_length}"
        )
        print(
            f"Embedding dim:    {inferred_embedding_dim}"
        )
        print(
            f"Vocabulary size:  {tokenizer.vocab_size}"
        )

        model = create_model(
            vocab_size=inferred_vocab_size,
            context_length=inferred_context_length,
            embedding_dim=inferred_embedding_dim,
            num_layers=inferred_num_layers,
            num_heads=inferred_num_heads,
        )

        model.load_state_dict(state_dict)

        upgraded_artifact = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": inferred_vocab_size,
                "context_length": inferred_context_length,
                "embedding_dim": inferred_embedding_dim,
                "num_layers": inferred_num_layers,
                "num_heads": inferred_num_heads,
            },
            "training_config": saved_config if CHECKPOINT_PATH.exists() else {},
            "tokenizer_config": {
                "tokens": tokenizer.tokens,
                "pad_token": tokenizer.pad_token,
                "unk_token": tokenizer.unk_token,
            },
        }

        torch.save(
            upgraded_artifact,
            model_path
        )

        print(
            f"[OK] Upgraded legacy model artifact: {model_path}"
        )

    model.to(DEVICE)
    model.eval()

    print()
    print("[OK] Model loaded successfully.")

    return model, tokenizer


# ============================================================
# TOP-K FILTERING
# ============================================================

def top_k_filter(logits, k):

    if k is None:
        return logits

    if k <= 0:
        return logits

    k = min(k, logits.size(-1))

    values, indices = torch.topk(
        logits,
        k=k,
        dim=-1
    )

    filtered = torch.full_like(
        logits,
        float("-inf")
    )

    filtered.scatter_(
        -1,
        indices,
        values
    )

    return filtered


# ============================================================
# SAMPLE NEXT TOKEN
# ============================================================

def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None
):

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than 0"
        )

    # --------------------------------------------------------
    # Temperature scaling
    # --------------------------------------------------------

    logits = logits / temperature

    # --------------------------------------------------------
    # Top-k filtering
    # --------------------------------------------------------

    logits = top_k_filter(
        logits,
        top_k
    )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    # --------------------------------------------------------
    # Sample from probability distribution
    # --------------------------------------------------------

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return next_token.item(), probabilities


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=8
):

    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    token_ids = tokenizer.encode(prompt)

    generated = list(token_ids)

    print()
    print("============================================================")
    print("GENERATION")
    print("============================================================")
    print()
    print(f"Prompt: {prompt}")
    print()
    print("Generating...")
    print()

    # --------------------------------------------------------
    # Autoregressive generation
    # --------------------------------------------------------

    for step in range(max_new_tokens):

        # Keep only the most recent context window.
        context = generated[-model.context_length:]

        input_ids = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits = model(input_ids)

        # Last position predicts the next token.
        next_token_logits = logits[:, -1, :]

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------

        next_token, probabilities = sample_next_token(
            next_token_logits[0],
            temperature=temperature,
            top_k=top_k
        )

        generated.append(next_token)

        # ----------------------------------------------------
        # Decode token
        # ----------------------------------------------------

        character = tokenizer.decode(
            [next_token]
        )

        confidence = probabilities[next_token].item()

        print(
            f"Step {step + 1:3d} | "
            f"Token: {next_token:3d} | "
            f"Character: {character!r:6s} | "
            f"Probability: {confidence:.6f}"
        )

    # --------------------------------------------------------
    # Decode complete sequence
    # --------------------------------------------------------

    generated_text = tokenizer.decode(
        generated
    )

    print()
    print("============================================================")
    print("GENERATED TEXT")
    print("============================================================")
    print()
    print(generated_text)

    return generated_text


# ============================================================
# MAIN
# ============================================================

def main():
    phase_h_artifact = os.environ.get(PHASE_H_ARTIFACT_ENV)
    if phase_h_artifact:
        runtime = BoundedStateRuntime.load(phase_h_artifact)
        result = runtime.generate(
            DEFAULT_PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            apply_safety=True,
        )
        print("=" * 60)
        print("OUTREACHLM PHASE H GENERATION")
        print("=" * 60)
        print()
        print(result["generated_text"])
        return

    model, tokenizer = load_model_and_tokenizer()

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=DEFAULT_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K
    )


if __name__ == "__main__":
    main()