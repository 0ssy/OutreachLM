import torch

from outreachlm.train import (
    DEVICE,
    CONTEXT_LENGTH,
    EMBEDDING_DIM,
    CORPUS_PATH,
    VALIDATION_SPLIT,
    MODEL_PATH,
    load_corpus,
    split_corpus,
    create_tokenizer,
    create_model,
)


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


# ============================================================
# LOAD MODEL
# ============================================================

def load_model_and_tokenizer():

    print("=" * 60)
    print("OUTREACHLM GENERATION")
    print("=" * 60)

    print()
    print(f"Device:           {DEVICE}")
    print(f"Context length:   {CONTEXT_LENGTH}")
    print(f"Embedding dim:    {EMBEDDING_DIM}")
    print(f"Model path:       {MODEL_PATH}")

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    text = load_corpus(CORPUS_PATH)

    training_text, _ = split_corpus(
        text,
        VALIDATION_SPLIT
    )

    # --------------------------------------------------------
    # Recreate tokenizer
    # --------------------------------------------------------

    tokenizer = create_tokenizer(
        training_text
    )

    print(
        f"Training characters: {len(training_text)}"
    )
    print(f"Vocabulary size:  {tokenizer.vocab_size}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        vocab_size=tokenizer.vocab_size
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

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
        context = generated[-CONTEXT_LENGTH:]

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