import torch

from outreachlm.model import OutreachModel


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16

MODEL_PATH = "outreachlm_model.pt"


# ============================================================
# LOAD MODEL
# ============================================================

model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint)

model.eval()


# ============================================================
# TRANSITION ANALYSIS
# ============================================================

print("=" * 60)
print("OUTREACHLM TRANSITION ANALYSIS")
print("=" * 60)

print()
print("Testing whether the model learned:")
print()
print("    x -> x + 1")
print()


with torch.no_grad():

    for token in range(VOCAB_SIZE - 1):

        # ----------------------------------------------------
        # Build a context around the token.
        #
        # We use the same type of 5-token context the model
        # was trained on.
        # ----------------------------------------------------

        start = max(0, token - 2)

        context = [
            start,
            start + 1,
            start + 2,
            start + 3,
            start + 4
        ]

        # Make sure every token remains inside vocabulary.
        context = [
            min(x, VOCAB_SIZE - 1)
            for x in context
        ]

        # Put the token we are testing at the final position.
        context[-1] = token

        input_ids = torch.tensor(
            [context],
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits, _ = model(input_ids)

        # ----------------------------------------------------
        # Only inspect the final position.
        # ----------------------------------------------------

        next_token_logits = logits[
            0,
            -1,
            :
        ]

        probabilities = torch.softmax(
            next_token_logits,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities
        ).item()

        confidence = probabilities[
            prediction
        ].item()

        expected = token + 1

        correct = prediction == expected

        print(
            f"{token:2d} -> "
            f"Predicted: {prediction:2d} | "
            f"Expected: {expected:2d} | "
            f"Confidence: {confidence:.6f} | "
            f"{'✓' if correct else '✗'}"
        )


print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)