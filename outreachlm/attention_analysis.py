import torch

from outreachlm.model import OutreachModel


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16

MODEL_PATH = "outreachlm_model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("=" * 60)
print("OUTREACHLM ATTENTION ANALYSIS")
print("=" * 60)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = OutreachModel(
    vocab_size=checkpoint["vocab_size"],
    context_length=checkpoint["context_length"],
    embedding_dim=checkpoint["embedding_dim"]
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("Model loaded successfully.")
print(f"Device: {DEVICE}")


# ============================================================
# TEST INPUT
# ============================================================

test_sequence = [
    0,
    1,
    2,
    3,
    4
]

input_ids = torch.tensor(
    test_sequence,
    dtype=torch.long,
    device=DEVICE
).unsqueeze(0)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    logits, attention_weights = model(
        input_ids
    )


# ============================================================
# SHAPES
# ============================================================

print()
print("=" * 60)
print("ATTENTION SHAPES")
print("=" * 60)

print()
print("Input shape:")
print(input_ids.shape)

print()
print("Logits shape:")
print(logits.shape)

print()
print("Attention shape:")
print(attention_weights.shape)


# ============================================================
# EXPECTED SHAPE
# ============================================================

batch_size = input_ids.shape[0]
num_heads = attention_weights.shape[1]
sequence_length = attention_weights.shape[2]

print()
print("Expected:")
print(
    f"[batch={batch_size}, "
    f"heads={num_heads}, "
    f"sequence={sequence_length}, "
    f"sequence={sequence_length}]"
)


# ============================================================
# ATTENTION MATRICES
# ============================================================

print()
print("=" * 60)
print("ATTENTION MATRICES")
print("=" * 60)

attention = attention_weights[0].cpu()


for head in range(num_heads):

    print()
    print("-" * 60)
    print(f"HEAD {head}")
    print("-" * 60)

    matrix = attention[head]

    for position in range(sequence_length):

        values = matrix[position]

        formatted = " ".join(
            f"{value:.4f}"
            for value in values
        )

        print(
            f"Position {position}: "
            f"[{formatted}]"
        )


# ============================================================
# STRONGEST ATTENTION
# ============================================================

print()
print("=" * 60)
print("STRONGEST ATTENTION TARGET")
print("=" * 60)

for head in range(num_heads):

    print()
    print(f"HEAD {head}")

    matrix = attention[head]

    for position in range(sequence_length):

        row = matrix[position]

        strongest_position = torch.argmax(
            row
        ).item()

        strongest_value = row[
            strongest_position
        ].item()

        print(
            f"Position {position} "
            f"-> Position {strongest_position} "
            f"| Weight: {strongest_value:.6f}"
        )


# ============================================================
# ATTENTION + TOKEN INFORMATION
# ============================================================

print()
print("=" * 60)
print("ATTENTION WITH TOKENS")
print("=" * 60)

for head in range(num_heads):

    print()
    print(f"HEAD {head}")

    matrix = attention[head]

    for position in range(sequence_length):

        row = matrix[position]

        strongest_position = torch.argmax(
            row
        ).item()

        source_token = test_sequence[position]

        attended_token = test_sequence[
            strongest_position
        ]

        weight = row[
            strongest_position
        ].item()

        print(
            f"Token {source_token} "
            f"(position {position}) "
            f"-> token {attended_token} "
            f"(position {strongest_position}) "
            f"| weight={weight:.6f}"
        )


# ============================================================
# CAUSALITY CHECK
# ============================================================

print()
print("=" * 60)
print("CAUSALITY CHECK")
print("=" * 60)

causality_passed = True

for head in range(num_heads):

    matrix = attention[head]

    for row in range(sequence_length):

        for column in range(
            row + 1,
            sequence_length
        ):

            value = matrix[
                row,
                column
            ].item()

            if abs(value) > 1e-6:

                causality_passed = False

                print(
                    f"WARNING: Head {head}, "
                    f"position {row} attends "
                    f"to future position {column}: "
                    f"{value:.8f}"
                )


if causality_passed:

    print()
    print(
        "✓ Causal masking is working."
    )

else:

    print()
    print(
        "✗ Causal masking violation detected."
    )


# ============================================================
# ATTENTION ENTROPY
# ============================================================

print()
print("=" * 60)
print("ATTENTION ENTROPY")
print("=" * 60)

for head in range(num_heads):

    matrix = attention[head]

    print()
    print(f"HEAD {head}")

    for position in range(sequence_length):

        row = matrix[position]

        # Avoid log(0)
        safe_row = row.clamp(
            min=1e-12
        )

        entropy = -torch.sum(
            safe_row * torch.log(safe_row)
        ).item()

        print(
            f"Position {position} "
            f"| Entropy: {entropy:.6f}"
        )


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("MODEL PREDICTIONS")
print("=" * 60)

probabilities = torch.softmax(
    logits,
    dim=-1
)

predictions = torch.argmax(
    probabilities,
    dim=-1
)

print()

for position in range(
    sequence_length
):

    prediction = predictions[
        0,
        position
    ].item()

    confidence = probabilities[
        0,
        position,
        prediction
    ].item()

    print(
        f"Input token: {test_sequence[position]} "
        f"| Predicted next token: {prediction} "
        f"| Confidence: {confidence:.6f}"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("ATTENTION ANALYSIS COMPLETE")
print("=" * 60)