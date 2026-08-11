import torch

from outreachlm.model import OutreachLM


MODEL_PATH = "outreachlm_model.pt"

CONTEXT_LENGTH = 5
VOCAB_SIZE = 20
EMBEDDING_DIM = 16


def main():

    print()
    print("=" * 60)
    print("OUTREACHLM ATTENTION ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = OutreachLM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        context_length=CONTEXT_LENGTH
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print()
    print("Model loaded successfully.")

    # --------------------------------------------------
    # Test sequence
    # --------------------------------------------------

    input_ids = torch.tensor(
        [[0, 1, 2, 3, 4]],
        dtype=torch.long
    )

    print()
    print("Input:")
    print(input_ids)

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        output = model(
            input_ids
        )

    # --------------------------------------------------
    # Handle model output
    # --------------------------------------------------

    if isinstance(output, tuple):

        logits = output[0]
        attention_maps = output[1]

    else:

        raise RuntimeError(
            "The model is not returning attention weights."
        )

    print()
    print("Logits shape:")
    print(logits.shape)

    # --------------------------------------------------
    # Attention analysis
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("ATTENTION MAPS")
    print("=" * 60)

    if not isinstance(attention_maps, list):

        attention_maps = [
            attention_maps
        ]

    for layer_index, attention_weights in enumerate(
        attention_maps
    ):

        print()
        print(
            f"Layer {layer_index}"
        )

        print(
            "Attention shape:",
            attention_weights.shape
        )

        # Expected:
        #
        # [batch, heads, sequence, sequence]
        #

        for head_index in range(
            attention_weights.shape[1]
        ):

            print()
            print(
                f"Head {head_index}"
            )

            head = attention_weights[
                0,
                head_index
            ]

            for position in range(
                head.shape[0]
            ):

                weights = head[
                    position
                ]

                print(
                    f"Position {position}:"
                )

                for attended_position in range(
                    weights.shape[0]
                ):

                    weight = weights[
                        attended_position
                    ].item()

                    if weight > 0.001:

                        print(
                            f"  -> position "
                            f"{attended_position}: "
                            f"{weight:.6f}"
                        )

    # --------------------------------------------------
    # Most attended position
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("MOST ATTENDED POSITIONS")
    print("=" * 60)

    for layer_index, attention_weights in enumerate(
        attention_maps
    ):

        print()
        print(
            f"Layer {layer_index}"
        )

        for head_index in range(
            attention_weights.shape[1]
        ):

            print()
            print(
                f"Head {head_index}"
            )

            head = attention_weights[
                0,
                head_index
            ]

            for position in range(
                head.shape[0]
            ):

                weights = head[
                    position
                ]

                max_position = torch.argmax(
                    weights
                ).item()

                max_weight = weights[
                    max_position
                ].item()

                print(
                    f"Position {position} "
                    f"-> Position {max_position} "
                    f"| Weight: "
                    f"{max_weight:.6f}"
                )

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()