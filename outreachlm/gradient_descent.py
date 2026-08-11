import torch


def main():

    print("=" * 60)
    print("OUTREACHLM GRADIENT DESCENT")
    print("=" * 60)

    # --------------------------------------------------
    # Simple problem
    #
    # y = wx
    #
    # x = 2
    # target = 10
    # --------------------------------------------------

    x = torch.tensor(
        2.0
    )

    target = torch.tensor(
        10.0
    )

    # Initial parameter
    w = torch.tensor(
        3.0,
        requires_grad=True
    )

    learning_rate = 0.1

    steps = 10

    print()
    print("Initial parameter:")
    print("w =", w.item())

    print()
    print("Learning rate:")
    print("eta =", learning_rate)

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for step in range(1, steps + 1):

        # --------------------------------------------------
        # FORWARD PASS
        # --------------------------------------------------

        y = w * x

        # --------------------------------------------------
        # LOSS
        # --------------------------------------------------

        loss = (
            y - target
        ) ** 2

        # --------------------------------------------------
        # BACKPROPAGATION
        # --------------------------------------------------

        loss.backward()

        # --------------------------------------------------
        # GRADIENT DESCENT
        #
        # Important:
        #
        # Parameter updates should not themselves
        # become part of the autograd graph.
        # --------------------------------------------------

        with torch.no_grad():

            w -= learning_rate * w.grad

        # --------------------------------------------------
        # Clear gradient before next iteration.
        #
        # PyTorch accumulates gradients by default.
        # --------------------------------------------------

        w.grad.zero_()

        print(
            f"Step {step:2d} | "
            f"w = {w.item():.8f} | "
            f"Loss = {loss.item():.8f}"
        )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print()
    print("Learned w:")
    print(
        f"{w.item():.8f}"
    )

    print()
    print("Expected w:")
    print(
        "5.00000000"
    )

    print()

    if abs(w.item() - 5.0) < 1e-3:

        print(
            "[OK] Gradient descent learned the parameter."
        )

    else:

        print(
            "[X] Gradient descent did not converge."
        )


if __name__ == "__main__":
    main()
