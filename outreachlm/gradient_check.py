import torch

torch.manual_seed(42)

print("=" * 60)
print("OUTREACHLM GRADIENT CHECK")
print("=" * 60)

# ------------------------------------------------------------
# Use float64 for numerical gradient checking.
# ------------------------------------------------------------

dtype = torch.float64

x = torch.tensor(
    2.0,
    dtype=dtype
)

target = torch.tensor(
    10.0,
    dtype=dtype
)

w = torch.tensor(
    3.0,
    dtype=dtype,
    requires_grad=True
)

print()
print("Initial parameter:")
print("w =", w.item())

# ------------------------------------------------------------
# Forward
# ------------------------------------------------------------

y = w * x

print()
print("Forward pass:")
print("x     =", x.item())
print("w     =", w.item())
print("y=wx  =", y.item())

# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

loss = (y - target) ** 2

print()
print("Target:")
print("target =", target.item())

print()
print("Loss:")
print("L = (y - target)^2")
print("L =", loss.item())

# ------------------------------------------------------------
# Backward
# ------------------------------------------------------------

loss.backward()

autograd_gradient = w.grad.item()

print()
print("Gradient calculated by PyTorch:")
print("dL/dw =", autograd_gradient)

# ------------------------------------------------------------
# Analytical derivative
#
# L = (wx - target)^2
#
# dL/dw = 2(wx - target)x
# ------------------------------------------------------------

analytical_gradient = (
    2.0
    * (w.detach() * x - target)
    * x
)

analytical_gradient_value = analytical_gradient.item()

print()
print("Analytical gradient:")
print("dL/dw =", analytical_gradient_value)

# ------------------------------------------------------------
# Numerical derivative
#
# f'(w) ≈
#
# [f(w + epsilon) - f(w - epsilon)]
# ---------------------------------
#             2 epsilon
# ------------------------------------------------------------

epsilon = 1e-5

w_plus = w.detach() + epsilon
w_minus = w.detach() - epsilon

loss_plus = (
    (w_plus * x - target) ** 2
)

loss_minus = (
    (w_minus * x - target) ** 2
)

numerical_gradient = (
    loss_plus - loss_minus
) / (2.0 * epsilon)

numerical_gradient_value = numerical_gradient.item()

print()
print("Numerical gradient:")
print("dL/dw =", numerical_gradient_value)

# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

analytical_error = abs(
    autograd_gradient
    - analytical_gradient_value
)

numerical_error = abs(
    autograd_gradient
    - numerical_gradient_value
)

print()
print("=" * 60)
print("GRADIENT COMPARISON")
print("=" * 60)

print()
print(
    f"Autograd gradient:     {autograd_gradient:.8f}"
)

print(
    f"Analytical gradient:   {analytical_gradient_value:.8f}"
)

print(
    f"Numerical gradient:    {numerical_gradient_value:.8f}"
)

print()
print(
    f"Autograd vs analytical error: "
    f"{analytical_error:.8e}"
)

print(
    f"Autograd vs numerical error:  "
    f"{numerical_error:.8e}"
)

print()

if analytical_error < 1e-10:
    print(
        "✓ Analytical gradient matches PyTorch autograd."
    )
else:
    print(
        "✗ Analytical gradient does not match autograd."
    )

if numerical_error < 1e-6:
    print(
        "✓ Numerical gradient agrees with autograd."
    )
else:
    print(
        "✗ Numerical gradient check failed."
    )

print()
print("=" * 60)
print("GRADIENT CHECK COMPLETE")
print("=" * 60)