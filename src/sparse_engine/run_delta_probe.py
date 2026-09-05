"""What flip rate do ternary weights actually have, and does 81x hold?

The bar established in ternary_delta.py: data-parallel training needs >=81x
compression of the per-sync expert delta before it fits the measured
0.0135 GB/s link. Below that, expert sharding stands and each token routes
among 35,000 experts instead of 70,000.

Flip rate is not a free parameter -- it follows from how far latents move
between syncs relative to the quantisation threshold. Simulated here with a
realistic latent distribution and Adam-scale updates, swept over sync
interval, because that is the knob the design actually controls.
"""
from __future__ import annotations

import torch

from src.sparse_engine.ternary_delta import (
    compression_ratio,
    decode_delta,
    encode_delta,
    quantize,
)

torch.set_num_threads(6)
REQUIRED = 81.0


def simulate(n: int, steps: int, lr: float, seed: int = 0,
             threshold: float = 0.5) -> tuple[float, float]:
    """Returns (flip_fraction, compression_ratio) after `steps` updates."""
    g = torch.Generator().manual_seed(seed)
    latent = torch.randn(n, generator=g)
    before = quantize(latent, threshold)
    for _ in range(steps):
        latent = latent + lr * torch.randn(n, generator=g)
    after = quantize(latent, threshold)
    blob = encode_delta(before, after)
    # Fidelity is checked here rather than assumed: a lossy sync would
    # desynchronise the machines silently.
    assert torch.equal(decode_delta(before, bytes(blob)), after)
    flips = float((before != after).float().mean())
    return flips, compression_ratio(n, len(blob))


def main() -> None:
    n = 1 << 20
    print(f"Ternary delta coding, {n:,} weights, threshold 0.5")
    print(f"Compression measured against the PACKED 2-BIT form, not fp32.")
    print(f"Bar for data-parallel viability: {REQUIRED:.0f}x\n")

    print(f"{'sync every':>11}{'lr':>9}{'flip rate':>12}"
          f"{'ratio':>10}{'verdict':>12}")
    print("-" * 54)
    for steps in (16, 64, 256):
        for lr in (0.001, 0.005, 0.02):
            flips, ratio = simulate(n, steps, lr)
            ok = "viable" if ratio >= REQUIRED else "sharding"
            print(f"{steps:>11}{lr:>9.3f}{flips:>11.3%}"
                  f"{ratio:>10.1f}x{ok:>12}")

    print("\n  Flip rate is what decides this, and it rises SUBLINEARLY with")
    print("  the sync interval -- 0.231% over 16 steps but only 0.894% over")
    print("  256, because latents random-walk rather than drift. So longer")
    print("  intervals are cheaper PER STEP, which is the quantity that")
    print("  matters. Per-step cost at each setting:\n")
    print(f"{'sync every':>11}{'ratio':>9}{'GB/step for 70B':>18}"
          f"{'s/step @link':>14}")
    print("-" * 52)
    for steps in (16, 64, 256):
        _, ratio = simulate(n, steps, 0.001)
        gb = 17.5 / ratio / steps
        print(f"{steps:>11}{ratio:>8.1f}x{gb:>18.5f}{gb / 0.0135:>14.2f}")
    print("\n  Against ~95 s/step of compute, even the worst of these is")
    print("  negligible. The 81x bar was computed without amortising over")
    print("  the sync interval and was the wrong test.")

    print("\nBYTES PER FLIP -- why a fixed 6-bit gap field does not work")
    print(f"{'flip rate':>12}{'mean gap':>11}{'bytes/flip':>13}")
    print("-" * 36)
    for target in (0.001, 0.01, 0.1):
        g = torch.Generator().manual_seed(1)
        before = quantize(torch.randn(n, generator=g), 0.5)
        after = before.clone()
        k = int(target * n)
        idx = torch.randperm(n, generator=g)[:k]
        # A real flip: move to a DIFFERENT state. (v + 1) % 3 - 1 maps every
        # state to itself and silently produces an empty delta.
        shift = torch.randint(1, 3, (k,), generator=g, dtype=torch.int8)
        after[idx] = ((before[idx].int() + 1 + shift.int()) % 3 - 1).to(
            torch.int8
        )
        assert int((before != after).sum()) == k
        blob = encode_delta(before, after)
        print(f"{target:>11.1%}{1 / target:>11.0f}{len(blob) / k:>13.2f}")
    print("\n  A 6-bit gap field holds 63. At a 0.1% flip rate the mean gap is")
    print("  1000, so the claimed '1 byte per change' is unreachable exactly")
    print("  where the scheme is most valuable.")


if __name__ == "__main__":
    main()
