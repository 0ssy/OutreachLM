"""Pin down the best expert shape, carefully enough to put in the budget.

A sweep suggested 190 GFLOP/s at (512 tokens x 2048 d_model x 512 d_ff),
against the 150 the budget currently assumes. That is a 1.27x compute win if
real, so it gets the same treatment as the K3 scaling numbers: min-of-many
trials, because this machine is shared and single-trial means drift.

The shape matters twice over: n=512 with d_model=2048 is a 1.05M-parameter
expert, which is the size the budget already uses, so this is a free win from
routing block size rather than an architecture change.
"""
from __future__ import annotations

import time

import torch

torch.set_num_threads(6)
TRIALS = 9


def best(fn, reps):
    fn()
    b = float("inf")
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        b = min(b, (time.perf_counter() - t0) / reps)
    return b


def main() -> None:
    d_model, d_ff = 2048, 512
    print(f"Expert tile {d_model} x {d_ff} "
          f"({d_model * d_ff / 1e6:.2f}M params, "
          f"{d_model * d_ff * 2 / 8 / 1e6:.2f} MB ternary)")
    print(f"min of {TRIALS} trials, 6 threads\n")
    print(f"{'tokens/expert':>15}{'ms':>9}{'GFLOP/s':>11}{'vs 150':>9}")
    print("-" * 44)
    g = torch.Generator().manual_seed(0)
    W = torch.randn(d_model, d_ff, generator=g)
    peak = 0.0
    for n in (128, 256, 512, 1024, 2048):
        X = torch.randn(n, d_model, generator=g)
        work = n * d_model * d_ff
        reps = max(5, min(300, 80_000_000 // max(1, work // 1000)))
        t = best(lambda: X @ W, reps)
        gf = 2.0 * work / t / 1e9
        peak = max(peak, gf)
        print(f"{n:>15}{t * 1000:>9.3f}{gf:>11.1f}{gf / 150:>8.2f}x")

    print(f"\n  sustained best {peak:.0f} GFLOP/s")
    print(f"  budget currently assumes 150 -> compute scales by "
          f"{150 / peak:.2f}x")
    print(f"  23.6 d compute becomes {23.6 * 150 / peak:.1f} d")


if __name__ == "__main__":
    main()
