"""Does restricted dX pay off at larger d? The scaling argument, measured.

At d=512 K3 wins only 1.05x. The profile explains why: restricting dX by half
saves ~21% of a step, and the bank's bookkeeping consumes most of it.

But the two terms scale differently:

    dX GEMM cost      2 * n * d_out * d_in      grows as d^2
    bank traffic      ~3 * n * d_out            grows as d

So the saving should dominate the overhead as d grows. If it does, restricted
backward is viable at transformer widths even though it is marginal at 512.
If it does not, the mechanism is a dead end regardless of implementation
quality.

Step-count parity is verified separately at keep=0.5 (G=2), where deferral is
2 steps: rung 3h measured 845 steps for BOTH dense and K3 to reach the same
target. Where step count is identical, the per-step ratio IS the end-to-end
ratio, which is why per-step timing is admissible here and was not before.
"""
from __future__ import annotations

import time

import torch

from src.restricted_backward.global_schedule import GlobalScheduleAccumulation

torch.set_num_threads(6)
N = 256
KEEP = 0.5
# Fewer inner reps at large d (each step is expensive), more outer trials,
# and min-of-trials -- this machine is shared, so the minimum is the least
# contended estimate and a single-trial mean drifts by >50% between runs.
REPS_FOR = {512: 60, 1024: 40, 2048: 16, 4096: 6}
TRIALS = 9


def bench_dense(d, trials=TRIALS):
    g = torch.Generator().manual_seed(0)
    W = torch.randn(d, d, generator=g) / d**0.5
    E = torch.randn(N, d, generator=g) * 0.05
    tgt = torch.randn(N, d, generator=g)
    lr = 1e-4

    def step():
        Y = E @ W.T
        G_Y = (2.0 / (N * d)) * (Y - tgt)
        E.sub_(G_Y @ W, alpha=lr)

    reps = REPS_FOR[d]
    step()
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            step()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def bench_k3(d, trials=TRIALS):
    m = GlobalScheduleAccumulation(
        d, d, keep_fraction=KEEP, n_params=N, seed=0
    )
    g = torch.Generator().manual_seed(0)
    E = torch.randn(N, d, generator=g) * 0.05
    tgt = torch.randn(N, d, generator=g)
    lr = 1e-4

    def step():
        Y = m.forward(E)
        G_Y = (2.0 / (N * d)) * (Y - tgt)
        ((a, b), vals), *_ = m.backward(G_Y, None)
        E[:, a:b].sub_(vals, alpha=lr)

    reps = REPS_FOR[d]
    step()
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            step()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def recovery_after(kind, d, seed, lr, steps):
    """Fixed-budget accuracy. If K3 matches dense here, its 2-step deferral
    costs no progress and the per-step ratio IS the end-to-end ratio."""
    if kind == "K3":
        m = GlobalScheduleAccumulation(
            d, d, keep_fraction=KEEP, n_params=N, seed=seed
        )
        W = m.W
    else:
        g0 = torch.Generator().manual_seed(seed)
        W = torch.randn(d, d, generator=g0) / d**0.5
        m = None
    g = torch.Generator().manual_seed(seed + 1)
    E_star = torch.randn(N, d, generator=g) * 0.3
    tgt = E_star @ W.T
    E = torch.randn(N, d, generator=g) * 0.05
    for _ in range(steps):
        G_Y = (2.0 / (N * d)) * (E @ W.T - tgt)
        if m is None:
            E = E - lr * (G_Y @ W)
        else:
            ((a, b), vals), *_ = m.backward(G_Y, None)
            E[:, a:b] -= lr * vals
        if not torch.isfinite(E).all():
            return float("inf")
    return float((E - E_star).norm() / E_star.norm())


def main() -> None:
    print(f"n={N} keep={KEEP} (G=2, deferral 2 steps), 6 threads\n")
    hdr = (f"{'d':>7}{'dense ms':>11}{'K3 ms':>10}{'ratio':>9}"
           f"{'bank MB':>10}")
    print(hdr)
    print("-" * len(hdr))
    for d in (512, 1024, 2048, 4096):
        de = bench_dense(d)
        k3 = bench_k3(d)
        m = GlobalScheduleAccumulation(
            d, d, keep_fraction=KEEP, n_params=N, seed=0
        )
        print(f"{d:>7}{de * 1000:>11.3f}{k3 * 1000:>10.3f}"
              f"{k3 / de:>8.2f}x{m.aux_bytes / 1e6:>10.1f}")

    print("\nPROGRESS PARITY -- same lr, same budget. The per-step ratio is")
    print("only the end-to-end ratio if K3's 2-step deferral costs nothing.")
    print(f"\n{'d':>7}{'lr':>9}{'steps':>7}{'dense':>10}{'K3':>10}{'parity':>9}")
    print("-" * 52)
    for d, lr, steps in ((512, 20000.0, 400), (1024, 20000.0, 400),
                         (2048, 20000.0, 200)):
        de = sum(recovery_after("dense", d, sd, lr, steps) for sd in (0, 1)) / 2
        k3 = sum(recovery_after("K3", d, sd, lr, steps) for sd in (0, 1)) / 2
        ok = "yes" if k3 <= de * 1.05 else "NO"
        print(f"{d:>7}{lr:>9.0f}{steps:>7}{de:>10.4f}{k3:>10.4f}{ok:>9}")


if __name__ == "__main__":
    main()
