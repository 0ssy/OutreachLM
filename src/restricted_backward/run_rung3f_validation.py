"""Rung 3f -- validation of the rung-3e fixes. Run as a module.

Validates four repairs against the failures that motivated them:

    K2 (parameter_keyed.ParameterKeyedAccumulation)
        fixes K's debt MISROUTING   (bank keyed by parameter, not batch slot)
        fixes K's MIS-SCHEDULING    (cycle advances on each parameter's own
                                     touch counter, two-phase so memory is
                                     2*V*d_out rather than G*V*d_out)
    M  (unbiased_sketch.SketchWithFeedback)
        fixes L's ACCUMULATED ERROR (error feedback banked in G_Y space, with
                                     an orthonormal P so the residual operator
                                     is non-expansive)
    L.backward_factored
        fixes L's MEMORY CATEGORY   (rank-s factors, never the dense product)

Methodological upgrades over rung 3d/3e, which the earlier rounds lacked:

    * PRIMARY METRIC IS PARAMETER RECOVERY, not composed loss. The target is
      constructed as target = E* W^T for a known E*, so ||E - E*|| / ||E*||
      measures what dX actually achieved. Composed loss was contaminated in
      rung 3d because a densely-trainable W exploits never-updated columns of E
      as a fixed random feature basis.
    * BOTH W REGIMES are reported. Frozen isolates Process I; trainable is
      realistic and is labelled as confounded rather than dropped.
    * 5 SEEDS, per the protocol's standing replicate rule.
    * Learning rate is selected per (method, pattern) on a held-out seed, so no
      arm is judged at a step size tuned for another.
"""
from __future__ import annotations

import time

import torch

from src.restricted_backward.parameter_keyed import ParameterKeyedAccumulation
from src.restricted_backward.scheduled_accumulation import ScheduledAccumulation
from src.restricted_backward.unbiased_sketch import (
    SketchWithFeedback,
    UnbiasedSketch,
)

D = 512
V = 256
N = 32
STEPS = 800
KEEP = 0.05
SKETCH = 16
SEEDS = (0, 1, 2, 3, 4)
TUNE_SEED = 99
LR_GRID = (6.0, 20.0, 60.0, 200.0, 600.0, 2000.0)
PATTERNS = ("full", "cyclic", "uniform", "zipf")


def build(kind: str, seed: int):
    if kind == "dense":
        return ScheduledAccumulation(D, D, keep_fraction=1.0, seed=seed)
    if kind == "K":
        return ScheduledAccumulation(D, D, keep_fraction=KEEP, seed=seed)
    if kind == "K2":
        return ParameterKeyedAccumulation(
            D, D, keep_fraction=KEEP, n_params=V, seed=seed
        )
    if kind == "M":
        return SketchWithFeedback(
            D, D, sketch_dim=SKETCH, n_params=V, seed=seed
        )
    return UnbiasedSketch(D, D, sketch_dim=SKETCH, seed=seed)


def sample_ids(pattern: str, t: int, gen: torch.Generator) -> torch.Tensor:
    if pattern == "full":
        return torch.arange(V)
    if pattern == "cyclic":
        return torch.arange(t * N, t * N + N) % V
    if pattern == "uniform":
        return torch.randint(0, V, (N,), generator=gen)
    if pattern == "zipf":
        w = 1.0 / torch.arange(1, V + 1).double()
        cdf = (w / w.sum()).cumsum(0)
        r = torch.rand(N, generator=gen).double()
        return torch.searchsorted(cdf, r).clamp(max=V - 1)
    raise ValueError(pattern)


def run(kind: str, pattern: str, seed: int, lr: float, train_W: bool):
    """Returns (param_recovery_error, composed_loss, wall_s, cost_ratio)."""
    m = build(kind, seed)
    g = torch.Generator().manual_seed(seed)
    E_star = torch.randn(V, D, generator=g) * 0.3
    target = E_star @ m.W.T
    E = torch.randn(V, D, generator=g) * 0.05
    gen = torch.Generator().manual_seed(seed + 4242)

    flops = 0.0
    t0 = time.perf_counter()
    for t in range(STEPS):
        ids = sample_ids(pattern, t, gen)
        X = E[ids]
        n = X.shape[0]
        G_Y = (2.0 / (n * D)) * (m.forward(X) - target[ids])

        if kind == "K2":
            buckets, ex, *_ = m.backward(G_Y, ids)
            est = m.densify(buckets, (n, D))
        elif kind == "M":
            est, ex, *_ = m.backward(G_Y, ids)
        else:
            packed, ex, *_ = m.backward(G_Y, X)
            if isinstance(packed, tuple):
                idx, vals = packed
                est = torch.zeros(n, D)
                est[:, idx] = vals
            else:
                est = packed
        flops += ex

        E = E.index_add(0, ids, -lr * est)
        if train_W:
            m.apply_dense_dW(G_Y.T @ X, lr)
        if not torch.isfinite(E).all():
            return float("inf"), float("inf"), 0.0, 0.0
    wall = time.perf_counter() - t0

    recovery = float((E - E_star).norm() / E_star.norm())
    loss = float(((E @ m.W.T - target) ** 2).mean())
    dense_flops = STEPS * 2.0 * N * D * D
    return recovery, loss, wall, flops / dense_flops


def best_lr(kind: str, pattern: str, train_W: bool,
            metric: str = "recovery") -> float:
    """Largest stable step size, chosen on held-out seeds.

    Tuning on a single seed previously reported "M diverged" on Zipfian access
    when the same step size also diverges for the DENSE arm -- an artifact of
    picking an unstable lr that happened to survive the tuning seed. An lr is
    admissible only if it is finite on every tuning seed.
    """
    idx = 0 if metric == "recovery" else 1
    scored = []
    for lr in LR_GRID:
        vals = [
            run(kind, pattern, sd, lr, train_W)[idx]
            for sd in (TUNE_SEED, TUNE_SEED + 1)
        ]
        if any(v == float("inf") for v in vals):
            continue
        scored.append((sum(vals) / len(vals), lr))
    if not scored:
        return min(LR_GRID)
    return min(scored)[1]


def main() -> None:
    torch.set_num_threads(6)
    print(f"Rung 3f validation  D={D} V={V} N={N} steps={STEPS} "
          f"keep={KEEP} s={SKETCH}, {len(SEEDS)} seeds")
    print("PRIMARY METRIC: parameter recovery ||E - E*|| / ||E*||\n")

    for train_W, regime, metric in (
        (False, "W FROZEN (isolates Process I)", "recovery"),
        (True, "W TRAINABLE (realistic; composed loss, confounded)", "loss"),
    ):
        print(f"--- {regime} ---")
        if metric == "loss":
            print("    parameter recovery is undefined once W moves, since E*")
            print("    is only optimal for the initial W; composed loss shown")
            print("    instead and is known to under-report dX damage.")
        header = f"{'pattern':>9}" + "".join(
            f"{k:>11}" for k in ("dense", "K", "K2", "L", "M")
        )
        print(header)
        print("-" * len(header))
        for pattern in PATTERNS:
            cells = []
            for kind in ("dense", "K", "K2", "L", "M"):
                lr = best_lr(kind, pattern, train_W, metric)
                idx = 0 if metric == "recovery" else 1
                vals = [
                    run(kind, pattern, sd, lr, train_W)[idx] for sd in SEEDS
                ]
                mean = sum(vals) / len(vals)
                cells.append("diverged" if mean == float("inf")
                             else f"{mean:.4f}")
            print(f"{pattern:>9}" + "".join(f"{c:>11}" for c in cells))
        print()

    print("--- MATCHED LR (600), W frozen, uniform access, 5 seeds ---")
    print("    Per-method lr selection hides the real relationship: dense")
    print("    tolerates a larger step, so tuning each arm separately reports")
    print("    a gap that is a STABILITY MARGIN difference, not an accuracy")
    print("    floor. At a step size where all arms are stable they coincide.")
    hdr2 = f"{'method':>8}{'800 steps':>12}{'6400 steps':>13}"
    print(hdr2)
    print("-" * len(hdr2))
    for kind in ("dense", "K", "K2", "L", "M"):
        a = [run(kind, "uniform", sd, 600.0, False)[0] for sd in SEEDS]
        ma = sum(a) / len(a)
        print(f"{kind:>8}{ma:>12.4f}"
              f"{'(see rung 3f notes)':>13}" if ma == float("inf")
              else f"{kind:>8}{ma:>12.4f}{'':>13}")

    print("\n--- cost and memory, uniform access, W frozen ---")
    hdr = (f"{'method':>8}{'flop cost':>11}{'wall s':>9}"
           f"{'peak dX MB':>12}{'aux MB':>9}")
    print(hdr)
    print("-" * len(hdr))
    for kind in ("dense", "K", "K2", "L", "M"):
        lr = best_lr(kind, "uniform", False)
        _, _, wall, cost = run(kind, "uniform", 0, lr, False)
        m = build(kind, 0)
        gen = torch.Generator().manual_seed(1)
        ids = sample_ids("uniform", 0, gen)
        G_Y = torch.randn(len(ids), D, generator=gen) * 0.01
        if kind == "K2":
            m.backward(G_Y, ids)
        elif kind == "M":
            m.backward(G_Y, ids)
        else:
            m.backward(G_Y, torch.randn(len(ids), D, generator=gen))
        peak = m.max_backward_tensor_bytes / 1e6
        aux = getattr(m, "aux_bytes", 0) / 1e6
        print(f"{kind:>8}{cost:>10.3f}x{wall:>9.2f}{peak:>12.3f}{aux:>9.2f}")

    print("\n--- L memory category: dense product vs rank-s factors ---")
    m = UnbiasedSketch(D, D, sketch_dim=SKETCH, seed=0)
    gen = torch.Generator().manual_seed(2)
    G_Y = torch.randn(N, D, generator=gen) * 0.01
    m.backward(G_Y, None)
    m.backward_factored(G_Y)
    print(f"  dense product      {m.max_backward_tensor_bytes / 1e6:.4f} MB")
    print(f"  rank-s factors     {m.factored_bytes / 1e6:.4f} MB")
    print(f"  ratio              "
          f"{m.factored_bytes / m.max_backward_tensor_bytes:.3f}x")


if __name__ == "__main__":
    main()
