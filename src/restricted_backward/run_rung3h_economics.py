"""Rung 3h -- can restricted backward beat dense on wall-clock to target?

Rung 3g answered "no" but the audit contained a methodology defect of its own:
learning rate was selected by FINAL ERROR AT A FIXED STEP COUNT, then the
result was reported as STEPS TO TARGET. Those are different objectives and
they select different learning rates -- a method whose best fixed-budget error
comes at lr=600 may reach a given target fastest at lr=2000.

This run fixes that and sweeps the two configuration knobs that were
previously frozen by assumption (keep=0.05 for the deferral methods, s=16 for
the sketch methods), against the metric that actually decides.

Selection rule: for each method and configuration, choose the learning rate
that minimises STEPS TO TARGET, then time exactly that many steps with no
instrumentation in the timed path.

ACCESS REGIME MATTERS, AND THE FIRST VERSION OF THIS BENCHMARK CHOSE THE WRONG
ONE. Sparse embedding access (V=256, N=32) means a parameter is touched once
per 8 steps, so a deferral of G=2 TOUCHES is 16 STEPS of staleness -- crippling
over a 100-step run. That is the worst case for deferral and it is not the case
Process I serves in a transformer, where dX carries gradient to activations
that are present at every step. Both regimes are reported:

    sparse   V=256, N=32   a parameter is touched every ~8 steps
    dense    V=N=256       every unit present every step (transformer-like)
"""
from __future__ import annotations

import time

import torch

from src.restricted_backward.global_schedule import GlobalScheduleAccumulation
from src.restricted_backward.parameter_keyed import ParameterKeyedAccumulation
from src.restricted_backward.scheduled_accumulation import ScheduledAccumulation
from src.restricted_backward.unbiased_sketch import (
    SketchWithFeedback,
    UnbiasedSketch,
)

D = 512
V = 256
N = 32
N_DENSE = 256
TARGET = 0.15
MAX_STEPS = 2500
CHECK_EVERY = 5
SEEDS = (0, 1, 2)
LR_LADDER = [60.0, 200.0, 600.0, 2000.0, 6000.0, 20000.0]


class PlainDense:
    """Honest dense reference: dX = G_Y @ W and nothing else.

    Earlier runs used ScheduledAccumulation(keep_fraction=1.0) as "dense".
    That class still runs the full banking path -- a float64 running total, a
    snapshot read and a snapshot write every step -- so it is Method K with
    G=1, not a dense baseline. Using it handicaps the reference and inflates
    every reported speedup. Any claim of a win must be made against this.
    """

    def __init__(self, d_in, d_out, seed=0, dtype=torch.float32):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5
        self.d_in, self.d_out = d_in, d_out
        self.max_backward_tensor_bytes = 0
        self.aux_bytes = 0

    def forward(self, X):
        return X @ self.W.T

    def backward(self, G_Y, X):
        dX = G_Y @ self.W
        self.max_backward_tensor_bytes = max(
            self.max_backward_tensor_bytes, dX.numel() * G_Y.element_size()
        )
        return dX, 2.0 * G_Y.shape[0] * self.d_out * self.d_in, 0.0, 0.0, 0.0

    def exact_dX(self, G_Y):
        return G_Y @ self.W

    def apply_dense_dW(self, dW, lr):
        self.W -= lr * dW


def build(spec, seed, n_params=V):
    kind = spec[0]
    if kind == "dense":
        return PlainDense(D, D, seed=seed)
    if kind == "K3":
        return GlobalScheduleAccumulation(
            D, D, keep_fraction=spec[1], n_params=n_params, seed=seed
        )
    if kind == "K2":
        return ParameterKeyedAccumulation(
            D, D, keep_fraction=spec[1], n_params=n_params, seed=seed
        )
    if kind == "L":
        return UnbiasedSketch(D, D, sketch_dim=spec[1], seed=seed)
    if kind == "M":
        return SketchWithFeedback(
            D, D, sketch_dim=spec[1], n_params=n_params, seed=seed
        )
    raise ValueError(spec)


def _apply(m, spec, G_Y, X, ids, n, E, lr, full):
    """Apply the restricted update WITHOUT materialising a dense estimate.

    Densifying to (n x d_in) and then index_add-ing the whole width discards
    the memory advantage the restricted methods exist to provide, and adds a
    zero-fill plus a full-width scatter to every step. A restricted method
    computed only some columns, so only those columns are written.
    """
    kind = spec[0]
    if kind == "dense":
        est, *_ = m.backward(G_Y, X)
        if full:
            E -= lr * est
            return E
        return E.index_add(0, ids, -lr * est)
    if kind == "K3":
        ((a, b), vals), *_ = m.backward(G_Y, None if full else ids)
        if full:
            E[:, a:b] -= lr * vals
        else:
            E[:, a:b] = E[:, a:b].index_add(0, ids, -lr * vals)
        return E
    if kind == "K2":
        buckets, *_ = m.backward(G_Y, ids)
        est = m.densify(buckets, (n, D))
    elif kind == "M":
        est, *_ = m.backward(G_Y, ids)
    else:
        packed, *_ = m.backward(G_Y, X)
        if isinstance(packed, tuple):
            idx, vals = packed
            if full:
                E[:, idx] -= lr * vals
                return E
            est = torch.zeros(n, D)
            est[:, idx] = vals
        else:
            est = packed
    if full:
        E -= lr * est
        return E
    return E.index_add(0, ids, -lr * est)


def tgt_rows(tgt, ids, regime):
    return tgt if regime != "sparse" else tgt[ids]


def steps_to_target(spec, seed, lr, regime="sparse", max_steps=MAX_STEPS):
    """Instrumented: returns step count at which recovery first hits TARGET."""
    n_p = V if regime == "sparse" else N_DENSE
    m = build(spec, seed, n_p)
    g = torch.Generator().manual_seed(seed)
    E_star = torch.randn(n_p, D, generator=g) * 0.3
    tgt = E_star @ m.W.T
    E = torch.randn(n_p, D, generator=g) * 0.05
    gen = torch.Generator().manual_seed(seed + 4242)
    full_ids = torch.arange(n_p)
    for t in range(max_steps):
        ids = (torch.randint(0, n_p, (N,), generator=gen)
               if regime == "sparse" else full_ids)
        n = ids.shape[0]
        X = E[ids] if regime == "sparse" else E
        G_Y = (2.0 / (n * D)) * (m.forward(X) - tgt_rows(tgt, ids, regime))
        E = _apply(m, spec, G_Y, X, ids, n, E, lr, regime != "sparse")
        if not torch.isfinite(E).all():
            return None
        if (t + 1) % CHECK_EVERY == 0:
            if float((E - E_star).norm() / E_star.norm()) <= TARGET:
                return t + 1
    return None


def timed(spec, seed, lr, steps, regime="sparse"):
    """Uninstrumented: no oracle, no error metric inside the timer."""
    n_p = V if regime == "sparse" else N_DENSE
    m = build(spec, seed, n_p)
    g = torch.Generator().manual_seed(seed)
    E_star = torch.randn(n_p, D, generator=g) * 0.3
    tgt = E_star @ m.W.T
    E = torch.randn(n_p, D, generator=g) * 0.05
    gen = torch.Generator().manual_seed(seed + 4242)
    full_ids = torch.arange(n_p)
    t0 = time.perf_counter()
    for _ in range(steps):
        ids = (torch.randint(0, n_p, (N,), generator=gen)
               if regime == "sparse" else full_ids)
        n = ids.shape[0]
        X = E[ids] if regime == "sparse" else E
        G_Y = (2.0 / (n * D)) * (m.forward(X) - tgt_rows(tgt, ids, regime))
        E = _apply(m, spec, G_Y, X, ids, n, E, lr, regime != "sparse")
    return time.perf_counter() - t0


def evaluate(spec, regime="sparse"):
    """LR chosen to minimise STEPS TO TARGET -- the metric being reported."""
    best = None
    for lr in LR_LADDER:
        hits = [steps_to_target(spec, sd, lr, regime) for sd in SEEDS]
        if any(h is None for h in hits):
            continue
        mean = sum(hits) / len(hits)
        if best is None or mean < best[0]:
            best = (mean, lr, hits)
    if best is None:
        return None
    mean_steps, lr, hits = best
    walls = [timed(spec, sd, lr, h, regime) for sd, h in zip(SEEDS, hits)]
    return mean_steps, lr, sum(walls) / len(walls)


def main() -> None:
    torch.set_num_threads(6)
    import sys

    regime = sys.argv[1] if len(sys.argv) > 1 else "sparse"
    global TARGET
    if len(sys.argv) > 2:
        TARGET = float(sys.argv[2])
    n_p = V if regime == "sparse" else N_DENSE
    print(f"Rung 3h  D={D} target={TARGET}, 3 seeds, regime={regime.upper()}")
    if regime == "sparse":
        print(f"  V={V} N={N}: a parameter is touched every ~{V // N} steps,")
        print("  so deferral by G touches costs 8G steps of staleness.")
    else:
        print(f"  batch = all {n_p} units every step (transformer-like dX),")
        print("  so one touch == one step and deferral costs G steps.")
    print("LR selected to MINIMISE STEPS TO TARGET.\n")

    if regime == "sparse":
        specs = [("dense",), ("L", 8), ("L", 16), ("M", 16), ("M", 32),
                 ("K3", 0.1), ("K3", 0.25), ("K3", 0.5), ("K2", 0.25)]
    else:
        specs = [("dense",), ("L", 16), ("M", 16),
                 ("K3", 0.25), ("K3", 0.4), ("K3", 0.5), ("K3", 0.6)]

    hdr = (f"{'method':>10}{'cfg':>7}{'lr':>8}{'steps':>8}"
           f"{'wall s':>9}{'s/step':>10}{'vs dense':>10}")
    print(hdr)
    print("-" * len(hdr))

    base = None
    rows = []
    for spec in specs:
        res = evaluate(spec, regime)
        label = spec[0]
        cfg = "-" if len(spec) == 1 else str(spec[1])
        if res is None:
            print(f"{label:>10}{cfg:>7}{'':>8}{'not reached':>27}")
            continue
        steps, lr, wall = res
        if base is None:
            base = wall
        rows.append((label, cfg, wall / base, steps))
        print(f"{label:>10}{cfg:>7}{lr:>8.0f}{steps:>8.0f}"
              f"{wall:>9.3f}{wall / steps:>10.5f}{wall / base:>9.2f}x")

    winners = [r for r in rows if r[2] < 0.98]
    print()
    if winners:
        b = min(winners, key=lambda r: r[2])
        print(f"  FASTEST TO TARGET: {b[0]} cfg={b[1]} at {b[2]:.2f}x dense "
              f"({1 / b[2]:.2f}x speedup), {b[3]:.0f} steps")
    else:
        print("  No configuration beats dense on wall-clock to target.")


if __name__ == "__main__":
    main()
