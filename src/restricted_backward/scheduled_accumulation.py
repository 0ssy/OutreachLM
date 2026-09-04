"""Method K -- Process I via scheduled accumulation. Fixes the rung 3b blocker.

The rung 3b finding was: dX[:,j] = G_Y @ W[:,j] is a dense contraction over
d_out, so per-column debt cannot be isolated without the full matvec, so error
feedback is unavailable, so the 0.959 energy floor becomes permanent bias.

That reasoning has a hole. The debt is not bankable *per column* -- but dX is
LINEAR in G_Y, and G_Y is shared across all columns and already materialized.
So bank the upstream gradient, keyed by a deterministic visit schedule:

    partition [d_in] into G = ceil(1/keep) groups
    at step t, visit group (t mod G)
    S_g = sum of G_Y over the steps since group g was last visited
    dX̂[:, group_g] = S_g @ W[:, group_g]        <- exact accumulated debt

One restricted GEMM, identical cost to rung 3b, and the delivered gradient is
the EXACT sum of every G_Y that column missed. Nothing is discarded.

What this changes relative to rung 3b:

    rung 3b   discarded 95% of dX energy permanently   -> bias floor 0.959
    Method K  defers 95%, repays it exactly on visit   -> only W-staleness error

Why no selector is needed: the schedule is deterministic round-robin, so there
is no ranking to estimate. This sidesteps the proxy problem entirely -- the
Cauchy-Schwarz degeneracy and the sketch's 31% recall become irrelevant rather
than needing to be solved.

CATEGORY: this is genuine elimination, not amortization. The dense dX object is
never constructed at any step. Unlike input_periodic.py, no periodic dense
recompute occurs.

Residual accounting:
    T   = running sum of all G_Y            (N x d_out, float64)
    T_g = snapshot of T at group g's last visit
    S_g = T - T_g                            exact, O(N*d_out) per step
Memory: G snapshots of (N x d_out). At G=20, N=32, d_out=1024 that is 5.2 MB
in float64 against a 4 MB weight matrix -- the one real cost of this method.

Residual error source: G_Y(s) is applied against W(t) rather than W(s), so the
error is W-staleness over at most G steps. This is a delayed-update error, the
same class as gradient accumulation, and is bounded by how fast W moves.

SCOPE LIMIT -- MEASURED IN RUNG 3E, LOAD-BEARING FOR ANY REUSE OF THIS CLASS
    K is valid only when EVERY unit of the restricted axis is present at EVERY
    step (full-batch access). It has two independent failure modes under any
    partial or irregular access, and only the first is repairable:

    1. MISROUTING. The bank above is shaped (N x d_out) -- keyed by BATCH SLOT,
       not by parameter identity. Under a partial batch, slot i holds a
       different parameter row each step, so debt banked for one row is repaid
       to another. Measured: with density, periodicity, coverage and aliasing
       all held perfect and ONLY slot->row identity permuted, K goes
       0.0000 -> 0.0462 while dense and Method L are unmoved.
       Re-keying the bank to (V x d_out) fixes this (0.0462 -> 0.0000) at a
       memory cost that scales with the number of distinct parameters rather
       than batch size (4.2 MB vs 0.3 MB at V=512, N=32).

    2. MIS-SCHEDULING. Not fixed by re-keying. Repayment advances one group per
       STEP, but a parameter's gradient arrives on its own TOUCH cadence. When
       those differ, hot parameters accumulate many steps of debt and are repaid
       on the same slow cadence as cold ones. Measured with the row-keyed bank,
       W frozen:

           pattern    K (slot)   K (row-keyed)   L(s=16)
           full        0.0000      0.0000         0.0005
           cyclic      0.1514      0.0093         0.0005
           uniform     0.1631      0.0516         0.0007
           zipf        0.1763      1.18e29        0.0534   <- diverges

    For ACTIVATION gradients neither fix is even definable: activations have no
    persistent identity across steps, so there is nothing to key a bank by, and
    an upstream layer needs dX within the step rather than deferred. K is
    therefore not a general Process I mechanism -- see PROTOCOL_TWO_SIDED.md,
    rung 3e.
"""
from __future__ import annotations

import time as _time

import torch

from src.restricted_backward.methods import reference_forward


class ScheduledAccumulation:
    """Process I with deterministic round-robin visits and exact debt repayment."""

    name = "K"

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        keep_fraction: float,
        dtype=torch.float32,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5
        self.d_in, self.d_out = d_in, d_out
        self.k = max(1, int(round(keep_fraction * d_in)))
        self.n_groups = max(1, -(-d_in // self.k))          # ceil(d_in / k)

        # Fixed partition of input channels into visit groups.
        perm = torch.randperm(d_in, generator=g)
        self.groups = [perm[i::self.n_groups] for i in range(self.n_groups)]

        self._running_total: torch.Tensor | None = None      # T, float64
        self._snapshots: list[torch.Tensor | None] = [None] * self.n_groups
        self._step = 0
        self.max_backward_tensor_bytes = 0
        self.aux_bytes = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    @property
    def visit_period(self) -> int:
        """Deterministic bound: every input channel is visited every G steps."""
        return self.n_groups

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor):
        n = G_Y.shape[0]
        elem = G_Y.element_size()

        _f0 = _time.perf_counter()
        if self._running_total is None:
            self._running_total = torch.zeros(n, self.d_out, dtype=torch.float64)
            self.aux_bytes = (
                self._running_total.numel() * 8 * (1 + self.n_groups)
            )
        self._running_total += G_Y.double()

        gi = self._step % self.n_groups
        idx = self.groups[gi]
        snap = self._snapshots[gi]
        # S_g = exact sum of every G_Y this group missed since its last visit.
        S = self._running_total if snap is None else self._running_total - snap
        self._snapshots[gi] = self._running_total.clone()
        feedback_s = _time.perf_counter() - _f0

        _g0 = _time.perf_counter()
        dX_partial = (S.to(G_Y.dtype)) @ self.W[:, idx]      # (N, |group|)
        gemm_s = _time.perf_counter() - _g0

        self._step += 1
        self.max_backward_tensor_bytes = max(
            self.max_backward_tensor_bytes, dX_partial.numel() * elem
        )
        executed = 2.0 * n * self.d_out * len(idx)
        return (idx, dX_partial), executed, 0.0, feedback_s, gemm_s

    def exact_dX(self, G_Y: torch.Tensor) -> torch.Tensor:
        """Oracle for error measurement only. Never in a timed path."""
        return G_Y @ self.W

    def apply_dense_dW(self, dW: torch.Tensor, lr: float) -> None:
        self.W -= lr * dW

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()
