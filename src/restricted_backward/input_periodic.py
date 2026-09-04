"""Rung 3b: Process I via periodic dense recompute + explicit age scheduling.

CATEGORY WARNING -- read before comparing this to Process O.

    Process O (F/G) ELIMINATES dense cost. The dW dense object is never
    constructed at any point, at any step.

    Process I (this module) AMORTIZES dense cost. The full dX GEMM is computed
    every `recompute_period` steps to refresh the ranking. Averaged cost is
    reduced; peak cost on a recompute step is the full dense op.

These are different categories of result. This module must never be described
as "restriction upstream of the computation" -- it is periodic dense recovery,
structurally the ReLoRA-style pattern, and the distinction was the founding
premise of this whole line of work.

Why no cheaper mechanism was used (measured, T=60, d=1024, k=5%):

    estimator        rank corr   top-k recall   dX cost
    cauchy-schwarz     +0.124          8.8%      0.05x   degenerate (scalar factorizes out)
    sketch s=8         +0.444         25.4%      0.30x   below bar
    sketch s=16        +0.571         31.3%      0.55x   clears bar, poor recall, barely cheaper
    periodic m=10      +0.984         90.8%      0.15x   dominates on all three axes

Why an explicit age term is required here but not for Process O:

    row_i(dW) = G_Y[:,i]^T X is additively separable per output channel, so
    banking the debt for row i costs exactly what computing row i costs --
    error feedback is self-financing, and anti-starvation falls out of it free.

    dX[:,j] = G_Y @ W[:,j] is a dense contraction over all of d_out for every
    j. There is no way to isolate the debt for column j without the full
    matvec. Error feedback is therefore unavailable at any useful price, and
    the anti-starvation property it provided must be supplied explicitly.
"""
from __future__ import annotations

import time as _time

import torch

from src.restricted_backward.methods import reference_forward


class InputRestrictedPeriodic:
    """Process I with amortized exact ranking and a hard wait-time bound.

    `max_age` force-selects any column starved beyond the bound, so worst-case
    wait is bounded by construction rather than emerging from selection
    dynamics. Default is 1.5x the round-robin floor (d_in / k).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        keep_fraction: float,
        recompute_period: int = 10,
        age_bounded: bool = True,
        max_age_multiplier: float = 1.5,
        dtype=torch.float32,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5
        self.d_in, self.d_out = d_in, d_out
        self.k = max(1, int(round(keep_fraction * d_in)))
        self.recompute_period = recompute_period
        self.age_bounded = age_bounded
        self.max_age = int(max_age_multiplier * d_in / self.k)
        self.name = "J" if age_bounded else "J-noage"

        self._stale_norms = torch.zeros(d_in, dtype=dtype)
        self._age = torch.zeros(d_in, dtype=torch.long)
        self._step = 0
        self.recompute_count = 0
        self.forced_count = 0
        self.max_backward_tensor_bytes = 0
        self.aux_bytes = (self._stale_norms.numel() * self._stale_norms.element_size()
                          + self._age.numel() * 8)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    def _select(self) -> torch.Tensor:
        if not self.age_bounded:
            return torch.topk(self._stale_norms, self.k, sorted=False).indices

        starved = torch.nonzero(self._age > self.max_age).flatten()
        if starved.numel() == 0:
            return torch.topk(self._stale_norms, self.k, sorted=False).indices

        # Oldest-first among the starved, capped at k.
        starved = starved[torch.argsort(self._age[starved], descending=True)][: self.k]
        self.forced_count += int(starved.numel())
        remaining = self.k - starved.numel()
        if remaining <= 0:
            return starved
        score = self._stale_norms.clone()
        score[starved] = float("-inf")
        rest = torch.topk(score, remaining, sorted=False).indices
        return torch.cat([starved, rest])

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor):
        n = X.shape[0]
        elem = G_Y.element_size()
        recomputed = (self._step % self.recompute_period) == 0

        _t0 = _time.perf_counter()
        if recomputed:
            # AMORTIZED DENSE COST: the full GEMM runs on this step.
            dX_full = G_Y @ self.W
            self._stale_norms = dX_full.norm(dim=0)
            self.recompute_count += 1
        estimator_s = _time.perf_counter() - _t0

        _s0 = _time.perf_counter()
        idx = self._select()
        selection_s = _time.perf_counter() - _s0

        _g0 = _time.perf_counter()
        if recomputed:
            dX_partial = dX_full[:, idx]          # reuse; no second GEMM
        else:
            dX_partial = G_Y @ self.W[:, idx]     # restricted GEMM
        gemm_s = _time.perf_counter() - _g0

        self._age += 1
        self._age[idx] = 0
        self._step += 1

        self.max_backward_tensor_bytes = max(
            self.max_backward_tensor_bytes,
            (dX_full.numel() if recomputed else dX_partial.numel()) * elem,
        )

        executed = (
            2.0 * n * self.d_out * self.d_in
            if recomputed
            else 2.0 * n * self.d_out * len(idx)
        )
        return (idx, dX_partial), executed, estimator_s, selection_s, gemm_s

    def exact_dX(self, G_Y: torch.Tensor) -> torch.Tensor:
        """Oracle for error measurement only. Never in a timed path."""
        return G_Y @ self.W

    def apply_dense_dW(self, dW: torch.Tensor, lr: float) -> None:
        """Process O stays dense at this rung, isolating Process I."""
        self.W -= lr * dW

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()
