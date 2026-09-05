"""Design constraints any restriction mechanism must satisfy BEFORE implementation.

These exist because each was learned by shipping a mechanism that violated it
and then diagnosing the wreckage. They are cheap, mechanical predicates: run
them against a candidate operator first, not after a divergent training run.

CONSTRAINT 1 -- NON-EXPANSIVE RESIDUAL (required for error feedback)
    Error feedback converges only if the residual operator R(v) = v - C(v) is
    non-expansive, i.e. ||R(v)|| <= ||v|| for all v. This is the standard
    contraction assumption in the compressed-SGD-with-error-feedback
    convergence literature, and it is the one-line check that would have caught
    Method M's first version before implementation rather than after it
    diverged to 3.2e6 accumulated error.

    Top-k and orthogonal projection satisfy it (norm exactly 1).
    A Gaussian sketch scaled for UNBIASEDNESS does not: E[P P^T] = I forces the
    operator to overshoot, so ||I - P P^T|| > 1 whenever s < N.

    Corollary, worth stating because it is not obvious: UNBIASEDNESS AND
    CONTRACTIVITY ARE IN CONFLICT. A compressor cannot generally be both, so a
    mechanism must choose -- unbiased and used WITHOUT feedback (Method L), or
    biased-but-contractive and used WITH feedback (Methods G, M).

CONSTRAINT 2 -- BANKING REQUIRES A PERSISTENT KEY
    A residual bank is only meaningful if the thing it is keyed by still means
    the same thing next step. Method K banked by batch SLOT, which names a
    different parameter each step under any partial batch, so debt was
    misrouted. Activations have no persistent identity at all, so no feedback
    mechanism is definable for them.

CONSTRAINT 3 -- SCHEDULES MUST ADVANCE IN THE SAME CLOCK AS ARRIVALS
    If repayment advances per STEP but gradients arrive per TOUCH, the two
    desynchronise for any parameter not touched every step.

Nothing here is a substitute for measurement; these only rule mechanisms OUT
cheaply. Passing all three does not imply a mechanism is economical -- see
rung 3g, where mechanisms satisfying all three still lost on wall-clock to
target.
"""
from __future__ import annotations

import torch


def residual_expansion_factor(
    compressor,
    n: int,
    d: int,
    *,
    trials: int = 64,
    seed: int = 0,
) -> float:
    """Max observed ||v - C(v)|| / ||v||. Must be <= 1 for error feedback.

    `compressor` maps an (n, d) tensor to its compressed approximation.
    Returns the worst ratio over random probes; > 1 means banking the residual
    will amplify it and the mechanism must not be used with error feedback.
    """
    g = torch.Generator().manual_seed(seed)
    worst = 0.0
    for _ in range(trials):
        v = torch.randn(n, d, generator=g)
        r = v - compressor(v)
        worst = max(worst, float(r.norm() / v.norm()))
    return worst


def admits_error_feedback(compressor, n: int, d: int, **kw) -> bool:
    """Constraint 1 as a boolean. Check this before writing the mechanism."""
    return residual_expansion_factor(compressor, n, d, **kw) <= 1.0 + 1e-5


def topk_rows(keep: float):
    """Reference contractive compressor: keeps the largest-norm columns."""

    def _c(v: torch.Tensor) -> torch.Tensor:
        k = max(1, int(round(keep * v.shape[1])))
        idx = torch.topk(v.norm(dim=0), k, sorted=False).indices
        out = torch.zeros_like(v)
        out[:, idx] = v[:, idx]
        return out

    return _c


def gaussian_sketch(s: int, seed: int = 0):
    """Reference UNBIASED compressor. Deliberately NOT contractive."""
    g = torch.Generator().manual_seed(seed)

    def _c(v: torch.Tensor) -> torch.Tensor:
        P = torch.randn(v.shape[0], s, generator=g) / s**0.5
        return P @ (P.T @ v)

    return _c


def orthonormal_sketch(s: int, seed: int = 0):
    """Reference contractive sketch: biased per step, safe under feedback."""
    g = torch.Generator().manual_seed(seed)

    def _c(v: torch.Tensor) -> torch.Tensor:
        Q, _ = torch.linalg.qr(
            torch.randn(v.shape[0], s, generator=g)
        )
        return Q @ (Q.T @ v)

    return _c
