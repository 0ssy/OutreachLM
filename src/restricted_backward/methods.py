"""Methods A-E for the CPU-native restricted backward experiment.

Every backward is implemented from the Phase 0 hand-derived expressions:

    dL/dW = G_Y^T X
    dL/dX = G_Y   W

No method calls autograd for the weight gradient; autograd is used only as an
independent oracle in the validation tests.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Phase 0 reference (the ground truth every method is judged against)
# ---------------------------------------------------------------------------
def reference_forward(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return X @ W.T


def reference_backward(
    G_Y: torch.Tensor, X: torch.Tensor, W: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-derived dense gradients. dW = G_Y^T X, dX = G_Y W."""
    return G_Y.T @ X, G_Y @ W


def dense_flops(n: int, d_in: int, d_out: int) -> dict[str, float]:
    """Multiply-accumulate counted as 2 FLOPs."""
    fwd = 2.0 * n * d_in * d_out
    dw = 2.0 * n * d_in * d_out
    dx = 2.0 * n * d_in * d_out
    return {"forward": fwd, "dW": dw, "dX": dx, "total": fwd + dw + dx}


# ---------------------------------------------------------------------------
# Method A: dense baseline
# ---------------------------------------------------------------------------
class MethodA:
    """Ordinary dense layer. The only method trusted by default."""

    name = "A"

    def __init__(self, d_in: int, d_out: int, *, dtype=torch.float32, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W = (torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5)
        self.d_in, self.d_out = d_in, d_out
        self.aux_bytes = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor):
        """Returns (dW_applied, dX, exact_dW, executed_flops_dW, selection_s, feedback_s)."""
        dW, dX = reference_backward(G_Y, X, self.W)
        n = X.shape[0]
        return dW, dX, dW, 2.0 * n * self.d_in * self.d_out, 0.0, 0.0

    def apply(self, dW: torch.Tensor, lr: float) -> None:
        self.W -= lr * dW

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()


# ---------------------------------------------------------------------------
# Method B: low-rank backward  (W_eff = W_0 + B A, W_0 frozen)
# ---------------------------------------------------------------------------
class MethodB:
    """Low-rank adapter. The (d_out x d_in) dense gradient is never formed.

    Hand-derived, with Y = X (W_0 + B A)^T:

        Y      = X W_0^T + (X A^T) B^T
        dL/dB  = G_Y^T (X A^T)          -> (d_out, r)
        dL/dA  = (G_Y B)^T X            -> (r, d_in)
        dL/dX  = G_Y W_0 + (G_Y B) A

    Every intermediate is (N, r), (d_out, r) or (r, d_in). Nothing of shape
    (d_out, d_in) is allocated in the backward path, which is asserted by the
    auxiliary-memory check rather than assumed from reading the code.
    """

    name = "B"

    def __init__(self, d_in: int, d_out: int, *, rank: int, dtype=torch.float32, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W0 = (torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5)
        # Standard LoRA-style init: A random, B zero, so W_eff == W_0 at t=0.
        self.A = (torch.randn(rank, d_in, generator=g, dtype=dtype) / d_in**0.5)
        self.B = torch.zeros(d_out, rank, dtype=dtype)
        self.rank = rank
        self.d_in, self.d_out = d_in, d_out
        self._XA = None
        self.aux_bytes = (self.A.numel() + self.B.numel()) * self.A.element_size()

    @property
    def W(self) -> torch.Tensor:
        """Materialized effective weight. Test/oracle use only, never in backward."""
        return self.W0 + self.B @ self.A

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self._XA = X @ self.A.T                       # (N, r)
        return X @ self.W0.T + self._XA @ self.B.T

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor):
        n = X.shape[0]
        dB = G_Y.T @ self._XA                          # (d_out, r)
        GB = G_Y @ self.B                              # (N, r)
        dA = GB.T @ X                                  # (r, d_in)
        dX = G_Y @ self.W0 + GB @ self.A               # (N, d_in)

        flops = (
            2.0 * n * self.d_out * self.rank           # dB
            + 2.0 * n * self.d_out * self.rank         # GB
            + 2.0 * n * self.rank * self.d_in          # dA
        )
        return (dB, dA), dX, None, flops, 0.0, 0.0

    def apply(self, grads, lr: float) -> None:
        dB, dA = grads
        self.B -= lr * dB
        self.A -= lr * dA

    def param_bytes(self) -> int:
        return (self.A.numel() + self.B.numel()) * self.A.element_size()


# ---------------------------------------------------------------------------
# Methods C / D / E: sparsified weight gradient
# ---------------------------------------------------------------------------
@dataclass
class SparseConfig:
    keep_fraction: float
    error_feedback: bool = False
    block_size: int | None = None      # None => unstructured (C/D); int => blocked (E)


class SparseMethod:
    """C = top-k, D = top-k + error feedback, E = block top-k + error feedback.

    Only dL/dW is sparsified. dL/dX stays dense, because it is what reaches
    upstream layers; sparsifying it would be a different experiment (and would
    break the 33.3% single-layer saving bound derived in Phase 0).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        config: SparseConfig,
        dtype=torch.float32,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.W = (torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5)
        self.cfg = config
        self.d_in, self.d_out = d_in, d_out
        self.residual = (
            torch.zeros(d_out, d_in, dtype=dtype) if config.error_feedback else None
        )
        self.aux_bytes = (
            self.residual.numel() * self.residual.element_size()
            if self.residual is not None
            else 0
        )
        if config.block_size is not None:
            self.name = "E"
        elif config.error_feedback:
            self.name = "D"
        else:
            self.name = "C"

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    # -- selection ---------------------------------------------------------
    def _select_unstructured(self, g: torch.Tensor) -> torch.Tensor:
        numel = g.numel()
        k = max(1, int(round(self.cfg.keep_fraction * numel)))
        flat = g.reshape(-1)
        _, idx = torch.topk(flat.abs(), k, sorted=False)
        out = torch.zeros_like(flat)
        out[idx] = flat[idx]
        return out.view_as(g)

    def _select_blocked(self, g: torch.Tensor) -> torch.Tensor:
        bs = int(self.cfg.block_size)
        d_out, d_in = g.shape
        if d_out % bs or d_in % bs:
            raise ValueError("block_size must divide both weight dimensions")
        blocks = g.reshape(d_out // bs, bs, d_in // bs, bs).permute(0, 2, 1, 3)
        norms = blocks.reshape(blocks.shape[0], blocks.shape[1], -1).norm(dim=-1)
        n_blocks = norms.numel()
        k = max(1, int(round(self.cfg.keep_fraction * n_blocks)))
        _, idx = torch.topk(norms.reshape(-1), k, sorted=False)
        mask = torch.zeros(n_blocks, dtype=torch.bool)
        mask[idx] = True
        mask = mask.view(norms.shape)[:, :, None, None]
        kept = blocks * mask
        return kept.permute(0, 2, 1, 3).reshape(d_out, d_in)

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor):
        n = X.shape[0]
        exact_dW, dX = reference_backward(G_Y, X, self.W)

        feedback_s = 0.0
        if self.residual is not None:
            _f0 = _time.perf_counter()
            candidate = exact_dW + self.residual
            feedback_s = _time.perf_counter() - _f0
        else:
            candidate = exact_dW

        _s0 = _time.perf_counter()
        if self.cfg.block_size is not None:
            sparse = self._select_blocked(candidate)
        else:
            sparse = self._select_unstructured(candidate)
        selection_s = _time.perf_counter() - _s0

        if self.residual is not None:
            _f1 = _time.perf_counter()
            self.residual = candidate - sparse
            feedback_s += _time.perf_counter() - _f1

        # Executed FLOPs for dW: only the surviving entries represent useful
        # multiply-accumulate work in an idealized sparse kernel.
        nnz = int((sparse != 0).sum())
        executed = 2.0 * n * nnz
        return sparse, dX, exact_dW, executed, selection_s, feedback_s

    def apply(self, dW: torch.Tensor, lr: float) -> None:
        self.W -= lr * dW

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()
