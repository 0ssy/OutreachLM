"""Method F/G: TRUE restricted backward (meProp-style).

The distinction this module exists to test, per the critique of C/D/E:

    C/D/E:  dW = G_Y^T X   (full 4096x4096 GEMM)  -> TopK(dW)
            The dense object is built and paid for, then compressed.
            Selection can only ADD cost. Memory footprint == dense.

    F/G:    idx = TopK(columns of G_Y)            (scan N x d_out = 131k)
            dW_partial = G_Y[:, idx]^T X          (GEMM is k x d_in)
            The dense object is NEVER constructed. The GEMM itself shrinks.

Selection happens on the UPSTREAM gradient (N x d_out = 131,072 elements)
rather than on the materialized weight gradient (d_out x d_in = 16,777,216
elements) -- 128x less data to scan, and it happens *before* the matmul.

Exactness property worth noting: because row i of dW equals G_Y[:,i]^T X,
selecting columns of G_Y yields EXACTLY the corresponding rows of the true
dW. The approximation is solely that unselected rows receive no update --
there is no within-row error, unlike TopK applied to dW.
"""
from __future__ import annotations

import time as _time

import torch

from src.restricted_backward.methods import reference_backward, reference_forward


class MethodF:
    """Restricted backward: select on G_Y, then run a smaller GEMM.

    dX is kept dense (it propagates upstream), so this remains directly
    comparable to C/D/E under the Phase 0 one-third saving bound.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        keep_fraction: float,
        error_feedback: bool = False,
        dtype=torch.float32,
        seed: int = 0,
    ):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_out, d_in, generator=g, dtype=dtype) / d_in**0.5
        self.d_in, self.d_out = d_in, d_out
        self.keep_fraction = keep_fraction
        self.error_feedback = error_feedback
        # Error feedback in the restricted setting lives on the UPSTREAM
        # gradient (N x d_out), not on dW (d_out x d_in). This is the whole
        # point: the residual buffer is 128x smaller than D/E's.
        self.residual = None
        self.name = "G" if error_feedback else "F"
        self.max_backward_tensor_bytes = 0
        self.aux_bytes = 0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor):
        n = X.shape[0]
        elem = G_Y.element_size()

        feedback_s = 0.0
        candidate = G_Y
        if self.error_feedback:
            _f0 = _time.perf_counter()
            if self.residual is None:
                self.residual = torch.zeros_like(G_Y)
                self.aux_bytes = self.residual.numel() * elem
            candidate = G_Y + self.residual
            feedback_s = _time.perf_counter() - _f0

        # --- selection: scan the UPSTREAM gradient, before any GEMM --------
        _s0 = _time.perf_counter()
        k = max(1, int(round(self.keep_fraction * self.d_out)))
        col_norms = candidate.norm(dim=0)               # (d_out,)
        _, idx = torch.topk(col_norms, k, sorted=False)
        selection_s = _time.perf_counter() - _s0

        if self.error_feedback:
            _f1 = _time.perf_counter()
            mask = torch.zeros(self.d_out, dtype=torch.bool)
            mask[idx] = True
            self.residual = candidate * (~mask)
            feedback_s += _time.perf_counter() - _f1

        # --- restricted GEMM: (N,k)^T @ (N,d_in) -> (k,d_in) ---------------
        _b0 = _time.perf_counter()
        Gs = candidate[:, idx]                          # (N, k)
        dW_partial = Gs.T @ X                           # (k, d_in)
        dX = G_Y @ self.W                               # dense, unrestricted
        restricted_backward_s = _time.perf_counter() - _b0

        self.max_backward_tensor_bytes = max(
            Gs.numel(), dW_partial.numel(), dX.numel()
        ) * elem

        executed = 2.0 * n * k * self.d_in              # the dW GEMM only
        return (
            (idx, dW_partial),
            dX,
            executed,
            selection_s,
            feedback_s,
            restricted_backward_s,
        )

    def exact_dW(self, G_Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Oracle for error measurement only. Never called in the timed path."""
        return reference_backward(G_Y, X, self.W)[0]

    def apply(self, grads, lr: float) -> None:
        idx, dW_partial = grads
        self.W[idx] -= lr * dW_partial

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()
