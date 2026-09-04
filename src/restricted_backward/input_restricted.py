"""Rung 3: input-side restriction in isolation (Process I only).

Per PROTOCOL_TWO_SIDED.md. dW stays DENSE here so that any effect observed is
attributable to Process I alone. Rung 4 combines the two.

    dX̂[:,T] = G_Y W[:,T]     exact on selected input channels
    dX̂[:,T̄] = 0

Selection uses ||G_Y||_F * ||W[:,j]||_2 (Cauchy-Schwarz upper bound on
||dX[:,j]||), with W's column norms maintained incrementally.
"""
from __future__ import annotations

import time as _time

import torch

from src.restricted_backward.methods import reference_backward, reference_forward


class InputRestricted:
    """Process I in isolation. name: H (no EF) / I (with EF)."""

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
        self.name = "I" if error_feedback else "H"

        # Process I residual lives on dX (N x d_in), allocated lazily.
        self.residual = None
        self.aux_bytes = 0
        self.max_backward_tensor_bytes = 0
        # Incrementally maintained column norms of W: O(k_out*d_in) upkeep.
        self._w_col_norms = self.W.norm(dim=0)
        self.last_proxy_rank_corr: float | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return reference_forward(X, self.W)

    def _select(self, G_Y: torch.Tensor) -> torch.Tensor:
        k = max(1, int(round(self.keep_fraction * self.d_in)))
        score = float(G_Y.norm()) * self._w_col_norms
        if self.residual is not None:
            score = score + self.residual.norm(dim=0)
        _, idx = torch.topk(score, k, sorted=False)
        return idx

    def backward(self, G_Y: torch.Tensor, X: torch.Tensor, *, measure_proxy: bool = False):
        n = X.shape[0]
        elem = G_Y.element_size()

        # dW stays fully dense: Process O is not restricted at this rung.
        _b0 = _time.perf_counter()
        dW = G_Y.T @ X
        dw_s = _time.perf_counter() - _b0

        feedback_s = 0.0
        _s0 = _time.perf_counter()
        idx = self._select(G_Y)
        selection_s = _time.perf_counter() - _s0

        # Restricted GEMM: (N,d_out) @ (d_out,k) -> (N,k)
        _x0 = _time.perf_counter()
        dX_partial = G_Y @ self.W[:, idx]
        dx_s = _time.perf_counter() - _x0

        if self.error_feedback:
            _f0 = _time.perf_counter()
            if self.residual is None:
                self.residual = torch.zeros(n, self.d_in, dtype=G_Y.dtype)
                self.aux_bytes = self.residual.numel() * elem
            mask = torch.zeros(self.d_in, dtype=torch.bool)
            mask[idx] = True
            # Scatter the exact selected columns, keep the rest as debt.
            full = self.residual.clone()
            full[:, idx] = dX_partial
            self.residual = full * (~mask)
            feedback_s = _time.perf_counter() - _f0

        self.max_backward_tensor_bytes = max(
            dW.numel(), dX_partial.numel()
        ) * elem

        executed_dx = 2.0 * n * self.d_out * len(idx)
        executed_dw = 2.0 * n * self.d_in * self.d_out

        if measure_proxy:
            exact_dX = G_Y @ self.W
            true_order = torch.argsort(exact_dX.norm(dim=0), descending=True)
            proxy_order = torch.argsort(
                float(G_Y.norm()) * self._w_col_norms, descending=True
            )
            rt = torch.empty(self.d_in); rt[true_order] = torch.arange(self.d_in, dtype=torch.float)
            rp = torch.empty(self.d_in); rp[proxy_order] = torch.arange(self.d_in, dtype=torch.float)
            rt = rt - rt.mean(); rp = rp - rp.mean()
            denom = float(rt.norm() * rp.norm())
            self.last_proxy_rank_corr = float((rt @ rp) / denom) if denom > 0 else 0.0

        return (
            dW,
            (idx, dX_partial),
            executed_dw + executed_dx,
            selection_s,
            feedback_s,
            dw_s + dx_s,
        )

    def exact_dX(self, G_Y: torch.Tensor) -> torch.Tensor:
        """Oracle for error measurement only. Never in the timed path."""
        return G_Y @ self.W

    def apply(self, dW: torch.Tensor, lr: float) -> None:
        self.W -= lr * dW
        self._w_col_norms = self.W.norm(dim=0)

    def param_bytes(self) -> int:
        return self.W.numel() * self.W.element_size()
