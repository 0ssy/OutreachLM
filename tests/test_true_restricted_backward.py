"""Verify Method F/G is a TRUE restricted backward, not GEMM-then-select."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from src.restricted_backward.true_restricted import MethodF


def test_restricted_dw_is_exact_on_selected_rows() -> None:
    """Row i of dW == G_Y[:,i]^T X, so column-selecting G_Y gives those rows
    EXACTLY. Unlike TopK(dW), there is no within-row approximation."""
    torch.manual_seed(0)
    d = 64
    layer = MethodF(d, d, keep_fraction=0.25, dtype=torch.float64)
    X = torch.randn(8, d, dtype=torch.float64)
    G_Y = torch.randn(8, d, dtype=torch.float64)

    (idx, dW_partial), _, _, _, _, _ = layer.backward(G_Y, X)
    exact = layer.exact_dW(G_Y, X)

    assert torch.allclose(dW_partial, exact[idx], atol=1e-12)


def test_never_materializes_dense_gradient() -> None:
    """The memory signature that separates F from C/D/E.

    C/D/E allocate d_out x d_in for the gradient. F must not.
    """
    torch.manual_seed(1)
    d = 512
    layer = MethodF(d, d, keep_fraction=0.02)
    X = torch.randn(32, d)
    G_Y = torch.randn(32, d)

    layer.backward(G_Y, X)

    dense_bytes = d * d * 4
    assert layer.max_backward_tensor_bytes < dense_bytes
    # dX (N x d_in) is the largest legitimate intermediate.
    assert layer.max_backward_tensor_bytes == 32 * d * 4


def test_error_feedback_residual_is_upstream_sized_not_weight_sized() -> None:
    """G's residual lives on G_Y (N x d_out), not dW (d_out x d_in)."""
    torch.manual_seed(2)
    d = 512
    n = 32
    layer = MethodF(d, d, keep_fraction=0.05, error_feedback=True)
    X = torch.randn(n, d)
    G_Y = torch.randn(n, d)
    layer.backward(G_Y, X)

    assert layer.residual.shape == (n, d)
    assert layer.aux_bytes == n * d * 4
    assert layer.aux_bytes < d * d * 4      # far smaller than D/E's buffer


def test_executed_flops_scale_with_k_not_d_out() -> None:
    torch.manual_seed(3)
    d, n = 256, 16
    for keep in (0.01, 0.10, 0.50):
        layer = MethodF(d, d, keep_fraction=keep)
        X = torch.randn(n, d)
        G_Y = torch.randn(n, d)
        _, _, executed, _, _, _ = layer.backward(G_Y, X)
        k = max(1, round(keep * d))
        assert executed == pytest.approx(2.0 * n * k * d)


def test_selection_scans_upstream_not_weight_gradient() -> None:
    """Selection input must be N x d_out, not d_out x d_in."""
    torch.manual_seed(4)
    d, n = 1024, 8
    layer = MethodF(d, d, keep_fraction=0.05)
    X = torch.randn(n, d)
    G_Y = torch.randn(n, d)
    _, _, _, sel_s, _, _ = layer.backward(G_Y, X)

    # Scanning n*d elements must be far cheaper than scanning d*d.
    baseline = torch.randn(d, d)
    import time
    t0 = time.perf_counter()
    torch.topk(baseline.reshape(-1).abs(), max(1, d * d // 20), sorted=False)
    dense_scan_s = time.perf_counter() - t0

    assert sel_s < dense_scan_s


def test_only_selected_rows_change() -> None:
    torch.manual_seed(5)
    d = 128
    layer = MethodF(d, d, keep_fraction=0.10)
    X = torch.randn(8, d)
    G_Y = torch.randn(8, d)
    before = layer.W.clone()

    (idx, dW_partial), _, _, _, _, _ = layer.backward(G_Y, X)
    layer.apply((idx, dW_partial), lr=0.1)

    changed = (layer.W != before).any(dim=1)
    assert int(changed.sum()) == len(idx)
    assert set(torch.nonzero(changed).flatten().tolist()) == set(idx.tolist())
