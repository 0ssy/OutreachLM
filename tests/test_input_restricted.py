"""Rung 3 validation: Process I in isolation against exact dense dX.

Protocol rule: input-side restriction must be established on its own before
rung 4 combines it with output-side restriction.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from src.restricted_backward.input_restricted import InputRestricted


def test_selected_columns_are_exact_against_dense_reference() -> None:
    """dX[:,j] = G_Y W[:,j], so column selection is exact on selected."""
    torch.manual_seed(0)
    d = 64
    layer = InputRestricted(d, d, keep_fraction=0.25, dtype=torch.float64)
    X = torch.randn(8, d, dtype=torch.float64)
    G_Y = torch.randn(8, d, dtype=torch.float64)

    _, (idx, dX_partial), _, _, _, _ = layer.backward(G_Y, X)
    exact = layer.exact_dX(G_Y)
    assert torch.allclose(dX_partial, exact[:, idx], atol=1e-12)


def test_dw_stays_dense_at_this_rung() -> None:
    """Process O must be unrestricted here, or the isolation is broken."""
    torch.manual_seed(1)
    d = 32
    layer = InputRestricted(d, d, keep_fraction=0.10, dtype=torch.float64)
    X = torch.randn(8, d, dtype=torch.float64)
    G_Y = torch.randn(8, d, dtype=torch.float64)

    dW, _, _, _, _, _ = layer.backward(G_Y, X)
    assert torch.allclose(dW, G_Y.T @ X, atol=1e-12)
    assert int((dW == 0).sum()) == 0


def test_unselected_columns_receive_exactly_zero() -> None:
    torch.manual_seed(2)
    d = 128
    layer = InputRestricted(d, d, keep_fraction=0.10)
    X = torch.randn(8, d)
    G_Y = torch.randn(8, d)

    _, (idx, dX_partial), _, _, _, _ = layer.backward(G_Y, X)
    reconstructed = torch.zeros(8, d)
    reconstructed[:, idx] = dX_partial
    unselected = torch.ones(d, dtype=torch.bool)
    unselected[idx] = False
    assert float(reconstructed[:, unselected].abs().max()) == 0.0


def test_dx_gemm_never_materializes_full_dx() -> None:
    """The restricted dX GEMM must produce (N,k), not (N,d_in)."""
    torch.manual_seed(3)
    d, n = 512, 32
    layer = InputRestricted(d, d, keep_fraction=0.05)
    X = torch.randn(n, d)
    G_Y = torch.randn(n, d)
    _, (idx, dX_partial), _, _, _, _ = layer.backward(G_Y, X)

    k = max(1, round(0.05 * d))
    assert dX_partial.shape == (n, k)
    assert dX_partial.numel() < n * d


def test_executed_flops_scale_with_k_on_the_dx_path() -> None:
    torch.manual_seed(4)
    d, n = 256, 16
    for keep in (0.01, 0.10, 0.50):
        layer = InputRestricted(d, d, keep_fraction=keep)
        X = torch.randn(n, d)
        G_Y = torch.randn(n, d)
        _, (idx, _), executed, _, _, _ = layer.backward(G_Y, X)
        expected = 2.0 * n * d * d + 2.0 * n * d * len(idx)
        assert executed == pytest.approx(expected)


def test_selection_proxy_is_computable_before_the_gemm() -> None:
    """Score must depend only on G_Y and cached W column norms."""
    torch.manual_seed(5)
    d, n = 256, 16
    layer = InputRestricted(d, d, keep_fraction=0.10)
    G_Y = torch.randn(n, d)
    idx = layer._select(G_Y)
    assert len(idx) == max(1, round(0.10 * d))
    assert layer._w_col_norms.shape == (d,)


def test_proxy_rank_correlation_is_measured_not_assumed() -> None:
    """The Cauchy-Schwarz proxy is an approximation; report its quality."""
    torch.manual_seed(6)
    d, n = 256, 16
    layer = InputRestricted(d, d, keep_fraction=0.10)
    X = torch.randn(n, d)
    G_Y = torch.randn(n, d)
    layer.backward(G_Y, X, measure_proxy=True)
    assert layer.last_proxy_rank_corr is not None
    assert -1.0 <= layer.last_proxy_rank_corr <= 1.0


def test_error_feedback_residual_is_dx_shaped() -> None:
    """Process I residual lives on dX (N x d_in), not on W."""
    torch.manual_seed(7)
    d, n = 256, 16
    layer = InputRestricted(d, d, keep_fraction=0.10, error_feedback=True)
    X = torch.randn(n, d)
    G_Y = torch.randn(n, d)
    layer.backward(G_Y, X)
    assert layer.residual.shape == (n, d)
    assert layer.aux_bytes == n * d * 4
