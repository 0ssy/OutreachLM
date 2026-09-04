"""Rung 3b validation, including the amortization-vs-elimination distinction."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from src.restricted_backward.input_periodic import InputRestrictedPeriodic


def test_selected_columns_are_exact() -> None:
    torch.manual_seed(0)
    d = 64
    layer = InputRestrictedPeriodic(d, d, keep_fraction=0.25, recompute_period=5,
                                    dtype=torch.float64)
    X = torch.randn(8, d, dtype=torch.float64)
    G_Y = torch.randn(8, d, dtype=torch.float64)
    (idx, dXp), _, _, _, _ = layer.backward(G_Y, X)
    assert torch.allclose(dXp, layer.exact_dX(G_Y)[:, idx], atol=1e-12)


def test_recompute_step_materializes_dense_but_interim_steps_do_not() -> None:
    """The defining property of AMORTIZATION rather than elimination.

    Process O never allocates the dense object. This does, periodically --
    and the test asserts that difference explicitly so it cannot be quietly
    described as elimination later.
    """
    torch.manual_seed(1)
    d, n = 256, 16
    layer = InputRestrictedPeriodic(d, d, keep_fraction=0.05, recompute_period=4)
    X = torch.randn(n, d)
    dense_bytes = n * d * 4

    peaks = []
    for _ in range(8):
        layer.max_backward_tensor_bytes = 0
        layer.backward(torch.randn(n, d), X)
        peaks.append(layer.max_backward_tensor_bytes)

    assert peaks[0] == dense_bytes          # recompute step: full dX allocated
    assert peaks[1] < dense_bytes           # interim step: only (n, k)
    assert layer.recompute_count == 2       # steps 0 and 4 in an 8-step run


def test_executed_flops_alternate_between_dense_and_restricted() -> None:
    torch.manual_seed(2)
    d, n = 128, 8
    layer = InputRestrictedPeriodic(d, d, keep_fraction=0.10, recompute_period=3)
    X = torch.randn(n, d)
    flops = [layer.backward(torch.randn(n, d), X)[1] for _ in range(6)]
    dense = 2.0 * n * d * d
    assert flops[0] == pytest.approx(dense)
    assert flops[1] < dense
    assert flops[3] == pytest.approx(dense)


def test_age_bound_guarantees_no_column_starves() -> None:
    """The property error feedback supplied free for Process O, here explicit."""
    torch.manual_seed(3)
    d, n = 256, 16
    layer = InputRestrictedPeriodic(d, d, keep_fraction=0.05, recompute_period=10,
                                    age_bounded=True)
    X = torch.randn(n, d)
    for _ in range(200):
        layer.backward(torch.randn(n, d), X)
    assert int(layer._age.max()) <= layer.max_age + 1


def test_without_age_bound_columns_do_starve() -> None:
    """Control: shows the age term is doing real work, not decoration."""
    torch.manual_seed(3)
    d, n = 256, 16
    layer = InputRestrictedPeriodic(d, d, keep_fraction=0.05, recompute_period=10,
                                    age_bounded=False)
    X = torch.randn(n, d)
    G_Y = torch.randn(n, d)
    for _ in range(200):
        layer.backward(G_Y, X)          # fixed G_Y => static stale ranking
    assert int(layer._age.max()) > layer.max_age


def test_recompute_step_does_not_run_a_second_gemm() -> None:
    """On a recompute step the partial must be sliced from the dense result."""
    torch.manual_seed(4)
    d, n = 128, 8
    layer = InputRestrictedPeriodic(d, d, keep_fraction=0.10, recompute_period=100)
    X = torch.randn(n, d)
    G_Y = torch.randn(n, d)
    (idx, dXp), executed, _, _, _ = layer.backward(G_Y, X)
    assert executed == pytest.approx(2.0 * n * d * d)   # dense only, not dense + restricted
    assert torch.allclose(dXp, (G_Y @ layer.W)[:, idx], atol=1e-5)


def test_amortized_cost_matches_the_stated_model() -> None:
    """Mean executed FLOPs must match [1 + (m-1)*k/d_in] / m x dense.

    Note the (m-1), not m: on a recompute step the partial is sliced from the
    already-computed dense result, so no restricted GEMM runs that step. The
    naive (1/m + k/d_in) model double-counts it and is ~3% pessimistic.
    """
    torch.manual_seed(5)
    d, n, m, keep = 256, 16, 10, 0.05
    layer = InputRestrictedPeriodic(d, d, keep_fraction=keep, recompute_period=m)
    X = torch.randn(n, d)
    steps = 100
    total = sum(layer.backward(torch.randn(n, d), X)[1] for _ in range(steps))
    dense = 2.0 * n * d * d
    predicted = (1.0 + (m - 1) * layer.k / d) / m * dense * steps
    assert total == pytest.approx(predicted, rel=0.01)
