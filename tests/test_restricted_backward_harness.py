"""Phase 2 exit criterion: the harness must be provably correct on Method A.

If A does not show exactly zero gradient error and exactly matching FLOP
counts, the harness is broken and no B-E result can be trusted.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.restricted_backward.harness import build_layer, make_task, rho_slope, run_config


def _run(kind: str, label: str, steps: int = 6, **kw):
    torch.manual_seed(0)
    d = 64
    X, Y_true = make_task(8, d, d, seed=0)
    layer = build_layer(kind, d, d, **kw)
    return run_config(
        kind, label, layer, X, Y_true,
        steps=steps, lr=0.1, rho_window=4, keep_trajectory=True,
    )


def test_harness_reports_exact_zero_error_for_method_a() -> None:
    """A is the reference: its gradient error must be identically zero."""
    summary, rows = _run("A", "dense")
    assert summary["mean_grad_error"] == 0.0
    assert all(row["grad_error"] == 0.0 for row in rows)


def test_harness_theoretical_equals_executed_flops_for_method_a() -> None:
    summary, rows = _run("A", "dense")
    assert summary["mean_executed_flops_per_step"] == summary["theoretical_flops_per_step"]
    for row in rows:
        assert row["executed_flops"] == row["theoretical_flops"]


def test_method_a_actually_converges() -> None:
    """Guards against a harness that reports zero error on a dead loop."""
    summary, rows = _run("A", "dense", steps=40)
    assert rows[-1]["loss"] < rows[0]["loss"]


def test_sparse_methods_have_no_residual_for_method_c() -> None:
    summary, rows = _run("C", "k10pct", keep=0.10)
    assert summary["rho_slope"] is None
    assert all(row["residual_ratio"] is None for row in rows)
    assert all(row["feedback_time"] == 0.0 for row in rows)


def test_method_d_tracks_residual_ratio() -> None:
    summary, rows = _run("D", "k10pct", keep=0.10, ef=True)
    assert summary["rho_slope"] is not None
    assert all(row["residual_ratio"] is not None for row in rows)


def test_executed_flops_below_theoretical_for_sparse() -> None:
    summary, _ = _run("C", "k1pct", keep=0.01)
    assert summary["mean_executed_flops_per_step"] < summary["theoretical_flops_per_step"]


def test_single_layer_saving_cannot_exceed_one_third() -> None:
    """Phase 0 bound: only dW is sparsified, so max saving is 33.3%."""
    summary, _ = _run("C", "k1pct", keep=0.01)
    saved = 1 - summary["mean_executed_flops_per_step"] / summary["theoretical_flops_per_step"]
    assert saved <= 1.0 / 3.0 + 1e-9


def test_rho_slope_detects_growing_residual() -> None:
    assert rho_slope([1.0, 2.0, 3.0, 4.0], window=4) > 0
    assert rho_slope([4.0, 3.0, 2.0, 1.0], window=4) < 0
    assert abs(rho_slope([2.0, 2.0, 2.0, 2.0], window=4)) < 1e-12
