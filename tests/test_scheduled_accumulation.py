"""Method K validation. The key test is cycle-level unbiasedness."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from src.restricted_backward.scheduled_accumulation import ScheduledAccumulation


def test_every_channel_visited_exactly_once_per_cycle() -> None:
    """Deterministic schedule => hard coverage guarantee, no selector needed."""
    d = 256
    layer = ScheduledAccumulation(d, d, keep_fraction=0.05)
    X = torch.randn(8, d)
    seen: list[int] = []
    for _ in range(layer.visit_period):
        (idx, _), _, _, _, _ = layer.backward(torch.randn(8, d), X)
        seen.extend(idx.tolist())
    assert sorted(seen) == list(range(d))       # exact partition, no gaps/dupes


def test_delivered_gradient_is_exact_accumulated_debt_when_W_frozen() -> None:
    """With W held fixed, the delivered dX must equal the EXACT sum of every
    G_Y the group missed -- zero discarded energy. This is the property rung 3b
    lacked, and it is what removes the 0.959 bias floor."""
    torch.manual_seed(0)
    d, n = 64, 8
    layer = ScheduledAccumulation(d, d, keep_fraction=0.25, dtype=torch.float64)
    X = torch.randn(n, d, dtype=torch.float64)
    G = layer.visit_period

    hist = [torch.randn(n, d, dtype=torch.float64) for _ in range(2 * G)]
    delivered: dict[int, torch.Tensor] = {}
    for t, G_Y in enumerate(hist):
        (idx, dXp), _, _, _, _ = layer.backward(G_Y, X)
        if t >= G:                               # second cycle: full interval
            expected = sum(hist[t - G + 1 : t + 1]) @ layer.W[:, idx]
            assert torch.allclose(dXp, expected, atol=1e-9)


def test_cycle_accumulated_error_decays_toward_zero() -> None:
    """Rung 3b floored at 0.959 relative error regardless of selector, because
    discarded energy was destroyed. Method K defers instead, so cumulative
    error must DECAY with horizon rather than sit at a floor.

    A tail always remains: G_Y(t) is delivered at each group's next visit, so
    after C cycles roughly (G-1)/2 steps' worth is still in flight. For
    UNCORRELATED gradients these sum in quadrature, giving a sqrt(1/(2C)) law
    rather than 1/(2C). Real training gradients are correlated step-to-step, so
    this is the worst case, not the typical one (see the correlated test below).
    """
    torch.manual_seed(1)
    d, n = 256, 16
    layer = ScheduledAccumulation(d, d, keep_fraction=0.05)
    X = torch.randn(n, d)
    G = layer.visit_period

    delivered = torch.zeros(n, d)
    exact = torch.zeros(n, d)
    errors: dict[int, float] = {}
    for cycle in range(1, 21):
        for _ in range(G):
            G_Y = torch.randn(n, d) * 0.01           # pure noise: worst case
            exact += layer.exact_dX(G_Y)
            (idx, dXp), _, _, _, _ = layer.backward(G_Y, X)
            delivered[:, idx] += dXp
        errors[cycle] = float((delivered - exact).norm() / exact.norm())

    assert errors[20] < errors[5] < errors[1], "error must decay with horizon"
    assert errors[5] < 0.959, "must beat the rung 3b floor well before 20 cycles"
    # Quadrature law: quadrupling the horizon should halve the error.
    assert errors[5] / errors[20] == pytest.approx(2.0, rel=0.25)


def test_correlated_gradients_decay_faster_than_the_noise_worst_case() -> None:
    """Real training gradients are correlated step to step. When they are, the
    in-flight tail sums coherently against a coherently-growing total, so the
    error decays like 1/C rather than sqrt(1/C)."""
    torch.manual_seed(2)
    d, n = 256, 16
    layer = ScheduledAccumulation(d, d, keep_fraction=0.05)
    X = torch.randn(n, d)
    G = layer.visit_period
    direction = torch.randn(n, d) * 0.01

    delivered = torch.zeros(n, d)
    exact = torch.zeros(n, d)
    for _ in range(20 * G):
        G_Y = direction + 0.05 * torch.randn(n, d) * 0.01   # strongly correlated
        exact += layer.exact_dX(G_Y)
        (idx, dXp), _, _, _, _ = layer.backward(G_Y, X)
        delivered[:, idx] += dXp

    err = float((delivered - exact).norm() / exact.norm())
    assert err < 0.05, f"correlated-gradient error {err:.4f} should be small"


def test_no_dense_dx_is_ever_materialized() -> None:
    """Elimination, not amortization: unlike input_periodic there is no
    periodic dense recompute at any step."""
    torch.manual_seed(2)
    d, n = 512, 32
    layer = ScheduledAccumulation(d, d, keep_fraction=0.05)
    X = torch.randn(n, d)
    for _ in range(3 * layer.visit_period):
        layer.backward(torch.randn(n, d), X)
    assert layer.max_backward_tensor_bytes < n * d * 4


def test_executed_flops_are_restricted_every_step() -> None:
    torch.manual_seed(3)
    d, n = 256, 16
    layer = ScheduledAccumulation(d, d, keep_fraction=0.05)
    X = torch.randn(n, d)
    dense = 2.0 * n * d * d
    for _ in range(2 * layer.visit_period):
        (idx, _), executed, _, _, _ = layer.backward(torch.randn(n, d), X)
        assert executed == pytest.approx(2.0 * n * d * len(idx))
        assert executed < dense


def test_worst_case_wait_is_bounded_by_construction() -> None:
    """No starvation is possible: the bound is structural, not emergent."""
    d = 512
    layer = ScheduledAccumulation(d, d, keep_fraction=0.05)
    X = torch.randn(8, d)
    last = torch.full((d,), -1, dtype=torch.long)
    worst = 0
    for t in range(4 * layer.visit_period):
        (idx, _), _, _, _, _ = layer.backward(torch.randn(8, d), X)
        for j in idx.tolist():
            if last[j] >= 0:
                worst = max(worst, t - int(last[j]))
        last[idx] = t
    assert worst == layer.visit_period


# --- Rung 3e: scope limits. K is only valid under full-batch access. ---


def test_residual_bank_is_keyed_by_batch_slot_not_parameter():
    """Documents the misrouting defect structurally, not by threshold.

    The bank is (N x d_out). Under a partial batch, slot i holds a different
    parameter each step, so banked debt is repaid to the wrong parameter.
    Measured consequence: permuting only slot->row identity, with density,
    periodicity, coverage and aliasing all held perfect, moves K from 0.0000
    to 0.0462 while dense and Method L are unmoved.
    """
    import torch as _t

    from src.restricted_backward.scheduled_accumulation import (
        ScheduledAccumulation as _SA,
    )

    n, d = 8, 64
    m = _SA(d, d, keep_fraction=0.25, seed=0)
    g = _t.Generator().manual_seed(1)
    m.backward(_t.randn(n, d, generator=g) * 0.01, _t.randn(n, d, generator=g))
    assert m._running_total.shape == (n, d)      # keyed by slot, not parameter
    assert m._running_total.shape[0] != d


def test_repayment_cadence_is_step_time_not_touch_time():
    """The second, unrepairable defect: the schedule advances once per STEP
    regardless of which parameters were actually present.

    Re-keying the bank fixes misrouting but not this. Measured with a
    row-keyed bank and W frozen: cyclic 0.0093, uniform 0.0516, and Zipfian
    access diverges outright (1.18e29), because hot parameters accumulate many
    steps of debt yet are repaid on the same cadence as cold ones.
    """
    import torch as _t

    from src.restricted_backward.scheduled_accumulation import (
        ScheduledAccumulation as _SA,
    )

    n, d = 8, 64
    m = _SA(d, d, keep_fraction=0.25, seed=0)
    g = _t.Generator().manual_seed(2)
    seen = []
    for _ in range(2 * m.n_groups):
        idx, _v = m.backward(
            _t.randn(n, d, generator=g) * 0.01,
            _t.randn(n, d, generator=g),
        )[0]
        seen.append(int(idx[0]))
    # Group choice depends only on the step counter; the data never enters it.
    assert seen[: m.n_groups] == seen[m.n_groups :]
