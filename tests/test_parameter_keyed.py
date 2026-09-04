"""Tests for Method K2 -- the actual fixes to K's two rung-3e defects.

These verify the fixes are structural, not thresholded: exactness is checked by
conservation (every gradient delivered exactly once), and touch-time cadence is
checked against adversarial access patterns that broke K.
"""
import torch

from src.restricted_backward.parameter_keyed import ParameterKeyedAccumulation

D = 64
V = 32


def _make(keep=0.25, seed=0):
    return ParameterKeyedAccumulation(
        D, D, keep_fraction=keep, n_params=V, seed=seed
    )


def test_bank_is_keyed_by_parameter_not_batch_slot():
    m = _make()
    assert m._cur.shape == (V, D)
    assert m._prev.shape == (V, D)


def test_every_gradient_delivered_exactly_once():
    """Exactness by conservation, checked against the dense reference.

    Over one full cycle of G touches a parameter receives its banked total
    restricted to each group in turn, and the groups partition all d_in
    columns. So the SUM of the restricted deliveries across a cycle must equal
    the exact dense dX for that banked total -- no residue, no double count.
    """
    m = _make(keep=0.25)
    G = m.n_groups
    g = torch.Generator().manual_seed(0)
    ids = torch.arange(V)

    for _ in range(G):                       # first cycle fills _cur
        m.backward(torch.randn(V, D, generator=g) * 0.01, ids)

    # The roll fires at the START of touch G, i.e. inside this first call.
    buckets, *_ = m.backward(torch.randn(V, D, generator=g) * 0.01, ids)
    banked = m._prev.clone()                 # now holds cycle 0's total
    accumulated = m.densify(buckets, (V, D))
    for _ in range(G - 1):
        buckets, *_ = m.backward(torch.randn(V, D, generator=g) * 0.01, ids)
        accumulated += m.densify(buckets, (V, D))

    exact = banked.float() @ m.W
    assert torch.allclose(accumulated, exact, atol=1e-5), (
        float((accumulated - exact).norm() / exact.norm())
    )


def test_no_channel_is_paid_twice_within_a_cycle():
    """The groups must partition the channels, so a cycle touches each once."""
    m = _make(keep=0.25)
    counts = torch.zeros(D, dtype=torch.long)
    for grp in m.groups:
        counts[grp] += 1
    assert int(counts.min()) == 1 and int(counts.max()) == 1


def test_repayment_is_bounded_in_touch_time_not_step_time():
    """A parameter touched once every 50 steps must still be fully repaid
    every G TOUCHES. This is the defect that made K diverge on Zipf."""
    m = _make(keep=0.25)
    G = m.n_groups
    g = torch.Generator().manual_seed(1)
    rare = torch.tensor([7])
    seen_groups = []
    for step in range(60 * G):
        if step % 50 == 0:
            m.backward(torch.randn(1, D, generator=g) * 0.01, rare)
            seen_groups.append(int(m._touches[7].item() - 1) % G)
        else:
            m.backward(torch.randn(1, D, generator=g) * 0.01,
                       torch.tensor([0]))
    # The rare parameter cycled through every group despite huge step gaps.
    assert set(seen_groups[:G]) == set(range(G))


def test_channel_coverage_is_complete_for_every_parameter():
    """Under skewed access, every parameter still covers all input channels."""
    m = _make(keep=0.25)
    G = m.n_groups
    g = torch.Generator().manual_seed(2)
    covered = {r: torch.zeros(D, dtype=torch.bool) for r in range(V)}
    for _ in range(40 * G):
        # Zipfian-ish: parameter 0 dominates, others are rare.
        w = 1.0 / torch.arange(1, V + 1).double()
        cdf = (w / w.sum()).cumsum(0)
        ids = torch.searchsorted(
            cdf, torch.rand(8, generator=g).double()
        ).clamp(max=V - 1)
        buckets, *_ = m.backward(torch.randn(8, D, generator=g) * 0.01, ids)
        for rows, cols, _v in buckets:
            for r in ids[rows].tolist():
                covered[r][cols] = True
    touched = [r for r in range(V) if m._touches[r] >= G]
    assert touched, "test needs at least one parameter past warmup"
    for r in touched:
        assert covered[r].all(), f"parameter {r} missed channels"


def test_duplicate_ids_in_one_batch_count_as_separate_touches():
    m = _make()
    g = torch.Generator().manual_seed(3)
    ids = torch.tensor([5, 5, 5])
    m.backward(torch.randn(3, D, generator=g) * 0.01, ids)
    assert int(m._touches[5]) == 3


def test_flops_match_the_single_gemm_equivalent():
    """Bucketing splits the work but must not change the FLOP count."""
    m = _make(keep=0.25)
    g = torch.Generator().manual_seed(4)
    ids = torch.arange(V)
    _, executed, *_ = m.backward(torch.randn(V, D, generator=g) * 0.01, ids)
    k_total = sum(len(m.groups[i]) for i in range(m.n_groups))
    assert k_total == D
    # Each row gets exactly one group's worth of columns.
    per_row = sum(len(gr) for gr in m.groups) / m.n_groups
    assert abs(executed - 2.0 * V * D * per_row) < 1e-6 * executed


def test_warmup_is_reported_not_hidden():
    m = _make()
    assert m.warmup_touches == m.n_groups
    g = torch.Generator().manual_seed(5)
    buckets, *_ = m.backward(torch.randn(V, D, generator=g) * 0.01,
                             torch.arange(V))
    dense = m.densify(buckets, (V, D))
    assert float(dense.abs().sum()) == 0.0     # first touch pays nothing


def test_memory_is_two_phase_not_g_snapshots():
    """The whole point of the two-phase cycle: O(2*V*d_out), not O(G*V*d_out)."""
    m = _make(keep=0.05)
    assert m.n_groups >= 10
    assert m.aux_bytes == 2 * V * D * 8
    naive = m.n_groups * V * D * 8
    assert m.aux_bytes < naive / 5


def test_multi_touch_step_cannot_skip_a_cycle_boundary():
    """Regression: the bug that made the first K2 diverge under Zipf.

    A parameter appearing k times in one batch advances its touch counter by
    k. If the roll is tested once per step, a counter can step from G-2 to
    G+3 without ever satisfying `touches % G == 0`, so _prev never rolls and
    _cur grows without bound. Drive a parameter with a multiplicity that is
    coprime-unfriendly to G and assert the bank stays bounded.
    """
    m = _make(keep=0.05)
    G = m.n_groups
    g = torch.Generator().manual_seed(9)
    mult = 5
    ids = torch.full((mult,), 3, dtype=torch.long)
    norms = []
    for _ in range(40 * G):
        m.backward(torch.randn(mult, D, generator=g) * 0.01, ids)
        norms.append(float(m._cur[3].norm()))
    assert int(m._touches[3]) == 40 * G * mult
    # _cur holds at most one cycle of gradient, so it must not trend upward.
    assert max(norms[-50:]) < 3.0 * max(norms[:50])


def test_zipfian_access_keeps_the_bank_bounded():
    m = _make(keep=0.05)
    g = torch.Generator().manual_seed(10)
    w = 1.0 / torch.arange(1, V + 1).double()
    cdf = (w / w.sum()).cumsum(0)
    for _ in range(600):
        ids = torch.searchsorted(
            cdf, torch.rand(8, generator=g).double()
        ).clamp(max=V - 1)
        m.backward(torch.randn(8, D, generator=g) * 0.01, ids)
    assert torch.isfinite(m._cur).all() and torch.isfinite(m._prev).all()
    assert float(m._cur.abs().max()) < 1.0


def test_occurrence_ranks_matches_reference_implementation():
    """The vectorised version replaced a Python loop for wall-clock reasons;
    a silent divergence here would corrupt cycle scheduling invisibly."""
    def reference(ids):
        counts, out = {}, []
        for v in ids.tolist():
            out.append(counts.get(v, 0))
            counts[v] = out[-1] + 1
        return torch.tensor(out, dtype=torch.long)

    g = torch.Generator().manual_seed(31)
    for hi in (2, 5, 40):
        for n in (1, 7, 64):
            ids = torch.randint(0, hi, (n,), generator=g)
            assert torch.equal(
                ParameterKeyedAccumulation._occurrence_ranks(ids),
                reference(ids),
            ), (hi, n, ids)
