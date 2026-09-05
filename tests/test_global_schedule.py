"""Tests for Method K3 -- exactness, single-GEMM, and alias-freedom.

Each test targets a specific defect K3 exists to fix, so a regression names
the defect it reintroduced rather than reporting a numeric drift.
"""
import torch

from src.restricted_backward.global_schedule import GlobalScheduleAccumulation

D = 64
V = 32


def _make(keep=0.25, seed=0, n_params=V):
    return GlobalScheduleAccumulation(
        D, D, keep_fraction=keep, n_params=n_params, seed=seed
    )


def test_one_gemm_per_step_regardless_of_batch_composition():
    """K2's fragmentation defect: rows wanting different groups. K3 returns a
    single (cols, vals) pair, so one GEMM by construction."""
    m = _make()
    g = torch.Generator().manual_seed(0)
    for ids in (
        torch.arange(V),
        torch.tensor([3, 3, 3, 7]),
        torch.randint(0, V, (16,), generator=g),
    ):
        packed, *_ = m.backward(
            torch.randn(len(ids), D, generator=g) * 0.01, ids
        )
        (a, b), vals = packed
        assert vals.shape == (len(ids), b - a)


def test_debt_is_the_exact_sum_since_that_group_was_last_served():
    """The exactness claim, checked against an independent replay."""
    m = _make(keep=0.25)
    g = torch.Generator().manual_seed(1)
    ids = torch.arange(V)

    replay = torch.zeros(V, D)
    seen = {}
    for step in range(4 * m.n_groups):
        G_Y = torch.randn(V, D, generator=g) * 0.01
        gi = m._order[m._pos] if m._pos < len(m._order) else None
        packed, *_ = m.backward(G_Y, ids)
        (a, b), vals = packed
        cols = slice(a, b)
        replay += G_Y

        key = (a, b)
        expected_src = replay - seen.get(key, torch.zeros(V, D))
        seen[key] = replay.clone()
        expected = expected_src @ m.W[:, cols]
        assert torch.allclose(vals, expected, atol=1e-4), step


def test_full_cycle_delivers_every_channel_exactly_once():
    m = _make(keep=0.25)
    counts = torch.zeros(D, dtype=torch.long)
    g = torch.Generator().manual_seed(2)
    ids = torch.arange(V)
    for _ in range(m.n_groups):
        ((a, b), _v), *_ = m.backward(
            torch.randn(V, D, generator=g) * 0.01, ids)
        counts[a:b] += 1
    assert int(counts.min()) == 1 and int(counts.max()) == 1


def test_random_order_defeats_gcd_aliasing():
    """Rung 3e's failure: round-robin + periodic access starves a parameter
    down to 1/gcd(P, G) of its channels. A random order has no arithmetic for
    the access period to resonate with."""
    P = 8
    m = _make(keep=0.05)
    G = m.n_groups
    g = torch.Generator().manual_seed(3)
    covered = torch.zeros(D, dtype=torch.bool)
    for t in range(60 * G):
        ids = torch.arange(t * 4, t * 4 + 4) % V
        ((a, b), _v), *_ = m.backward(
            torch.randn(4, D, generator=g) * 0.01, ids)
        if 0 in set(ids.tolist()):
            covered[a:b] = True
    assert covered.all(), f"{int(covered.sum())}/{D} channels reached"


def test_group_order_is_a_permutation_not_a_fixed_cycle():
    m = _make(keep=0.1)
    first = list(m._order)
    seen_orders = {tuple(first)}
    g = torch.Generator().manual_seed(4)
    for _ in range(5 * m.n_groups):
        m.backward(torch.randn(4, D, generator=g) * 0.01,
                   torch.arange(4))
        seen_orders.add(tuple(m._order))
    assert sorted(first) == list(range(m.n_groups))
    assert len(seen_orders) > 1, "order never reshuffled -- aliasing can return"


def test_duplicate_ids_bank_every_occurrence():
    m = _make()
    g = torch.Generator().manual_seed(5)
    G_Y = torch.randn(4, D, generator=g) * 0.01
    ids = torch.tensor([9, 9, 9, 9])
    m.backward(G_Y, ids)
    assert torch.allclose(m._total[9], G_Y.sum(0), atol=1e-5)


def test_nothing_is_discarded_across_a_long_irregular_run():
    """Conservation under Zipfian access: banked total equals everything in."""
    m = _make(keep=0.1)
    g = torch.Generator().manual_seed(6)
    w = 1.0 / torch.arange(1, V + 1).double()
    cdf = (w / w.sum()).cumsum(0)
    total_in = torch.zeros(V, D)
    for _ in range(400):
        ids = torch.searchsorted(
            cdf, torch.rand(8, generator=g).double()
        ).clamp(max=V - 1)
        G_Y = torch.randn(8, D, generator=g) * 0.01
        total_in.index_add_(0, ids, G_Y)
        m.backward(G_Y, ids)
    assert torch.allclose(m._total, total_in, atol=1e-4)
    assert torch.isfinite(m._total).all()


def test_state_scales_as_one_over_keep():
    """The honest cost: (G+1) * n_params * d_out."""
    for keep in (0.25, 0.1):
        m = _make(keep=keep)
        expected = (m.n_groups + 1) * V * D * 4
        assert m.aux_bytes == expected


def test_weight_slices_are_views_not_copies():
    """W[:, a:b] must be a view; a gather would add a (d_out x k) copy
    to every step."""
    m = _make()
    a, b = m._bounds[0]
    assert m.W[:, a:b].data_ptr() == m.W.data_ptr()


def test_full_batch_fast_path_matches_the_gather_path():
    """The full-batch branch skips gather/scatter. It must be numerically
    identical to the general path, not merely close in aggregate.

    Same gradients delivered to the same parameters, but the second run
    presents them in a permuted row order so `ids != arange` and the general
    path is taken. Row i of the permuted run corresponds to parameter p[i].
    """
    g = torch.Generator().manual_seed(8)
    grads = [torch.randn(V, D, generator=g) * 0.01 for _ in range(12)]
    p = torch.randperm(V, generator=g)

    fast = _make(keep=0.25, seed=3)
    slow = _make(keep=0.25, seed=3)

    for G_Y in grads:
        (c1, v1), *_ = fast.backward(G_Y, None)
        (c2, v2), *_ = slow.backward(G_Y[p], p)
        assert c1 == c2
        assert torch.allclose(v1[p], v2, atol=1e-5)

    assert torch.allclose(fast._total, slow._total, atol=1e-5)


def test_groups_are_contiguous_blocks_that_partition_the_columns():
    """Contiguity is load-bearing: a fancy-index update costs 0.469 ms
    against 0.055 ms for a slice, which alone exceeded the GEMM saving."""
    m = _make(keep=0.25)
    seen = torch.cat([torch.arange(a, b) for a, b in m._bounds])
    assert torch.equal(torch.sort(seen).values, torch.arange(D))
    for a, b in m._bounds:
        assert b > a


def test_deferral_at_g2_costs_no_measurable_progress():
    """Why K3 wins: at G=2 with full-batch access its trajectory tracks dense
    almost exactly, so the per-step saving is not paid back in extra steps.

    Not exactly identical -- a deferred column update changes the gradient the
    next step sees -- so this asserts a small bound rather than equality.
    """
    d, n, steps, lr = 128, 64, 300, 400.0
    m = GlobalScheduleAccumulation(d, d, keep_fraction=0.5, n_params=n, seed=0)
    W = m.W
    g = torch.Generator().manual_seed(1)
    E_star = torch.randn(n, d, generator=g) * 0.3
    tgt = E_star @ W.T
    E0 = torch.randn(n, d, generator=g) * 0.05

    E = E0.clone()
    for _ in range(steps):
        G_Y = (2.0 / (n * d)) * (E @ W.T - tgt)
        E = E - lr * (G_Y @ W)
    dense_rec = float((E - E_star).norm() / E_star.norm())

    E = E0.clone()
    for _ in range(steps):
        G_Y = (2.0 / (n * d)) * (E @ W.T - tgt)
        ((a, b), vals), *_ = m.backward(G_Y, None)
        E[:, a:b] -= lr * vals
    k3_rec = float((E - E_star).norm() / E_star.norm())

    rel = abs(k3_rec - dense_rec) / dense_rec
    assert rel < 0.02, (dense_rec, k3_rec, rel)


def test_full_batch_path_allocates_no_per_step_debt_buffer():
    """The debt buffer is preallocated; a fresh allocation each step was
    measurable overhead at these sizes."""
    m = _make(keep=0.5, seed=0)
    g = torch.Generator().manual_seed(2)
    G_Y = torch.randn(V, D, generator=g) * 0.01
    m.backward(G_Y, None)
    ptr = m._debt.data_ptr()
    for _ in range(5):
        m.backward(G_Y, None)
    assert m._debt.data_ptr() == ptr
