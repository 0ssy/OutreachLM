"""Tests for banked expert pruning -- the fix for magnitude-pruning freeze.

Each test targets the specific failure measured in the naive scheme, so a
regression names the defect it reintroduced.
"""
import pytest
import torch

from src.sparse_engine.banked_pruning import BankedExpertPruner

N, KEEP, STEPS = 2000, 20, 400


def _world(seed=0):
    g = torch.Generator().manual_seed(seed)
    # Heavy-tailed true gradient scale, as real MoE expert usage is.
    true_scale = torch.exp(torch.randn(N, generator=g))
    # Router affinity correlates with true value but is not identical.
    affinity = true_scale * torch.exp(0.5 * torch.randn(N, generator=g))
    affinity = affinity / affinity.sum() * N * 0.01
    return g, true_scale, affinity


def _run_banked(seed=0, decay=1.0, steps=STEPS, max_staleness=None):
    g, true_scale, affinity = _world(seed)
    p = BankedExpertPruner(N, KEEP, decay=decay,
                           max_staleness=max_staleness, seed=seed)
    for _ in range(steps):
        sel = p.select()
        obs = true_scale[sel] * torch.exp(
            0.1 * torch.randn(len(sel), generator=g)
        )
        p.observe(sel, obs, affinity)
    return p


def _run_naive(seed=0):
    """The scheme that freezes: magnitude only, no credit."""
    g, true_scale, _ = _world(seed)
    est = torch.ones(N)
    updates = torch.zeros(N)
    for _ in range(STEPS):
        sel = torch.topk(est, KEEP).indices
        updates[sel] += 1
        est[sel] = true_scale[sel] * torch.exp(
            0.1 * torch.randn(len(sel), generator=g)
        )
    return int((updates == 0).sum())


def test_naive_pruning_freezes_most_of_the_pool():
    """The defect, reproduced so the fix has something to be measured against."""
    never = _run_naive()
    assert never / N > 0.9


def test_banking_defers_rather_than_starves():
    """Horizon-bounded, per the rung-3e stopping-time rule: "never served in
    400 steps" is not "never served". Measured on this pool (8051x affinity
    spread): 795 -> 144 -> 10 -> 0 at 400 / 1600 / 6400 / 25600 steps."""
    short = _run_banked(steps=400).never_updated
    longer = _run_banked(steps=1600).never_updated
    assert short < _run_naive()
    assert longer < short / 3


def test_no_expert_starves_permanently_without_an_age_bound():
    p = _run_banked(steps=25600)
    assert p.never_updated == 0


def test_credit_accumulates_only_while_an_expert_is_denied():
    p = BankedExpertPruner(8, 2, decay=1.0)
    aff = torch.ones(8)
    sel = torch.tensor([0, 1])
    p.observe(sel, torch.tensor([5.0, 5.0]), aff)
    assert float(p.credit[0]) == 0.0
    first = float(p.credit[2])
    assert first > 0.0
    p.observe(sel, torch.tensor([5.0, 5.0]), aff)
    assert float(p.credit[2]) > first


def test_credit_is_cleared_when_the_expert_is_finally_served():
    p = BankedExpertPruner(8, 2, decay=1.0)
    aff = torch.ones(8)
    p.observe(torch.tensor([0, 1]), torch.tensor([5.0, 5.0]), aff)
    assert float(p.credit[3]) > 0
    p.observe(torch.tensor([3, 4]), torch.tensor([0.1, 0.1]), aff)
    assert float(p.credit[3]) == 0.0


def test_wanted_experts_re_enter_but_unwanted_ones_stay_out():
    """The saving comes from experts the router genuinely does not want.
    Banking must not simply round-robin everything back in."""
    g = torch.Generator().manual_seed(3)
    affinity = torch.full((64,), 1e-6)
    affinity[:8] = 1.0                      # only 8 experts are ever wanted
    p = BankedExpertPruner(64, 4, decay=0.9)
    for _ in range(300):
        sel = p.select()
        p.observe(sel, torch.full((len(sel),), 0.01), affinity)
    wanted = int((p.updates[:8] > 0).sum())
    unwanted_share = float(p.updates[8:].sum()) / float(p.updates.sum())
    assert wanted == 8
    assert unwanted_share < 0.5, unwanted_share


def test_decay_below_one_reintroduces_permanent_starvation():
    """Why decay defaults to 1.0. Any decay < 1 caps credit at
    rate/(1-decay); if that ceiling is under the selection threshold the
    expert never re-enters, which is the bug this class exists to fix."""
    capped = _run_banked(steps=6400, decay=0.9).never_updated
    uncapped = _run_banked(steps=6400, decay=1.0).never_updated
    assert capped > uncapped


def test_credit_grows_without_bound_under_default_decay():
    p = BankedExpertPruner(16, 2, decay=1.0)
    aff = torch.ones(16)
    seen = []
    for _ in range(50):
        p.observe(torch.tensor([0, 1]), torch.tensor([9.0, 9.0]), aff)
        seen.append(float(p.credit[5]))
    assert seen[-1] > seen[len(seen) // 2] > seen[0] > 0


def test_age_bound_converts_the_guarantee_from_asymptotic_to_deterministic():
    """Without a bound, latency is inverse to priority and can reach 128x the
    fair-share interval. With one, staleness is capped by construction -- and
    the cost of that cap is reported rather than hidden."""
    p = _run_banked(steps=1200, max_staleness=400)
    assert int(p.staleness().max()) <= 400 + 1
    assert p.never_updated == 0
    assert p.forced_admissions > 0


def test_age_bound_below_fair_share_is_rejected():
    with pytest.raises(ValueError, match="fair-share"):
        BankedExpertPruner(1000, 10, max_staleness=50)


def test_service_rate_tracks_how_much_the_router_wants_an_expert():
    """Not equal service -- prioritised service. This is the rung-3e result:
    the discarded signal is deferred, not destroyed, and latency is inverse
    to priority."""
    g = torch.Generator().manual_seed(11)
    aff = torch.exp(torch.randn(200, generator=g))
    p = BankedExpertPruner(200, 10, decay=1.0)
    for _ in range(600):
        sel = p.select()
        p.observe(sel, torch.full((len(sel),), 1.0), aff)
    top = torch.topk(aff, 40).indices
    bot = torch.topk(-aff, 40).indices
    assert float(p.updates[top].float().mean()) > \
        float(p.updates[bot].float().mean())


def test_rejects_invalid_keep():
    with pytest.raises(ValueError):
        BankedExpertPruner(10, 0)
    with pytest.raises(ValueError):
        BankedExpertPruner(10, 11)


def test_observe_rejects_misaligned_magnitudes():
    p = BankedExpertPruner(8, 2)
    with pytest.raises(ValueError):
        p.observe(torch.tensor([0, 1]), torch.tensor([1.0]), torch.ones(8))
