"""Expert pruning that does NOT freeze the pruned pool.

THE PROBLEM, MEASURED
    Gradient-magnitude pruning of experts is self-reinforcing: a pruned expert
    receives no tokens, so it emits no gradient, so its magnitude estimate
    never refreshes, so it is never reselected. Simulated at 95.8% of experts
    never updated. Adding random re-probing restores coverage (34.9% never
    updated) but spends exactly the compute the pruning was meant to save.

THE FIX, AND WHY IT IS NOT NEW
    This project already solved this at row granularity in rung 3e. Method F
    selected rows by gradient magnitude and starved 31% of them; Method G
    added error feedback -- banking the discarded signal so a neglected unit
    accumulates residual until its effective magnitude becomes competitive --
    and reached 100% coverage, more uniform than a uniform-random selector.
    The measured mechanism was that the residual supplied 93.5% of the
    selection statistic and dominated in 99.2% of selections.

    The same mechanism applies at expert granularity, and the accounting is
    even cheaper here. Banking a full expert gradient would cost what
    computing it costs, which would defeat the purpose. But selection does not
    need the gradient -- it needs a SCALAR priority per expert. So this banks
    the ROUTER AFFINITY that a pruned expert would have received, which is
    already computed for every expert on every token as part of routing, and
    costs nothing extra.

        priority_i = observed_grad_magnitude_i + credit_i
        credit_i  += fair_share_rate * (affinity_i / mean_affinity)
        credit_i   = 0                           when expert i is selected

    SCALE MATCHING IS LOAD-BEARING, AND GETTING IT WRONG SILENTLY FAILS.
    A first version of this class accumulated raw router affinity, whose mean
    was ~0.01, into a priority competing against gradient magnitudes whose
    mean was ~1.65, under a decay of 0.9 that capped credit at 0.1. Credit
    could never reach the selection threshold and 1963 of 2000 experts still
    froze -- barely better than the 1916 of the naive scheme it was meant to
    fix. Method G works precisely because its residual accumulates in the SAME
    UNITS as the statistic it competes against.

    The rate is therefore set so that an expert of mean affinity accumulates
    one typical magnitude over one fair-share interval (n_experts / keep
    steps). An expert wanted 10x more than average re-enters 10x sooner; one
    wanted 10x less waits 10x longer. That reproduces the rung-3e measurement
    directly, where the mean selection gap was 19.7 against a round-robin
    floor of 20.0.

    DECAY MUST BE 1.0 FOR THE GUARANTEE. Any decay < 1 caps credit at
    rate / (1 - decay); if that ceiling sits below the selection threshold the
    expert starves permanently, which is the exact failure being fixed. Decay
    is exposed only so the failure can be demonstrated in tests.

WHAT THIS DOES AND DOES NOT BUY
    It makes pruning safe: no permanent starvation, and re-entry time inverse
    to how much the router wants the expert.
    It does NOT make pruning free capacity. Compute is 6 * P_active * T, so
    reducing active experts reduces P_active, which is the model. Pruning is
    a way to spend a FIXED active budget on the experts that want it most, not
    a way to shrink the clock at constant quality.
"""
from __future__ import annotations

import torch


class BankedExpertPruner:
    """Gradient-magnitude expert selection with scale-matched credit."""

    def __init__(self, n_experts: int, keep: int, *,
                 decay: float = 1.0, max_staleness: int | None = None,
                 seed: int = 0):
        if keep < 1 or keep > n_experts:
            raise ValueError("keep must be in [1, n_experts]")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if max_staleness is not None and max_staleness < n_experts / keep:
            raise ValueError(
                "max_staleness below the fair-share interval forces every "
                "expert in every cycle, which is not pruning"
            )
        self.n_experts = n_experts
        self.keep = keep
        self.decay = decay
        self.max_staleness = max_staleness
        self.magnitude = torch.ones(n_experts)
        self.credit = torch.zeros(n_experts)
        self.updates = torch.zeros(n_experts, dtype=torch.long)
        self.last_seen = torch.zeros(n_experts, dtype=torch.long)
        self.forced_admissions = 0
        self._mag_scale = 1.0
        self._step = 0

    @property
    def fair_share_interval(self) -> float:
        """Steps between visits if every expert were served equally."""
        return self.n_experts / self.keep

    @property
    def priority(self) -> torch.Tensor:
        return self.magnitude + self.credit

    def select(self) -> torch.Tensor:
        """Top-k by priority, with any over-stale expert force-admitted.

        Without `max_staleness` the guarantee is asymptotic: credit grows
        without bound so every expert is eventually served, but latency is
        inverse to priority and can be long. Measured on a pool of 2000 with
        an 8051x affinity spread, experts never served fell 795 -> 144 -> 10
        -> 0 at 400 / 1600 / 6400 / 25600 steps, with worst-case latency
        12,803 steps against a fair-share interval of 100. That is deferral,
        not starvation -- but it is only bounded in the limit.

        `max_staleness` converts it into a deterministic bound, at the cost of
        spending some of the active budget on experts the router does not
        want. `forced_admissions` reports exactly how much.
        """
        if self.max_staleness is None:
            return torch.topk(self.priority, self.keep).indices

        stale = (self.staleness() > self.max_staleness).nonzero(
            as_tuple=True
        )[0]
        if stale.numel() == 0:
            return torch.topk(self.priority, self.keep).indices
        forced = stale[:self.keep]
        self.forced_admissions += int(forced.numel())
        if forced.numel() == self.keep:
            return forced
        remaining = self.keep - int(forced.numel())
        pri = self.priority.clone()
        pri[forced] = float("-inf")
        rest = torch.topk(pri, remaining).indices
        return torch.cat([forced, rest])

    def observe(self, selected: torch.Tensor, grad_magnitude: torch.Tensor,
                router_affinity: torch.Tensor) -> None:
        """Update state after a step.

        `grad_magnitude` is defined only for `selected` experts -- that is the
        point, the others were never computed. `router_affinity` is defined
        for ALL experts and is a by-product of routing, so banking it is free.
        """
        if grad_magnitude.shape[0] != selected.shape[0]:
            raise ValueError("grad_magnitude must align with selected")
        if router_affinity.shape[0] != self.n_experts:
            raise ValueError("router_affinity must cover every expert")
        self._step += 1

        # Running scale of observed magnitudes, so credit is comparable.
        obs = float(grad_magnitude.abs().mean())
        self._mag_scale = 0.9 * self._mag_scale + 0.1 * obs

        self.magnitude[selected] = grad_magnitude
        self.credit[selected] = 0.0
        self.updates[selected] += 1
        self.last_seen[selected] = self._step

        mean_aff = float(router_affinity.abs().mean()) or 1.0
        rate = self._mag_scale / self.fair_share_interval
        gain = rate * (router_affinity / mean_aff)

        mask = torch.ones(self.n_experts, dtype=torch.bool)
        mask[selected] = False
        self.credit[mask] = self.credit[mask] * self.decay + gain[mask]

    @property
    def never_updated(self) -> int:
        return int((self.updates == 0).sum())

    def staleness(self) -> torch.Tensor:
        """Steps since each expert was last selected."""
        return self._step - self.last_seen
