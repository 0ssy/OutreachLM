"""A real MoE on a task where routing quality is the thing being measured.

The 70B design rests on two untested quality assumptions:

  1. 1000x sparsity (70,000 experts, ~70 active) retains the benefit of the
     large expert table. Published MoE runs sit far lower: Mixtral 47B/13B is
     3.6x, Switch explored ~100x with degradation.
  2. Sharding the table across two laptops, so each token routes among 35,000
     experts rather than 70,000, costs little.

Neither is settled by the compute budget, and both are cheap to probe.

THE TASK is built so that routing is the difficulty, not capacity. Each token
id belongs to one of C latent concepts, and the target is a concept-specific
linear map of the token's embedding:

    y = A_{concept(token)} @ x

A model that routes tokens of the same concept to the same expert can solve
this exactly once E >= C. A model that routes badly cannot, no matter how many
parameters it has. So final loss measures ROUTING, which is what the extreme
sparsity ratio actually stresses -- an ordinary language-modelling loss would
confound routing quality with capacity.

Controls that make the comparison fair:
  * active experts per token (k) is held FIXED across every arm, so compute
    per token is identical and only the pool size varies.
  * every arm gets the same token budget and the same seeds.
  * a dense (no routing) baseline of equal active size bounds what perfect
    routing could achieve.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE(nn.Module):
    """Top-k routed experts with an optional disjoint shard restriction."""

    def __init__(self, d: int, n_experts: int, k: int, *,
                 shards: int = 1, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d, self.n_experts, self.k, self.shards = d, n_experts, k, shards
        # Experts as a single batched tensor: (E, d, d).
        w = torch.randn(n_experts, d, d, generator=g) / math.sqrt(d)
        self.experts = nn.Parameter(w)
        self.router = nn.Linear(d, n_experts)
        with torch.no_grad():
            self.router.weight.copy_(
                torch.randn(n_experts, d, generator=g) * 0.02
            )
            self.router.bias.zero_()

    def forward(self, x: torch.Tensor, shard_id: torch.Tensor | None = None):
        logits = self.router(x)                          # (N, E)
        if self.shards > 1 and shard_id is not None:
            per = self.n_experts // self.shards
            ar = torch.arange(self.n_experts, device=x.device)
            owner = ar // per
            mask = owner.unsqueeze(0) != shard_id.unsqueeze(1)
            logits = logits.masked_fill(mask, float("-inf"))

        topv, topi = torch.topk(logits, self.k, dim=-1)
        gate = F.softmax(topv, dim=-1)                   # (N, k)

        # (N, k, d, d) gather would be huge; loop over k instead.
        out = torch.zeros_like(x)
        for j in range(self.k):
            idx = topi[:, j]                             # (N,)
            W = self.experts[idx]                        # (N, d, d)
            contrib = torch.bmm(x.unsqueeze(1), W).squeeze(1)
            out = out + gate[:, j].unsqueeze(1) * contrib

        # Load-balancing loss (Switch Transformer style): without it the
        # router collapses onto a few experts and the pool size is moot.
        probs = F.softmax(logits.masked_fill(
            torch.isinf(logits), -1e9), dim=-1)
        frac = torch.zeros(self.n_experts, device=x.device)
        frac.scatter_add_(
            0, topi.reshape(-1),
            torch.ones(topi.numel(), device=x.device),
        )
        frac = frac / max(1, topi.numel())
        bal = self.n_experts * (frac * probs.mean(0)).sum()
        return out, bal, topi


def make_task(vocab: int, concepts: int, d: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(vocab, d, generator=g)
    concept_of = torch.randint(0, concepts, (vocab,), generator=g)
    maps = torch.randn(concepts, d, d, generator=g) / math.sqrt(d)
    return emb, concept_of, maps


def run(n_experts: int, k: int, *, shards: int = 1, steps: int = 600,
        d: int = 32, vocab: int = 512, concepts: int = 32,
        batch: int = 256, seed: int = 0, lr: float = 3e-3,
        bal_weight: float = 1e-2) -> dict:
    emb, concept_of, maps = make_task(vocab, concepts, d, seed)
    model = MoE(d, n_experts, k, shards=shards, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 99)

    used = torch.zeros(n_experts)
    for _ in range(steps):
        ids = torch.randint(0, vocab, (batch,), generator=g)
        x = emb[ids]
        y = torch.bmm(x.unsqueeze(1), maps[concept_of[ids]]).squeeze(1)
        shard_id = (ids % shards) if shards > 1 else None
        pred, bal, topi = model(x, shard_id)
        loss = F.mse_loss(pred, y) + bal_weight * bal
        opt.zero_grad()
        loss.backward()
        opt.step()
        used.scatter_add_(
            0, topi.reshape(-1), torch.ones(topi.numel())
        )

    with torch.no_grad():
        ids = torch.arange(vocab)
        x = emb[ids]
        y = torch.bmm(x.unsqueeze(1), maps[concept_of[ids]]).squeeze(1)
        shard_id = (ids % shards) if shards > 1 else None
        pred, _, _ = model(x, shard_id)
        final = float(F.mse_loss(pred, y))
        base = float((y ** 2).mean())

    return {
        "loss": final,
        "relative": final / base,
        "experts_used": int((used > 0).sum()),
        "n_experts": n_experts,
        "dead_frac": 1.0 - float((used > 0).sum()) / n_experts,
    }
