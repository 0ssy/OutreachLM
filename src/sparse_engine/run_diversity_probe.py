"""Does the expert pool help beyond the data's own diversity?

Probe A showed loss improving from 32 to 128 experts and then FLAT to 1024,
on a task with 32 latent concepts. That flat region is the important part: it
suggests the useful pool size is set by the number of distinct things there
are to specialise in, not by how many experts you can afford to store.

If useful experts scale with concepts, then 70,000 experts is only justified
if the data contains on the order of thousands of genuinely distinct latent
structures. Otherwise the extra experts are storage and I/O that buy nothing,
and the 70B table should be spent differently.

Two measurements:
  1. Sweep concepts against pool size. Find where each curve flattens.
  2. Measure how many experts are LOAD-BEARING rather than merely visited --
     an expert can be routed to and still contribute nothing.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.sparse_engine.routing_probe import MoE, make_task, run

torch.set_num_threads(6)
SEEDS = (0, 1)
K = 2


def agg(**kw) -> float:
    rs = [run(seed=s, **kw) for s in SEEDS]
    return sum(r["relative"] for r in rs) / len(rs)


def knee(losses: dict[int, float], tol: float = 0.10) -> int:
    """Smallest pool within `tol` of the best loss achieved."""
    best = min(losses.values())
    for e in sorted(losses):
        if losses[e] <= best * (1.0 + tol):
            return e
    return max(losses)


def essential_experts(n_experts: int, concepts: int, seed: int = 0,
                      d: int = 32, vocab: int = 512) -> int:
    """How many experts carry real load, by routing mass rather than by
    whether they were ever visited."""
    emb, concept_of, maps = make_task(vocab, concepts, d, seed)
    model = MoE(d, n_experts, K, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed + 99)
    for _ in range(600):
        ids = torch.randint(0, vocab, (256,), generator=g)
        x = emb[ids]
        y = torch.bmm(x.unsqueeze(1), maps[concept_of[ids]]).squeeze(1)
        pred, bal, _ = model(x, None)
        loss = F.mse_loss(pred, y) + 1e-2 * bal
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x = emb[torch.arange(vocab)]
        logits = model.router(x)
        topv, topi = torch.topk(logits, K, dim=-1)
        gate = F.softmax(topv, dim=-1)
        mass = torch.zeros(n_experts)
        mass.scatter_add_(0, topi.reshape(-1), gate.reshape(-1))
        mass = mass / mass.sum()
        # Experts covering 95% of routing mass, largest first.
        srt = torch.sort(mass, descending=True).values
        return int((torch.cumsum(srt, 0) < 0.95).sum()) + 1


def main() -> None:
    print("Does the expert pool help beyond the data's own diversity?")
    print(f"k={K} active per token in every arm. 2 seeds.\n")

    print("A. POOL SIZE vs TASK DIVERSITY (relative loss)")
    pools = (16, 32, 64, 128, 256, 512)
    hdr = f"{'concepts':>10}" + "".join(f"{p:>9}" for p in pools) + \
          f"{'knee':>8}{'knee/C':>9}"
    print(hdr)
    print("-" * len(hdr))
    knees = {}
    for c in (8, 16, 32, 64):
        losses = {}
        cells = []
        for p in pools:
            lo = agg(n_experts=p, k=K, concepts=c)
            losses[p] = lo
            cells.append(f"{lo:>9.4f}")
        kn = knee(losses)
        knees[c] = kn
        print(f"{c:>10}" + "".join(cells) + f"{kn:>8}{kn / c:>9.1f}")

    print("\n  'knee' is the smallest pool within 10% of that row's best loss.")
    print("  If knee/C is roughly constant, useful experts scale with the")
    print("  number of distinct structures in the data, not with storage.")

    print("\nB. LOAD-BEARING EXPERTS (95% of routing mass), 512-expert pool")
    print(f"{'concepts':>10}{'pool':>8}{'load-bearing':>15}{'of pool':>10}")
    print("-" * 43)
    for c in (8, 16, 32, 64):
        n = essential_experts(512, c)
        print(f"{c:>10}{512:>8}{n:>15}{n / 512:>9.0%}")


if __name__ == "__main__":
    main()
