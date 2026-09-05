"""Probe C: a task hard enough that the expert pool has to earn its size.

Probes A and B both bottomed out at a relative loss of ~0.004 for every pool
size and every concept count, and the knee sat at ~128 experts whether the
task had 8 latent concepts or 64. A knee that does not move when the task
changes is not measuring the task -- it is measuring a floor, and any
conclusion about pool size drawn from it would be an artifact.

Two corrections here:

  1. HARDER TASK. Concepts are raised to 256 and 512 so that solving it
     requires real specialisation rather than a generic mixture. If a pool
     larger than the concept count still helps, that is evidence for the
     large-table design; if it flattens at roughly the concept count, the
     70,000-expert table is only justified by data with comparable diversity.

  2. UNCONTAMINATED UTILISATION. The load-balancing loss deliberately spreads
     routing mass, so "fraction of experts carrying 95% of mass" measures the
     balance coefficient, not the model. Utilisation is therefore reported at
     two balance weights, including zero.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.sparse_engine.routing_probe import MoE, make_task, run

torch.set_num_threads(6)
K = 2


def agg(seeds=(0, 1), **kw) -> float:
    rs = [run(seed=s, **kw) for s in seeds]
    return sum(r["relative"] for r in rs) / len(rs)


def utilisation(n_experts: int, concepts: int, bal_weight: float,
                seed: int = 0, d: int = 32, vocab: int = 1024,
                steps: int = 900) -> tuple[int, float]:
    emb, concept_of, maps = make_task(vocab, concepts, d, seed)
    model = MoE(d, n_experts, K, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed + 99)
    for _ in range(steps):
        ids = torch.randint(0, vocab, (256,), generator=g)
        x = emb[ids]
        y = torch.bmm(x.unsqueeze(1), maps[concept_of[ids]]).squeeze(1)
        pred, bal, _ = model(x, None)
        loss = F.mse_loss(pred, y) + bal_weight * bal
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
        srt = torch.sort(mass, descending=True).values
        n95 = int((torch.cumsum(srt, 0) < 0.95).sum()) + 1
        y = torch.bmm(x.unsqueeze(1), maps[concept_of[torch.arange(vocab)]]
                      ).squeeze(1)
        pred, _, _ = model(x, None)
        rel = float(F.mse_loss(pred, y)) / float((y ** 2).mean())
    return n95, rel


def main() -> None:
    print("Probe C -- harder task, and utilisation measured without the")
    print("load-balancing loss dictating the answer.\n")

    print("A. POOL vs CONCEPTS, relative loss (vocab 1024, 900 steps)")
    pools = (64, 128, 256, 512, 1024)
    hdr = f"{'concepts':>10}" + "".join(f"{p:>10}" for p in pools)
    print(hdr)
    print("-" * len(hdr))
    for c in (64, 256, 512):
        cells = []
        for p in pools:
            lo = agg(n_experts=p, k=K, concepts=c, vocab=1024, steps=900)
            cells.append(f"{lo:>10.4f}")
        print(f"{c:>10}" + "".join(cells))

    print("\n  If a row keeps improving past its own concept count, pool size")
    print("  buys something beyond one-expert-per-structure. If it flattens")
    print("  at roughly C, the table is sized by the data, not by storage.")

    print("\nB. UTILISATION vs BALANCE WEIGHT (512 experts, 256 concepts)")
    print(f"{'bal weight':>12}{'experts @95% mass':>20}{'relative loss':>16}")
    print("-" * 48)
    for bw in (0.0, 1e-2):
        n95, rel = utilisation(512, 256, bw)
        print(f"{bw:>12.0e}{n95:>20}{rel:>16.4f}")
    print("\n  A large gap between these means the earlier '~40% of experts'")
    print("  figure was the balance coefficient talking, not the model.")


if __name__ == "__main__":
    main()
