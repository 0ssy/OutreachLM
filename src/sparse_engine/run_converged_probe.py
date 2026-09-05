"""Probe D: train past the floor, so pool size can actually separate.

Probes A-C all bottomed at a relative loss of ~0.005 regardless of pool size
or task diversity. The task is noiseless -- y = A_c x exactly -- so a perfect
model reaches 0 and that floor is an OPTIMISATION limit, not the task. Every
conclusion about pool size drawn at the floor is therefore unresolved rather
than negative.

This trains long enough for the arms to separate, and reports the trajectory
rather than a single endpoint so that "flat" can be distinguished from "not
yet converged".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.sparse_engine.routing_probe import MoE, make_task

torch.set_num_threads(6)
K = 2
D = 32
VOCAB = 1024
CONCEPTS = 512
MARKS = (900, 2700, 5400)


def train(n_experts: int, seed: int, steps: int = max(MARKS),
          bal_weight: float = 1e-3, shards: int = 1) -> dict[int, float]:
    emb, concept_of, maps = make_task(VOCAB, CONCEPTS, D, seed)
    model = MoE(D, n_experts, K, shards=shards, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed + 99)
    ids_all = torch.arange(VOCAB)
    x_all = emb[ids_all]
    y_all = torch.bmm(x_all.unsqueeze(1),
                      maps[concept_of[ids_all]]).squeeze(1)
    base = float((y_all ** 2).mean())
    sid_all = (ids_all % shards) if shards > 1 else None
    out = {}
    for t in range(1, steps + 1):
        ids = torch.randint(0, VOCAB, (256,), generator=g)
        x = emb[ids]
        y = torch.bmm(x.unsqueeze(1), maps[concept_of[ids]]).squeeze(1)
        sid = (ids % shards) if shards > 1 else None
        pred, bal, _ = model(x, sid)
        loss = F.mse_loss(pred, y) + bal_weight * bal
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if t in MARKS:
            with torch.no_grad():
                p, _, _ = model(x_all, sid_all)
                out[t] = float(F.mse_loss(p, y_all)) / base
    return out


def main() -> None:
    print(f"Probe D -- trained to {max(MARKS)} steps so arms can separate.")
    print(f"{CONCEPTS} concepts, k={K}, vocab {VOCAB}, 2 seeds.")
    print("Noiseless task: a perfect model reaches 0, so any plateau above")
    print("that is optimisation, not the task.\n")

    hdr = f"{'experts':>9}{'ratio':>8}" + "".join(
        f"{f'@{m}':>11}" for m in MARKS
    ) + f"{'vs 128':>9}"
    print(hdr)
    print("-" * len(hdr))
    ref = None
    for e in (128, 512, 2048):
        runs = [train(e, s) for s in (0, 1)]
        avg = {m: sum(r[m] for r in runs) / len(runs) for m in MARKS}
        if ref is None:
            ref = avg[max(MARKS)]
        print(f"{e:>9}{e // K:>7}x"
              + "".join(f"{avg[m]:>11.5f}" for m in MARKS)
              + f"{avg[max(MARKS)] / ref:>8.2f}x")

    print("\n  A pool that keeps pulling ahead at the longest horizon is real")
    print("  capacity. Arms that converge together mean the extra experts")
    print("  bought nothing and the table is oversized for this data.")

    print("\nSHARDING at the same converged horizon. The earlier sharding")
    print("result was taken at the floor and is void; this is the retest.")
    print(f"\n{'experts':>9}{'shards':>8}{'reachable':>11}"
          f"{'@5400':>11}{'vs full':>9}")
    print("-" * 48)
    for e in (512, 2048):
        full = [train(e, s, shards=1) for s in (0, 1)]
        sh = [train(e, s, shards=2) for s in (0, 1)]
        fa = sum(r[max(MARKS)] for r in full) / len(full)
        sa = sum(r[max(MARKS)] for r in sh) / len(sh)
        print(f"{e:>9}{1:>8}{e:>11}{fa:>11.5f}{1.0:>8.2f}x")
        print(f"{e:>9}{2:>8}{e // 2:>11}{sa:>11.5f}{sa / fa:>8.2f}x")


if __name__ == "__main__":
    main()
