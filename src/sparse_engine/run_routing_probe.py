"""Run the routing-quality probe: does extreme sparsity hold up?"""
from __future__ import annotations

import torch

from src.sparse_engine.routing_probe import run

torch.set_num_threads(6)
SEEDS = (0, 1, 2)
K = 2
CONCEPTS = 32


def agg(**kw):
    rs = [run(seed=s, **kw) for s in SEEDS]
    return {
        "relative": sum(r["relative"] for r in rs) / len(rs),
        "dead_frac": sum(r["dead_frac"] for r in rs) / len(rs),
        "used": sum(r["experts_used"] for r in rs) / len(rs),
    }


def main() -> None:
    print(f"Routing probe: {CONCEPTS} latent concepts, k={K} active experts")
    print("per token in EVERY arm, so compute is identical and only the")
    print("pool size varies. 3 seeds. 'relative' is loss / loss-of-zeros.\n")

    print("A. SPARSITY RATIO -- pool grows, active count fixed")
    print(f"{'experts':>9}{'ratio':>9}{'relative loss':>16}"
          f"{'experts used':>15}{'dead':>8}")
    print("-" * 58)
    base = None
    for e in (32, 64, 128, 256, 512, 1024):
        r = agg(n_experts=e, k=K)
        if base is None:
            base = r["relative"]
        print(f"{e:>9}{e / K:>8.0f}x{r['relative']:>16.4f}"
              f"{r['used']:>15.0f}{r['dead_frac']:>7.0%}")

    print("\nB. SHARDING -- same pool, but each token may only route within")
    print("   its own half (the two-laptop expert-ownership design)")
    print(f"{'experts':>9}{'shards':>8}{'reachable':>11}"
          f"{'relative loss':>16}{'vs full':>10}")
    print("-" * 55)
    for e in (256, 512, 1024):
        full = agg(n_experts=e, k=K)
        sh = agg(n_experts=e, k=K, shards=2)
        print(f"{e:>9}{1:>8}{e:>11}{full['relative']:>16.4f}"
              f"{1.0:>9.2f}x")
        print(f"{e:>9}{2:>8}{e // 2:>11}{sh['relative']:>16.4f}"
              f"{sh['relative'] / full['relative']:>9.2f}x")


if __name__ == "__main__":
    main()
