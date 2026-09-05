"""Rung 3g -- two audits the LR-selection defect makes mandatory.

AUDIT 1: IS THERE A FLOOR IN USABLE LEARNING RATE?
    Rung 3f showed L and M match dense exactly at a MATCHED lr, concluding
    "no floor". That conclusion is incomplete. If dense tolerates a larger step
    *because* it carries no estimator noise, then L/M have a floor in usable
    lr, which becomes a floor in steps-to-target -- the same gap relocated from
    final error to training speed. Measured directly: max stable lr per method,
    and steps to reach a fixed error target at each method's OWN best lr.

AUDIT 2: METHOD G's HEADLINE, RE-CHECKED
    The original F/G sweep used a single fixed lr=0.5 for EVERY arm, never
    tuned per method (`run_true_restricted.py`, `--lr` default 0.5). That is a
    stronger version of the defect just found in rung 3f. It matters
    specifically for G because error feedback ACCUMULATES residual and then
    applies it, so G's effective step size is larger than its nominal lr. At a
    shared lr, G would appear to beat dense purely by taking bigger steps.
    Re-run with per-method tuning to see whether "G 0.0247 beats dense 0.0493"
    survives.
"""
from __future__ import annotations

import torch

from src.restricted_backward.methods import MethodA
from src.restricted_backward.harness import make_task
from src.restricted_backward.parameter_keyed import ParameterKeyedAccumulation
from src.restricted_backward.run_rung3f_validation import build, sample_ids
from src.restricted_backward.true_restricted import MethodF

D = 512
V = 256
N = 32
SEEDS = (0, 1, 2)
LR_LADDER = [6.0, 20.0, 60.0, 200.0, 600.0, 2000.0, 6000.0, 20000.0]

# Audit 2 reproduces the ORIGINAL F/G conditions, which used a much larger
# layer; using the rung-3f scale here would not be a reproduction.
D_O = 4096
N_O = 32
SEEDS_O = (0, 1)


# ---------------------------------------------------------------- audit 1

def process_i_run(kind, seed, lr, steps, target=None):
    """Returns (final_recovery, steps_to_target). W frozen throughout."""
    m = build(kind, seed)
    g = torch.Generator().manual_seed(seed)
    E_star = torch.randn(V, D, generator=g) * 0.3
    tgt = E_star @ m.W.T
    E = torch.randn(V, D, generator=g) * 0.05
    gen = torch.Generator().manual_seed(seed + 4242)
    hit = None
    for t in range(steps):
        ids = sample_ids("uniform", t, gen)
        X = E[ids]
        n = X.shape[0]
        G_Y = (2.0 / (n * D)) * (m.forward(X) - tgt[ids])
        if kind == "K2":
            buckets, *_ = m.backward(G_Y, ids)
            est = m.densify(buckets, (n, D))
        elif kind == "M":
            est, *_ = m.backward(G_Y, ids)
        else:
            packed, *_ = m.backward(G_Y, X)
            if isinstance(packed, tuple):
                idx, vals = packed
                est = torch.zeros(n, D)
                est[:, idx] = vals
            else:
                est = packed
        E = E.index_add(0, ids, -lr * est)
        if not torch.isfinite(E).all():
            return float("inf"), None
        if hit is None and target is not None and (t + 1) % 50 == 0:
            if float((E - E_star).norm() / E_star.norm()) <= target:
                hit = t + 1
    return float((E - E_star).norm() / E_star.norm()), hit


def audit_usable_lr():
    print("AUDIT 1 -- is the dense advantage a floor in USABLE LEARNING RATE?")
    print("uniform access, W frozen, 3 seeds, 800 steps\n")
    print(f"{'method':>8}{'max stable lr':>15}{'best lr':>10}"
          f"{'recovery @ best':>17}")
    print("-" * 50)
    best_of = {}
    for kind in ("dense", "L", "M", "K2"):
        max_stable, best_lr, best_val = None, None, float("inf")
        for lr in LR_LADDER:
            vals = [process_i_run(kind, sd, lr, 800)[0] for sd in SEEDS]
            if any(v == float("inf") for v in vals):
                continue
            max_stable = lr
            mean = sum(vals) / len(vals)
            if mean < best_val:
                best_val, best_lr = mean, lr
        best_of[kind] = (best_lr, best_val)
        ms = "none" if max_stable is None else f"{max_stable:.0f}"
        print(f"{kind:>8}{ms:>15}{best_lr:>10.0f}{best_val:>17.4f}")

    print("\n  If L/M's max stable lr equals dense's, there is no lr floor and")
    print("  'indistinguishable' is fair. If it is lower, the rung 3f framing")
    print("  is misleading and the gap is real but relocated to step count.\n")

    target = 0.15
    print(f"AUDIT 1b -- steps to reach recovery <= {target}, each at its OWN "
          f"best lr")
    print(f"{'method':>8}{'best lr':>10}{'steps to target':>18}")
    print("-" * 36)
    for kind in ("dense", "L", "M", "K2"):
        lr = best_of[kind][0]
        hits = [process_i_run(kind, sd, lr, 3000, target)[1] for sd in SEEDS]
        if any(h is None for h in hits):
            print(f"{kind:>8}{lr:>10.0f}{'not reached':>18}")
        else:
            print(f"{kind:>8}{lr:>10.0f}{sum(hits) / len(hits):>18.0f}")


# ---------------------------------------------------------------- audit 2

def process_o_run(kind, seed, lr, steps, keep=0.05, target=None):
    """Original F/G task and scale. Returns (final_loss, steps_to_target).

    Uses `make_task` (a realizable target Y = X W_true^T) at the original
    d=4096, matching the conditions the 0.0247-vs-0.0493 claim was made under.
    """
    layer_seed = seed
    X, Y_true = make_task(N_O, D_O, D_O, 1337 + seed)
    if kind == "dense":
        layer = MethodA(D_O, D_O)
    else:
        layer = MethodF(D_O, D_O, keep_fraction=keep,
                        error_feedback=(kind == "G"), seed=layer_seed)
    hit = None
    for t in range(steps):
        Y = layer.forward(X)
        G_Y = (2.0 / (N_O * D_O)) * (Y - Y_true)
        if kind == "dense":
            dW, *_ = layer.backward(G_Y, X)
            layer.apply(dW, lr)
        else:
            (idx, dWp), *_ = layer.backward(G_Y, X)
            layer.apply((idx, dWp), lr)
        if not torch.isfinite(layer.W).all():
            return float("inf"), None
        if hit is None and target is not None:
            cur = float(((layer.forward(X) - Y_true) ** 2).mean())
            if cur <= target:
                hit = t + 1
    return float(((layer.forward(X) - Y_true) ** 2).mean()), hit


def audit_method_g():
    print("\n\nAUDIT 2 -- Method G's headline under per-method lr tuning")
    print(f"original conditions: make_task, d={D_O}, single shared lr=0.5.")
    print("G's error feedback banks residual then applies it, so its")
    print("EFFECTIVE step exceeds its nominal lr -- at a shared lr that alone")
    print("could produce the reported advantage.\n")
    steps = 200
    ladder = [0.05, 0.15, 0.5, 1.5, 5.0, 15.0]
    print(f"{'lr':>8}{'dense':>14}{'F (no EF)':>14}{'G (EF)':>14}")
    print("-" * 50)
    best = {k: (None, float("inf")) for k in ("dense", "F", "G")}
    for lr in ladder:
        cells = []
        for kind in ("dense", "F", "G"):
            vals = [process_o_run(kind, sd, lr, steps)[0] for sd in SEEDS_O]
            mean = (float("inf") if any(v == float("inf") for v in vals)
                    else sum(vals) / len(vals))
            if mean < best[kind][1]:
                best[kind] = (lr, mean)
            cells.append("diverged" if mean == float("inf") else f"{mean:.5f}")
        print(f"{lr:>8.2f}" + "".join(f"{c:>14}" for c in cells))

    print(f"\n{'method':>8}{'best lr':>10}{'best loss':>13}"
          f"{'max stable lr':>16}")
    print("-" * 47)
    for kind in ("dense", "F", "G"):
        stable = [lr for lr in ladder
                  if all(process_o_run(kind, sd, lr, steps)[0] != float("inf")
                         for sd in SEEDS_O)]
        ms = "none" if not stable else f"{max(stable):.2f}"
        print(f"{kind:>8}{best[kind][0]:>10.2f}{best[kind][1]:>13.5f}{ms:>16}")

    print("\n  original claim, shared lr=0.5: G 0.0247 vs dense 0.0493")
    print("  the claim survives only if G still wins when BOTH are tuned.")


def audit_time_to_target():
    """The economic question audit 1 forces: per-step savings vs step count.

    A method that costs 0.48x per step but needs 3.5x more steps is a NET
    LOSS. Rung 3f reported per-step cost only, which is the same category of
    incomplete comparison as reporting FLOPs without wall-clock. Measured end
    to end rather than composed from two separate numbers.
    """
    import time as _t

    target = 0.15
    print("\n\nAUDIT 3 -- wall-clock TO TARGET, not per step")
    print(f"uniform access, W frozen, target recovery <= {target}, "
          f"each at its OWN best lr, 3 seeds\n")
    print(f"{'method':>8}{'best lr':>9}{'steps':>8}{'s/step':>10}"
          f"{'total s':>10}{'vs dense':>10}")
    print("-" * 55)

    best_lr = {"dense": 2000.0, "L": 600.0, "M": 600.0, "K2": 60.0}
    base = None
    for kind in ("dense", "L", "M", "K2"):
        lr = best_lr[kind]
        totals, stepcounts = [], []
        for sd in SEEDS:
            _, hit = process_i_run(kind, sd, lr, 4000, target)
            if hit is None:
                totals = None
                break
            t1 = _t.perf_counter()
            process_i_run(kind, sd, lr, hit)
            totals.append(_t.perf_counter() - t1)
            stepcounts.append(hit)
        if totals is None:
            print(f"{kind:>8}{lr:>9.0f}{'not reached':>19}")
            continue
        steps = sum(stepcounts) / len(stepcounts)
        tot = sum(totals) / len(totals)
        if base is None:
            base = tot
        print(f"{kind:>8}{lr:>9.0f}{steps:>8.0f}{tot / steps:>10.4f}"
              f"{tot:>10.2f}{tot / base:>9.2f}x")

    print("\n  A per-step saving is only a saving if step count holds.")


def main() -> None:
    torch.set_num_threads(6)
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        audit_usable_lr()
    if which in ("all", "2"):
        audit_method_g()
    if which in ("all", "3"):
        audit_time_to_target()


if __name__ == "__main__":
    main()
