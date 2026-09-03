"""Phase 2 instrumentation harness + Phases 4-6 sweep runner.

Separation enforced throughout:
  theoretical_flops : Phase 0 paper math for a DENSE step
  executed_flops    : work an idealized sparse kernel would actually do
  wall_clock        : measured on this CPU
These three are allowed to diverge, and that divergence is the result.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.restricted_backward.methods import (
    MethodA,
    MethodB,
    SparseConfig,
    SparseMethod,
    dense_flops,
)

RESULTS = ROOT / "experiments" / "restricted_backward" / "results"


def make_task(n: int, d_in: int, d_out: int, seed: int):
    """Fixed input/target distribution representative of an FFN projection."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g)
    W_true = torch.randn(d_out, d_in, generator=g) / d_in**0.5
    Y_true = X @ W_true.T
    return X, Y_true


def rho_slope(values: list[float], window: int) -> float:
    """Least-squares slope of the last `window` residual ratios."""
    tail = values[-window:]
    if len(tail) < 2:
        return 0.0
    n = len(tail)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(tail) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, tail))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def run_config(
    method_name: str,
    config_name: str,
    layer,
    X: torch.Tensor,
    Y_true: torch.Tensor,
    *,
    steps: int,
    lr: float,
    rho_window: int,
    keep_trajectory: bool,
) -> tuple[dict, list[dict]]:
    n, d_in = X.shape
    d_out = Y_true.shape[1]
    theo = dense_flops(n, d_in, d_out)

    rows: list[dict] = []
    losses: list[float] = []
    grad_errors: list[float] = []
    rhos: list[float] = []
    exec_totals: list[float] = []
    step_times: list[float] = []

    for step in range(1, steps + 1):
        t_f0 = time.perf_counter()
        Y = layer.forward(X)
        forward_time = time.perf_counter() - t_f0

        diff = Y - Y_true
        loss = float((diff**2).mean())
        G_Y = (2.0 / (n * d_out)) * diff

        t_b0 = time.perf_counter()
        applied, dX, exact_dW, exec_dw_flops, sel_s, fb_s = layer.backward(G_Y, X)
        backward_time = time.perf_counter() - t_b0 - sel_s - fb_s

        t_u0 = time.perf_counter()
        layer.apply(applied, lr)
        update_time = time.perf_counter() - t_u0

        # Gradient error vs the exact dense gradient. Undefined for B, whose
        # parameters are (B, A) rather than W -- reported as null, not faked.
        if exact_dW is not None and isinstance(applied, torch.Tensor):
            denom = float(exact_dW.norm())
            grad_error = float((applied - exact_dW).norm() / denom) if denom > 0 else 0.0
        else:
            grad_error = None

        residual_ratio = None
        if getattr(layer, "residual", None) is not None and exact_dW is not None:
            gn = float(exact_dW.norm())
            residual_ratio = float(layer.residual.norm() / gn) if gn > 0 else 0.0
            rhos.append(residual_ratio)

        executed_total = theo["forward"] + theo["dX"] + exec_dw_flops
        step_time = forward_time + backward_time + sel_s + fb_s + update_time

        losses.append(loss)
        if grad_error is not None:
            grad_errors.append(grad_error)
        exec_totals.append(executed_total)
        step_times.append(step_time)

        if keep_trajectory:
            rows.append(
                {
                    "step": step,
                    "method": method_name,
                    "config": config_name,
                    "loss": round(loss, 6),
                    "grad_error": None if grad_error is None else round(grad_error, 6),
                    "residual_ratio": None
                    if residual_ratio is None
                    else round(residual_ratio, 6),
                    "forward_time": round(forward_time, 6),
                    "backward_time": round(backward_time, 6),
                    "selection_time": round(sel_s, 6),
                    "feedback_time": round(fb_s, 6),
                    "update_time": round(update_time, 6),
                    "theoretical_flops": theo["total"],
                    "executed_flops": executed_total,
                    "peak_memory": layer.param_bytes() + layer.aux_bytes,
                }
            )

    slope = rho_slope(rhos, rho_window) if rhos else None
    summary = {
        "method": method_name,
        "config": config_name,
        "final_loss": round(losses[-1], 6),
        "mean_grad_error": round(statistics.fmean(grad_errors), 6) if grad_errors else None,
        "rho_slope": None if slope is None else round(slope, 8),
        "rho_slope_flag": bool(slope is not None and slope > 0),
        "theoretical_flops_per_step": theo["total"],
        "mean_executed_flops_per_step": statistics.fmean(exec_totals),
        "mean_step_seconds": statistics.fmean(step_times),
        "peak_memory": layer.param_bytes() + layer.aux_bytes,
    }
    return summary, rows


def build_layer(kind: str, d_in: int, d_out: int, **kw):
    if kind == "A":
        return MethodA(d_in, d_out)
    if kind == "B":
        return MethodB(d_in, d_out, rank=kw["rank"])
    return SparseMethod(
        d_in,
        d_out,
        config=SparseConfig(
            keep_fraction=kw["keep"],
            error_feedback=kw.get("ef", False),
            block_size=kw.get("block"),
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--rho-window", type=int, default=20)  # fixed BEFORE running
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--trajectory", type=str, default="")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    d = args.dim
    X, Y_true = make_task(args.batch, d, d, args.seed)

    specs: list[tuple[str, str, dict]] = [("A", "dense", {})]
    for r in (16, 32, 64, 128, 256):
        specs.append(("B", f"r{r}", {"rank": r}))
    keeps = [(0.01, "k1pct"), (0.02, "k2pct"), (0.05, "k5pct"), (0.10, "k10pct"), (0.25, "k25pct")]
    for keep, label in keeps:
        specs.append(("C", label, {"keep": keep}))
    for keep, label in keeps:
        specs.append(("D", label, {"keep": keep, "ef": True}))
    for keep, label in keeps:
        for bs in (16, 64):
            specs.append(("E", f"{label}_b{bs}", {"keep": keep, "ef": True, "block": bs}))

    print(f"# dim={d} batch={args.batch} steps={args.steps} lr={args.lr} "
          f"threads={torch.get_num_threads()} rho_window={args.rho_window}", flush=True)

    summaries: list[dict] = []
    trajectories: list[dict] = []
    for kind, label, kw in specs:
        layer = build_layer(kind, d, d, **kw)
        want_traj = f"{kind}:{label}" == args.trajectory
        summary, rows = run_config(
            kind, label, layer, X, Y_true,
            steps=args.steps, lr=args.lr,
            rho_window=args.rho_window, keep_trajectory=want_traj,
        )
        summaries.append(summary)
        trajectories.extend(rows)
        print(json.dumps(summary), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    baseline = next(s for s in summaries if s["method"] == "A")
    for s in summaries:
        s["total_flops_saved_pct"] = round(
            100.0 * (1 - s["mean_executed_flops_per_step"] / baseline["mean_executed_flops_per_step"]), 2
        )
        s["total_wallclock_saved_pct"] = round(
            100.0 * (1 - s["mean_step_seconds"] / baseline["mean_step_seconds"]), 2
        )
    (RESULTS / "summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if trajectories:
        (RESULTS / "trajectory.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    print("# wrote", RESULTS / "summaries.json", flush=True)


if __name__ == "__main__":
    main()
