"""Runner for Method F/G (true restricted backward) + Method B replicates.

Two things the first sweep got wrong, addressed here:

1. C/D/E were GEMM-then-select (post-hoc compression). F/G restrict the GEMM
   itself. Reported side by side so the distinction is visible in the data.
2. B's wall-clock was a single run per config, and the savings were
   non-monotonic in r. Replicated here so noise can be separated from signal.
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

from src.restricted_backward.harness import make_task, rho_slope
from src.restricted_backward.methods import MethodA, MethodB, dense_flops
from src.restricted_backward.true_restricted import MethodF

RESULTS = ROOT / "experiments" / "restricted_backward" / "results"


def run_dense(X, Y_true, steps, lr):
    n, d = X.shape
    layer = MethodA(d, d)
    theo = dense_flops(n, d, d)
    losses, times = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        Y = layer.forward(X)
        diff = Y - Y_true
        loss = float((diff**2).mean())
        G_Y = (2.0 / (n * d)) * diff
        dW, dX, _, _, _, _ = layer.backward(G_Y, X)
        layer.apply(dW, lr)
        times.append(time.perf_counter() - t0)
        losses.append(loss)
    return {
        "method": "A",
        "config": "dense",
        "final_loss": round(losses[-1], 6),
        "mean_grad_error": 0.0,
        "rho_slope": None,
        "rho_slope_flag": False,
        "theoretical_flops_per_step": theo["total"],
        "mean_executed_flops_per_step": theo["total"],
        "mean_step_seconds": statistics.fmean(times),
        "peak_memory": layer.param_bytes(),
        "max_backward_tensor_bytes": d * d * 4,
        "restriction_locus": "none",
    }


def run_restricted(X, Y_true, steps, lr, keep, ef, rho_window, measure_error):
    n, d = X.shape
    layer = MethodF(d, d, keep_fraction=keep, error_feedback=ef)
    theo = dense_flops(n, d, d)
    losses, errors, rhos, times, execs = [], [], [], [], []

    for _ in range(steps):
        t0 = time.perf_counter()
        Y = layer.forward(X)
        fwd_s = time.perf_counter() - t0

        diff = Y - Y_true
        loss = float((diff**2).mean())
        G_Y = (2.0 / (n * d)) * diff

        (idx, dWp), dX, executed, sel_s, fb_s, bwd_s = layer.backward(G_Y, X)

        t_u0 = time.perf_counter()
        layer.apply((idx, dWp), lr)
        upd_s = time.perf_counter() - t_u0

        # Oracle for error metrics only -- explicitly OUTSIDE the timed path.
        if measure_error:
            exact = layer.exact_dW(G_Y, X)
            approx = torch.zeros_like(exact)
            approx[idx] = dWp
            den = float(exact.norm())
            errors.append(float((approx - exact).norm() / den) if den > 0 else 0.0)
            if ef and layer.residual is not None:
                gn = float(G_Y.norm())
                rhos.append(float(layer.residual.norm() / gn) if gn > 0 else 0.0)

        times.append(fwd_s + sel_s + fb_s + bwd_s + upd_s)
        execs.append(theo["forward"] + theo["dX"] + executed)
        losses.append(loss)

    slope = rho_slope(rhos, rho_window) if rhos else None
    return {
        "method": layer.name,
        "config": f"k{int(keep*100)}pct" if keep >= 0.01 else f"k{keep}",
        "final_loss": round(losses[-1], 6),
        "mean_grad_error": round(statistics.fmean(errors), 6) if errors else None,
        "rho_slope": None if slope is None else round(slope, 8),
        "rho_slope_flag": bool(slope is not None and slope > 0),
        "theoretical_flops_per_step": theo["total"],
        "mean_executed_flops_per_step": statistics.fmean(execs),
        "mean_step_seconds": statistics.fmean(times),
        "peak_memory": layer.param_bytes() + layer.aux_bytes,
        "max_backward_tensor_bytes": layer.max_backward_tensor_bytes,
        "restriction_locus": "pre-GEMM (upstream gradient)",
    }


def run_lowrank(X, Y_true, steps, lr, rank):
    n, d = X.shape
    layer = MethodB(d, d, rank=rank)
    theo = dense_flops(n, d, d)
    losses, times, execs = [], [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        Y = layer.forward(X)
        diff = Y - Y_true
        loss = float((diff**2).mean())
        G_Y = (2.0 / (n * d)) * diff
        grads, dX, _, flops, _, _ = layer.backward(G_Y, X)
        layer.apply(grads, lr)
        times.append(time.perf_counter() - t0)
        execs.append(theo["forward"] + theo["dX"] + flops)
        losses.append(loss)
    return {
        "method": "B",
        "config": f"r{rank}",
        "final_loss": round(losses[-1], 6),
        "mean_grad_error": None,
        "rho_slope": None,
        "rho_slope_flag": False,
        "theoretical_flops_per_step": theo["total"],
        "mean_executed_flops_per_step": statistics.fmean(execs),
        "mean_step_seconds": statistics.fmean(times),
        "peak_memory": layer.param_bytes(),
        "max_backward_tensor_bytes": max(d * rank, n * d) * 4,
        "restriction_locus": "architectural (low-rank)",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=4096)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--rho-window", type=int, default=20)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--replicates", type=int, default=3)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    d = args.dim
    keeps = [0.01, 0.02, 0.05, 0.10, 0.25]

    print(f"# dim={d} batch={args.batch} steps={args.steps} lr={args.lr} "
          f"threads={torch.get_num_threads()} replicates={args.replicates}", flush=True)

    all_rows: list[dict] = []
    replicate_times: dict[str, list[float]] = {}

    for rep in range(args.replicates):
        torch.manual_seed(args.seed + rep)
        X, Y_true = make_task(args.batch, d, d, args.seed + rep)

        specs = [("A", "dense", run_dense, {})]
        for r in (16, 32, 64, 128, 256):
            specs.append(("B", f"r{r}", run_lowrank, {"rank": r}))
        for k in keeps:
            specs.append(("F", f"k{int(k*100)}pct", run_restricted,
                          {"keep": k, "ef": False}))
        for k in keeps:
            specs.append(("G", f"k{int(k*100)}pct", run_restricted,
                          {"keep": k, "ef": True}))

        for kind, label, fn, kw in specs:
            if fn is run_restricted:
                row = fn(X, Y_true, args.steps, args.lr,
                         kw["keep"], kw["ef"], args.rho_window,
                         measure_error=(rep == 0))
            elif fn is run_lowrank:
                row = fn(X, Y_true, args.steps, args.lr, kw["rank"])
            else:
                row = fn(X, Y_true, args.steps, args.lr)
            key = f"{row['method']}:{row['config']}"
            replicate_times.setdefault(key, []).append(row["mean_step_seconds"])
            if rep == 0:
                all_rows.append(row)

    base = statistics.fmean(replicate_times["A:dense"])
    base_flops = next(r for r in all_rows if r["method"] == "A")["mean_executed_flops_per_step"]

    for row in all_rows:
        key = f"{row['method']}:{row['config']}"
        ts = replicate_times[key]
        row["mean_step_seconds"] = round(statistics.fmean(ts), 6)
        row["step_seconds_stdev"] = round(statistics.stdev(ts), 6) if len(ts) > 1 else 0.0
        row["replicates"] = len(ts)
        row["total_flops_saved_pct"] = round(
            100.0 * (1 - row["mean_executed_flops_per_step"] / base_flops), 2)
        row["total_wallclock_saved_pct"] = round(
            100.0 * (1 - row["mean_step_seconds"] / base), 2)
        print(json.dumps(row), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "true_restricted_summaries.json").write_text(
        json.dumps(all_rows, indent=2), encoding="utf-8")
    print("# wrote", RESULTS / "true_restricted_summaries.json", flush=True)


if __name__ == "__main__":
    main()
