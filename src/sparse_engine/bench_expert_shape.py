"""Where does block-sparse expert routing keep CPU throughput?

This decides whether a 70B-capacity, sparsely-activated model is trainable on
this machine, so it is measured before anything is designed around it.

An earlier check measured 0.1% UNSTRUCTURED sparsity at 201x speedup but only
~12 GFLOP/s effective -- the active submatrix was too small to use the
machine. Dense peak here is ~130 GFLOP/s. So the question is not "does
sparsity help" (it does) but "how large must an expert tile be before the
sparse path still runs near roofline".

That number sets the minimum expert size, which sets the expert count for a
given active-parameter budget, which sets the whole architecture.
"""
from __future__ import annotations

import time

import torch

torch.set_num_threads(6)
PEAK = 130.5


def best(fn, reps, trials=4):
    fn()
    b = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        b = min(b, (time.perf_counter() - t0) / reps)
    return b


def main() -> None:
    print("Block-sparse expert throughput. An 'expert' is one FFN tile")
    print(f"d_model x d_ff applied to its routed tokens. Dense peak {PEAK} "
          f"GFLOP/s.\n")

    d_model = 2048
    print(f"{'d_ff':>7}{'tokens':>8}{'tern MB':>10}{'ms':>9}"
          f"{'GFLOP/s':>10}{'of peak':>10}")
    print("-" * 54)
    for d_ff in (256, 512, 1024, 2048, 8192):
        for n in (64, 256, 1024):
            g = torch.Generator().manual_seed(0)
            W = torch.randn(d_model, d_ff, generator=g)
            X = torch.randn(n, d_model, generator=g)
            work = n * d_model * d_ff
            reps = max(3, min(200, 40_000_000 // max(1, work // 1000)))
            t = best(lambda: X @ W, reps, trials=3)
            gf = 2.0 * work / t / 1e9
            mb = d_model * d_ff * 2 / 8 / 1e6
            print(f"{d_ff:>7}{n:>8}{mb:>10.2f}{t * 1000:>9.3f}"
                  f"{gf:>10.1f}{gf / PEAK:>9.0%}")

    print("\nGATHER OVERHEAD -- routed tokens are scattered in the batch and")
    print("must be collected before the expert GEMM can run.\n")
    d_ff = 2048
    g = torch.Generator().manual_seed(1)
    W = torch.randn(d_model, d_ff, generator=g)
    big = torch.randn(8192, d_model, generator=g)
    print(f"{'tokens':>8}{'gather ms':>12}{'gemm ms':>10}{'overhead':>11}")
    print("-" * 41)
    for n in (64, 256, 1024):
        idx = torch.randperm(8192, generator=g)[:n]
        gt = best(lambda: big[idx], 200)
        X = big[idx].contiguous()
        mt = best(lambda: X @ W, 50)
        print(f"{n:>8}{gt * 1000:>12.3f}{mt * 1000:>10.3f}{gt / mt:>10.0%}")


if __name__ == "__main__":
    main()
