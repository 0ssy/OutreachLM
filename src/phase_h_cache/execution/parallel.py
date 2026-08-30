from __future__ import annotations

import os
import time
from typing import Callable

import numpy as np


def set_thread_env(core_count: int) -> None:
    if core_count <= 0:
        raise ValueError("core_count must be > 0")
    value = str(core_count)
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value


def benchmark_inference(
    infer_fn: Callable[[], np.ndarray],
    *,
    steps: int,
) -> dict[str, float]:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    latencies_us: list[float] = []
    begin = time.perf_counter()
    for _ in range(steps):
        t0 = time.perf_counter()
        _ = infer_fn()
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000.0)
    elapsed = time.perf_counter() - begin
    throughput = steps / max(elapsed, 1e-12)
    return {
        "steps": float(steps),
        "elapsed_seconds": elapsed,
        "tokens_per_second": throughput,
        "latency_mean_us": float(np.mean(np.asarray(latencies_us, dtype=np.float64))),
        "latency_std_us": float(np.std(np.asarray(latencies_us, dtype=np.float64))),
    }

