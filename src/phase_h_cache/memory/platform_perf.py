from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import psutil

from .accounting import deep_sizeof_bytes, process_memory_bytes


@dataclass(frozen=True)
class LatencySample:
    elapsed_seconds: float
    latency_us: float
    cycles_per_token_estimate: float


def estimate_cycles_per_token(latency_us: float) -> float:
    frequency = psutil.cpu_freq()
    if frequency is None or frequency.current <= 0:
        return 0.0
    return float(latency_us * frequency.current)


def benchmark_steps(step_fn: Callable[[], Any], steps: int) -> LatencySample:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    t0 = perf_counter()
    for _ in range(steps):
        step_fn()
    elapsed = perf_counter() - t0
    latency_us = (elapsed / steps) * 1_000_000.0
    return LatencySample(
        elapsed_seconds=elapsed,
        latency_us=latency_us,
        cycles_per_token_estimate=estimate_cycles_per_token(latency_us),
    )


def build_padding_state(target_bytes: int, chunk_bytes: int = 1024) -> dict[str, bytes]:
    if target_bytes <= 0:
        return {}
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be > 0")
    chunks = max(1, target_bytes // chunk_bytes)
    payload = b"x" * chunk_bytes
    return {f"pad_{idx}": payload for idx in range(chunks)}


def size_and_memory(obj: Any) -> dict[str, int]:
    mem = process_memory_bytes()
    return {
        "logical_bytes": int(deep_sizeof_bytes(obj)),
        "rss_bytes": int(mem["rss_bytes"]),
        "vms_bytes": int(mem["vms_bytes"]),
    }
