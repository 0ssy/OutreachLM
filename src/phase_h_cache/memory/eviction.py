from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NodeStats:
    total_count: float
    last_access: int
    protected: bool


def _evict_frequency(nodes: dict[Any, dict[str, Any]], n_remove: int) -> list[Any]:
    ranked = sorted(
        ((key, float(value["total_count"])) for key, value in nodes.items() if not value["protected"]),
        key=lambda item: item[1],
    )
    return [key for key, _ in ranked[:n_remove]]


def _evict_lru(nodes: dict[Any, dict[str, Any]], n_remove: int) -> list[Any]:
    ranked = sorted(
        ((key, int(value["last_access"])) for key, value in nodes.items() if not value["protected"]),
        key=lambda item: item[1],
    )
    return [key for key, _ in ranked[:n_remove]]


def _evict_utility(nodes: dict[Any, dict[str, Any]], n_remove: int) -> list[Any]:
    entries = [(key, value) for key, value in nodes.items() if not value["protected"]]
    if not entries:
        return []

    min_count = min(float(value["total_count"]) for _, value in entries)
    max_count = max(float(value["total_count"]) for _, value in entries)
    min_age = min(int(value["last_access"]) for _, value in entries)
    max_age = max(int(value["last_access"]) for _, value in entries)

    def norm(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return (x - lo) / (hi - lo)

    scored: list[tuple[Any, float]] = []
    for key, value in entries:
        count_score = norm(float(value["total_count"]), min_count, max_count)
        recency_score = norm(float(value["last_access"]), min_age, max_age)
        utility = (0.65 * count_score) + (0.35 * recency_score)
        scored.append((key, utility))
    scored.sort(key=lambda item: item[1])
    return [key for key, _ in scored[:n_remove]]


def choose_eviction_keys(
    nodes: dict[Any, dict[str, Any]],
    *,
    max_nodes: int,
    strategy: str,
) -> list[Any]:
    overflow = len(nodes) - max_nodes
    if overflow <= 0:
        return []

    if strategy == "frequency":
        return _evict_frequency(nodes, overflow)
    if strategy == "lru":
        return _evict_lru(nodes, overflow)
    if strategy == "utility":
        return _evict_utility(nodes, overflow)
    if strategy == "none":
        return []
    raise ValueError(f"Unknown eviction strategy: {strategy}")

