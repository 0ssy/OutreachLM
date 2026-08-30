from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

from .accounting import deep_sizeof_bytes
from .eviction import choose_eviction_keys


@dataclass
class TopologyGuardResult:
    accepted: bool
    evicted: int
    halted: bool


class TopologyGuard:
    def __init__(self, max_nodes: int, strategy: str) -> None:
        self.max_nodes = max_nodes
        self.strategy = strategy
        self.nodes: dict[Hashable, dict[str, Any]] = {}
        self.halted = False

    def observe(
        self,
        context_key: Hashable,
        next_token: int,
        *,
        timestamp: int,
        protected: bool = False,
    ) -> TopologyGuardResult:
        if self.halted:
            return TopologyGuardResult(accepted=False, evicted=0, halted=True)

        node = self.nodes.get(context_key)
        if node is None:
            node = {
                "total_count": 0.0,
                "last_access": timestamp,
                "protected": protected,
                "next_counts": {},
            }
            self.nodes[context_key] = node

        node["total_count"] += 1.0
        node["last_access"] = timestamp
        node["protected"] = bool(node["protected"] or protected)
        next_counts: dict[int, float] = node["next_counts"]
        next_counts[next_token] = next_counts.get(next_token, 0.0) + 1.0

        evicted = 0
        if len(self.nodes) > self.max_nodes:
            keys = choose_eviction_keys(self.nodes, max_nodes=self.max_nodes, strategy=self.strategy)
            if self.strategy == "none" and not keys:
                self.halted = True
                return TopologyGuardResult(accepted=False, evicted=0, halted=True)
            for key in keys:
                self.nodes.pop(key, None)
            evicted = len(keys)
            if len(self.nodes) > self.max_nodes:
                self.halted = True
                return TopologyGuardResult(accepted=False, evicted=evicted, halted=True)
        return TopologyGuardResult(accepted=True, evicted=evicted, halted=False)

    def distribution(self, context_key: Hashable, *, vocab_size: int, alpha: float) -> list[float]:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        node = self.nodes.get(context_key)
        if node is None:
            uniform = 1.0 / vocab_size
            return [uniform] * vocab_size
        counts: dict[int, float] = node["next_counts"]
        total = sum(counts.values())
        denominator = total + (alpha * vocab_size)
        out = [alpha / denominator] * vocab_size
        for token_id, count in counts.items():
            if 0 <= token_id < vocab_size:
                out[token_id] = (count + alpha) / denominator
        return out

    def logical_size_bytes(self) -> int:
        return deep_sizeof_bytes(self.nodes)

