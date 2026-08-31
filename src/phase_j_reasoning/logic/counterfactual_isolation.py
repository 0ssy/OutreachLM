from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Hashable


def _freeze_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _freeze_for_hash(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_freeze_for_hash(item) for item in value]
    if isinstance(value, set):
        return sorted(_freeze_for_hash(item) for item in value)
    return value


def fingerprint_factual_manifold(manifold: dict[Hashable, Any]) -> str:
    serialized = json.dumps(_freeze_for_hash(manifold), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class VirtualCoWManifold:
    """Isolate speculative counterfactual writes from the immutable factual manifold."""

    def __init__(self, factual_core_manifold: dict[Hashable, Any] | None = None) -> None:
        self.world_0: dict[Hashable, Any] = deepcopy(factual_core_manifold or {})
        self.world_1: dict[Hashable, dict[Hashable, float]] = {}

    def read_transition_mass(self, source_id: Hashable, target_id: Hashable) -> float:
        if source_id in self.world_1 and target_id in self.world_1[source_id]:
            return float(self.world_1[source_id][target_id])
        if source_id in self.world_0 and target_id in self.world_0.get(source_id, {}):
            return float(self.world_0[source_id][target_id])
        return 0.0

    def write_speculative_mass(self, source_id: Hashable, target_id: Hashable, weight_delta: float) -> float:
        if source_id not in self.world_1:
            self.world_1[source_id] = {}
        if target_id not in self.world_1[source_id]:
            base_mass = self.world_0.get(source_id, {}).get(target_id, 0.0)
            self.world_1[source_id][target_id] = float(base_mass)
        self.world_1[source_id][target_id] += float(weight_delta)
        return float(self.world_1[source_id][target_id])

    def clear_speculative_universe(self) -> None:
        self.world_1.clear()

    def factual_fingerprint(self) -> str:
        return fingerprint_factual_manifold(self.world_0)


def verify_j6_isolation(
    factual_manifold: dict[Hashable, Any],
    speculative_updates: list[tuple[Hashable, Hashable, float]],
) -> tuple[str, str]:
    engine = VirtualCoWManifold(factual_manifold)
    baseline_fingerprint = engine.factual_fingerprint()
    for source_id, target_id, delta in speculative_updates:
        engine.write_speculative_mass(source_id, target_id, delta)
    engine.clear_speculative_universe()
    post_test_fingerprint = engine.factual_fingerprint()
    if baseline_fingerprint != post_test_fingerprint:
        return "FAIL", "TRUE"
    return "PASS", "FALSE"


__all__ = [
    "VirtualCoWManifold",
    "fingerprint_factual_manifold",
    "verify_j6_isolation",
]
