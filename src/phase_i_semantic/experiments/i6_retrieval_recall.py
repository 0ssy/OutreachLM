from __future__ import annotations

from typing import Any

from src.phase_i_semantic.memory.context_hierarchy import MultiTierContextHierarchy


def run() -> dict[str, Any]:
    hierarchy = MultiTierContextHierarchy()
    keys = [f"KEY_{idx}" for idx in range(30)]
    values = [f"VAL_{idx}" for idx in range(30)]

    contamination = 0
    for key, value in zip(keys, values):
        hierarchy.ingest_tokens([key, value] + ["noise"] * 40, max_local=256)

    correct = 0
    collisions = 0
    for key, value in zip(keys, values):
        recall = hierarchy.recall(key)
        if recall == value:
            correct += 1
        elif recall is None:
            contamination += 1
        else:
            collisions += 1
    total = len(keys)
    return {
        "memory_retrieval_accuracy_rate": correct / total,
        "irrelevant_context_contamination_rate": contamination / total,
        "stale_memory_collision_count": collisions,
    }

