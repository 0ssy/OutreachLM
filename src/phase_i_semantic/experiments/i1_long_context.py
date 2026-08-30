from __future__ import annotations

from typing import Any

from src.phase_i_semantic.experiments.common import get_config
from src.phase_i_semantic.memory.context_hierarchy import MultiTierContextHierarchy


def run() -> dict[str, Any]:
    config = get_config()
    max_tokens = int(config["memory_hierarchy"]["max_context_horizon"])
    hierarchy = MultiTierContextHierarchy()

    sequence = []
    sequence.extend(["KEY_NEEDLE", "VALUE_TRUTH"])
    filler = ["noise"] * max(0, max_tokens - 2)
    sequence.extend(filler)
    hierarchy.ingest_tokens(sequence, max_local=512)

    recall = hierarchy.recall("KEY_NEEDLE")
    accuracy = 1.0 if recall == "VALUE_TRUTH" else 0.0
    smearing = accuracy < float(config["memory_hierarchy"]["retrieval_threshold"])
    return {
        "max_tested_context_tokens": max_tokens,
        "needle_recall_accuracy_at_16k": accuracy,
        "long_context_smearing_detected": bool(smearing),
    }

