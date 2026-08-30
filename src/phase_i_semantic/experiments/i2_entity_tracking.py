from __future__ import annotations

from typing import Any

from src.phase_i_semantic.memory.tracker import RelationshipTracker


def run() -> dict[str, Any]:
    tracker = RelationshipTracker()
    scenarios = [
        ("John gave Mary the book.", "book", "Mary"),
        ("Mary gave it to Peter.", "book", "Peter"),
        ("Alice gave Bob the key.", "key", "Bob"),
        ("Bob gave it to Carol.", "key", "Carol"),
        ("Nina gave Omar the ring.", "ring", "Omar"),
        ("Omar gave it to Pia.", "ring", "Pia"),
    ]
    correct = 0
    for sentence, item, expected_owner in scenarios:
        tracker.apply_sentence(sentence)
        owner = tracker.current_owner(item)
        if owner == expected_owner:
            correct += 1
    accuracy = correct / len(scenarios)
    return {
        "active_tracked_relationships": len(tracker.ownership),
        "coreference_resolution_accuracy_rate": accuracy,
        "transactional_state_errors_detected": tracker.errors,
    }

