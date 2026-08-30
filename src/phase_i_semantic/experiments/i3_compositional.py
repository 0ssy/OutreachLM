from __future__ import annotations

from typing import Any

from src.phase_i_semantic.logic.compositional import RelationGraph


def run() -> dict[str, Any]:
    graph = RelationGraph()
    direct = [("A", "B"), ("B", "C"), ("C", "D"), ("X", "Y"), ("Y", "Z")]
    for left, right in direct:
        graph.add(left, right)

    memorized_checks = [graph.has_direct(left, right) for left, right in direct]
    compositional_checks = [
        graph.infer_transitive("A", "C"),
        graph.infer_transitive("A", "D"),
        graph.infer_transitive("X", "Z"),
    ]
    memorized_rate = sum(1 for item in memorized_checks if item) / len(memorized_checks)
    generalized_rate = sum(1 for item in compositional_checks if item) / len(compositional_checks)
    status = "PASS" if generalized_rate >= 0.8 else "FAIL"
    return {
        "memorized_sequence_reproduction_rate": memorized_rate,
        "unseen_compositional_generalization_rate": generalized_rate,
        "structural_composition_status": status,
    }

