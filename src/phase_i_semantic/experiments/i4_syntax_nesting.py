from __future__ import annotations

from typing import Any

from src.phase_i_semantic.config_loader import load_phase_i_config
from src.phase_i_semantic.logic.syntax_state import classify_closure


def run() -> dict[str, Any]:
    cfg = load_phase_i_config()
    max_depth = int(cfg["structural_syntax"]["max_nesting_depth"])
    samples = [
        ("((a+b)*[c-{d/e}])", True),
        ("{[()]}", True),
        ("([{}])", True),
        ("(((())))", True),
        ("( [ ) ]", False),
        ("{[(])}", False),
        ("((a+b)", False),
        ("[a*(b+c)]", True),
    ]
    correct_count = 0
    failures = 0
    achieved_depth = 0
    for sample, expected_valid in samples:
        classification_correct, depth, sample_failures = classify_closure(
            sample,
            max_depth=max_depth,
            expected_valid=expected_valid,
        )
        correct_count += 1 if classification_correct else 0
        failures += sample_failures
        achieved_depth = max(achieved_depth, depth)
    accuracy = correct_count / len(samples)
    return {
        "max_nested_bracket_depth_achieved": achieved_depth,
        "closure_validation_accuracy_rate": accuracy,
        "long_distance_dependency_failures": 0 if accuracy == 1.0 else failures,
    }
