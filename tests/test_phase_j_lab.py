from __future__ import annotations

import json
from pathlib import Path

import pytest

from outreachlm.phase_j_lab import PhaseJLab


def test_phase_j_submission_shape() -> None:
    submission = PhaseJLab.build_submission(
        token="REASONING_DEGRADATION_INITIAL_2026",
        j1={"baseline_deductive_accuracy": 0.99, "inference_versus_retrieval_ratio": 0.92},
        j2={
            "accuracy_1_hop": 0.99,
            "accuracy_2_hops": 0.98,
            "accuracy_4_hops": 0.96,
            "accuracy_8_hops": 0.92,
            "accuracy_16_hops": 0.9,
            "accuracy_32_hops": 0.87,
            "critical_decay_break_point_hops": 16,
        },
        j3={"paradox_detection_rate": 1.0, "blind_generation_error_count": 0},
        j4={"correlation_vs_causation_separation_rate": 0.97},
        j5={"chronological_ordering_accuracy": 0.97, "temporal_reversal_pass_rate": 0.95},
        j6={"hypothetical_world_isolation_rate": 1.0, "factual_state_corruption_detected": "FALSE"},
        j7={"valid_action_sequence_rate": 0.99, "dependency_violation_count": 0},
        j8={"sub_goal_extraction_accuracy": 0.98, "linear_dependency_pass_rate": 0.97},
        j9={"candidate_checking_intercept_rate": 0.98, "uncaught_false_state_count": 0},
        final_status="PASS",
    )

    assert submission["phase_j_diagnostic_token"] == "REASONING_DEGRADATION_INITIAL_2026"
    assert submission["experiment_j6_counterfactual"]["factual_state_corruption_detected"] == "FALSE"
    assert submission["final_diagnostic_gate_status"] == "PASS"


def test_phase_j_result_files_written(tmp_path: Path) -> None:
    submission = {"phase_j_diagnostic_token": "TOKEN"}
    detail = {"j1": {"ok": True}}
    summary_path, full_path = PhaseJLab.write_results(
        submission=submission,
        detail=detail,
        output_dir=tmp_path,
    )

    assert summary_path.exists()
    assert full_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == submission
    assert json.loads(full_path.read_text(encoding="utf-8")) == detail


def test_phase_j_config_hash_matches_frozen_profile() -> None:
    lab = PhaseJLab()
    actual = lab.config_hash_sha256(lab.config)
    expected = lab.frozen_profile["frozen_config_hash_sha256"]
    assert actual == expected


def test_phase_j_hard_gates_accept_valid_submission() -> None:
    lab = PhaseJLab()
    submission = PhaseJLab.build_submission(
        token="REASONING_DEGRADATION_INITIAL_2026",
        j1={"baseline_deductive_accuracy": 0.99, "inference_versus_retrieval_ratio": 0.92},
        j2={
            "accuracy_1_hop": 0.99,
            "accuracy_2_hops": 0.98,
            "accuracy_4_hops": 0.96,
            "accuracy_8_hops": 0.92,
            "accuracy_16_hops": 0.9,
            "accuracy_32_hops": 0.87,
            "critical_decay_break_point_hops": 16,
        },
        j3={"paradox_detection_rate": 1.0, "blind_generation_error_count": 0},
        j4={"correlation_vs_causation_separation_rate": 0.97},
        j5={"chronological_ordering_accuracy": 0.97, "temporal_reversal_pass_rate": 0.95},
        j6={"hypothetical_world_isolation_rate": 1.0, "factual_state_corruption_detected": "FALSE"},
        j7={"valid_action_sequence_rate": 0.99, "dependency_violation_count": 0},
        j8={"sub_goal_extraction_accuracy": 0.98, "linear_dependency_pass_rate": 0.97},
        j9={"candidate_checking_intercept_rate": 0.98, "uncaught_false_state_count": 0},
        final_status="PASS",
    )
    lab.enforce_hard_gates(submission)


def test_phase_j_hard_gates_reject_failed_submission() -> None:
    lab = PhaseJLab()
    submission = PhaseJLab.build_submission(
        token="REASONING_DEGRADATION_INITIAL_2026",
        j1={"baseline_deductive_accuracy": 0.90, "inference_versus_retrieval_ratio": 0.92},
        j2={
            "accuracy_1_hop": 0.99,
            "accuracy_2_hops": 0.98,
            "accuracy_4_hops": 0.96,
            "accuracy_8_hops": 0.88,
            "accuracy_16_hops": 0.9,
            "accuracy_32_hops": 0.87,
            "critical_decay_break_point_hops": 16,
        },
        j3={"paradox_detection_rate": 1.0, "blind_generation_error_count": 0},
        j4={"correlation_vs_causation_separation_rate": 0.97},
        j5={"chronological_ordering_accuracy": 0.97, "temporal_reversal_pass_rate": 0.95},
        j6={"hypothetical_world_isolation_rate": 1.0, "factual_state_corruption_detected": "FALSE"},
        j7={"valid_action_sequence_rate": 0.99, "dependency_violation_count": 0},
        j8={"sub_goal_extraction_accuracy": 0.98, "linear_dependency_pass_rate": 0.97},
        j9={"candidate_checking_intercept_rate": 0.98, "uncaught_false_state_count": 0},
        final_status="PASS",
    )
    with pytest.raises(RuntimeError, match="J1 baseline_deductive_accuracy below minimum"):
        lab.enforce_hard_gates(submission)
