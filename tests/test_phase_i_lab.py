from __future__ import annotations

import json
from pathlib import Path

import pytest

from outreachlm.phase_i_lab import PhaseILab


def test_phase_i_submission_shape() -> None:
    submission = PhaseILab.build_submission(
        token="TOKEN",
        i1={
            "max_tested_context_tokens": 16384,
            "needle_recall_accuracy_at_16k": 1.0,
            "long_context_smearing_detected": False,
        },
        i2={
            "active_tracked_relationships": 3,
            "coreference_resolution_accuracy_rate": 1.0,
            "transactional_state_errors_detected": 0,
        },
        i3={
            "memorized_sequence_reproduction_rate": 1.0,
            "unseen_compositional_generalization_rate": 1.0,
            "structural_composition_status": "PASS",
        },
        i4={
            "max_nested_bracket_depth_achieved": 4,
            "closure_validation_accuracy_rate": 1.0,
            "long_distance_dependency_failures": 0,
        },
        i5={
            "extracted_tuple_invariance_rate": 1.0,
            "paraphrase_matching_confidence_score": 1.0,
        },
        i6={
            "memory_retrieval_accuracy_rate": 1.0,
            "irrelevant_context_contamination_rate": 0.0,
            "stale_memory_collision_count": 0,
        },
        i7={
            "factual_consistency_maintenance_rate": 1.0,
            "degenerate_repetition_run_count": 0,
            "structural_output_validity_score": 1.0,
        },
        i8={
            "english_to_swahili_state_match_rate": 1.0,
            "japanese_to_mandarin_state_match_rate": 1.0,
            "cross_lingual_relation_invariance_status": "PASS",
        },
        i9={
            "misleading_distractor_bypass_rate": 1.0,
            "contradictory_fact_rejection_rate": 1.0,
            "primary_architectural_break_point_log": "none",
        },
        final_status="PASS",
    )

    assert submission["phase_i_capability_token"] == "TOKEN"
    assert submission["experiment_i1_memory"]["long_context_smearing_detected"] == "FALSE"
    assert submission["final_scientific_gate_status"] == "PASS"


def test_phase_i_result_files_written(tmp_path: Path) -> None:
    submission = {"phase_i_capability_token": "TOKEN"}
    detail = {"i1": {"ok": True}}
    summary_path, full_path = PhaseILab.write_results(
        submission=submission,
        detail=detail,
        output_dir=tmp_path,
    )

    assert summary_path.exists()
    assert full_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == submission
    assert json.loads(full_path.read_text(encoding="utf-8")) == detail


def test_phase_i_config_hash_matches_frozen_profile() -> None:
    lab = PhaseILab()
    actual = lab.config_hash_sha256(lab.config)
    expected = lab.frozen_profile["frozen_config_hash_sha256"]
    assert actual == expected


def test_phase_i_config_hash_lock_rejects_mutation() -> None:
    lab = PhaseILab(config={**PhaseILab().config, "phase_i_capability_token": "CHANGED"})
    with pytest.raises(RuntimeError, match="frozen profile violated"):
        lab._enforce_frozen_config()


def test_phase_i_hard_gates_reject_failed_submission() -> None:
    lab = PhaseILab()
    submission = PhaseILab.build_submission(
        token="SEMANTIC_INTELLIGENCE_VERIFIED_2026",
        i1={
            "max_tested_context_tokens": 16384,
            "needle_recall_accuracy_at_16k": 1.0,
            "long_context_smearing_detected": False,
        },
        i2={
            "active_tracked_relationships": 3,
            "coreference_resolution_accuracy_rate": 1.0,
            "transactional_state_errors_detected": 0,
        },
        i3={
            "memorized_sequence_reproduction_rate": 1.0,
            "unseen_compositional_generalization_rate": 1.0,
            "structural_composition_status": "PASS",
        },
        i4={
            "max_nested_bracket_depth_achieved": 4,
            "closure_validation_accuracy_rate": 0.90,
            "long_distance_dependency_failures": 1,
        },
        i5={
            "extracted_tuple_invariance_rate": 1.0,
            "paraphrase_matching_confidence_score": 1.0,
        },
        i6={
            "memory_retrieval_accuracy_rate": 1.0,
            "irrelevant_context_contamination_rate": 0.0,
            "stale_memory_collision_count": 0,
        },
        i7={
            "factual_consistency_maintenance_rate": 1.0,
            "degenerate_repetition_run_count": 0,
            "structural_output_validity_score": 1.0,
        },
        i8={
            "english_to_swahili_state_match_rate": 1.0,
            "japanese_to_mandarin_state_match_rate": 1.0,
            "cross_lingual_relation_invariance_status": "PASS",
        },
        i9={
            "misleading_distractor_bypass_rate": 1.0,
            "contradictory_fact_rejection_rate": 1.0,
            "primary_architectural_break_point_log": "none",
        },
        final_status="PASS",
    )
    with pytest.raises(RuntimeError, match="I4 closure_validation_accuracy_rate below minimum"):
        lab.enforce_hard_gates(submission)


def test_phase_i_hard_gates_accept_valid_submission() -> None:
    lab = PhaseILab()
    submission = PhaseILab.build_submission(
        token="SEMANTIC_INTELLIGENCE_VERIFIED_2026",
        i1={
            "max_tested_context_tokens": 16384,
            "needle_recall_accuracy_at_16k": 1.0,
            "long_context_smearing_detected": False,
        },
        i2={
            "active_tracked_relationships": 3,
            "coreference_resolution_accuracy_rate": 1.0,
            "transactional_state_errors_detected": 0,
        },
        i3={
            "memorized_sequence_reproduction_rate": 1.0,
            "unseen_compositional_generalization_rate": 1.0,
            "structural_composition_status": "PASS",
        },
        i4={
            "max_nested_bracket_depth_achieved": 4,
            "closure_validation_accuracy_rate": 1.0,
            "long_distance_dependency_failures": 0,
        },
        i5={
            "extracted_tuple_invariance_rate": 1.0,
            "paraphrase_matching_confidence_score": 1.0,
        },
        i6={
            "memory_retrieval_accuracy_rate": 1.0,
            "irrelevant_context_contamination_rate": 0.0,
            "stale_memory_collision_count": 0,
        },
        i7={
            "factual_consistency_maintenance_rate": 1.0,
            "degenerate_repetition_run_count": 0,
            "structural_output_validity_score": 1.0,
        },
        i8={
            "english_to_swahili_state_match_rate": 1.0,
            "japanese_to_mandarin_state_match_rate": 1.0,
            "cross_lingual_relation_invariance_status": "PASS",
        },
        i9={
            "misleading_distractor_bypass_rate": 1.0,
            "contradictory_fact_rejection_rate": 1.0,
            "primary_architectural_break_point_log": "none",
        },
        final_status="PASS",
    )
    lab.enforce_hard_gates(submission)
