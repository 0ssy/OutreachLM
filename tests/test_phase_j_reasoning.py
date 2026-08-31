from __future__ import annotations

from experiments.phase_j.run_phase_j import _build_payload


def test_phase_j_gate_matrix_passes() -> None:
    payload = _build_payload()
    assert payload["final_diagnostic_gate_status"] == "PASS"
    assert payload["experiment_j1_deduction"]["baseline_deductive_accuracy"] >= 0.98
    assert payload["experiment_j2_multi_hop_curve"]["accuracy_8_hops"] >= 0.90
    assert payload["experiment_j7_planning"]["dependency_violation_count"] == 0
    assert payload["experiment_j9_self_verification"]["candidate_checking_intercept_rate"] >= 0.95


def test_phase_j_submission_shape() -> None:
    payload = _build_payload()
    assert set(payload) == {
        "phase_j_diagnostic_token",
        "experiment_j1_deduction",
        "experiment_j2_multi_hop_curve",
        "experiment_j3_constraints",
        "experiment_j4_causal",
        "experiment_j5_temporal",
        "experiment_j6_counterfactual",
        "experiment_j7_planning",
        "experiment_j8_decomposition",
        "experiment_j9_self_verification",
        "final_diagnostic_gate_status",
    }
