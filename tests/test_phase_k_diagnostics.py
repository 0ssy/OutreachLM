from __future__ import annotations

from experiments.phase_k.run_phase_k import DEPTHS, build_k1_profile


def test_k1_profiles_all_requested_depths_without_repair_mechanisms() -> None:
    payload = build_k1_profile()
    profile = payload["experiment_k1_reasoning_depth"]

    assert profile["depths_tested"] == list(DEPTHS)
    assert payload["mechanisms_applied"] == []
    assert payload["final_diagnostic_gate_status"] == "AWAITING_DEGRADATION_PROFILING"
    assert len(profile["observations"]) == len(DEPTHS)


def test_k1_records_first_failure_state_for_degraded_depth() -> None:
    payload = build_k1_profile()
    observations = payload["experiment_k1_reasoning_depth"]["observations"]
    degraded = [
        row for row in observations if row["final_proof_confidence"] < 0.90
    ]

    assert degraded
    assert all(row["first_failure_state"] is not None for row in degraded)
