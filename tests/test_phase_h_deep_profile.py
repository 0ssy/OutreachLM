from __future__ import annotations

import json
from pathlib import Path

from outreachlm.phase_h_deep_profile import PhaseHDeepProfileLab


def test_phase_h_deep_profile_submission_shape() -> None:
    payload = PhaseHDeepProfileLab.build_submission(
        token="TOKEN",
        h6={
            "measured_cache_spillover_threshold_mb": 4.0,
            "latency_increase_rate_post_spillover": 0.2,
            "cycles_per_token_at_optimal_size": 1000.0,
        },
        h7={
            "max_brutal_ingested_tokens": 10000,
            "terminal_process_rss_bytes": 1,
            "terminal_unk_mass_percentage": 1.0,
            "unbounded_ingestion_stability_status": "PASS",
        },
        h8={
            "maximum_graceful_context_sequence_limit": 1024,
            "context_smearing_detected_at_length": 2048,
            "nested_structure_tracking_rate": 0.75,
        },
        h9={
            "baseline_repetition_run_length": 10,
            "governed_repetition_run_length": 3,
            "safety_layer_mass_loss_error": 0.0,
            "sequence_entropy_shift_rate": -0.05,
        },
        h10={
            "runtime_bridge_integration_score": "100_PERCENT_GREEN",
            "final_defensible_model_status": "SCIENTIFICALLY_LOCKED",
        },
    )
    assert payload["phase_h_deep_profile_token"] == "TOKEN"
    assert payload["experiment_h10"]["runtime_bridge_integration_score"] == "100_PERCENT_GREEN"


def test_phase_h_deep_profile_write_results(tmp_path: Path) -> None:
    submission = {"phase_h_deep_profile_token": "X"}
    detail = {"h6": {"ok": True}}
    summary_path, full_path = PhaseHDeepProfileLab.write_results(
        submission=submission,
        detail=detail,
        output_dir=tmp_path,
    )
    assert summary_path.exists()
    assert full_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == submission
    assert json.loads(full_path.read_text(encoding="utf-8")) == detail
