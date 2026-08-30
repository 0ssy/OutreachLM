from __future__ import annotations

import json
from pathlib import Path

import pytest

from outreachlm.phase_h_lab import PhaseHLab


def test_phase_h_submission_shape() -> None:
    submission = PhaseHLab.build_submission(
        submission_token="TOKEN",
        h1={
            "final_ingested_tokens": 100000,
            "peak_logical_model_size_bytes": 2000,
            "peak_process_rss_bytes": 3000,
            "memory_scaling_profile": "sublinear",
        },
        h2={
            "selected_tokenizer_profile": "online_bpe",
            "final_vocabulary_size": 1024,
            "average_compression_ratio": 2.5,
        },
        h3={
            "winning_precision_format": "fp16",
            "measured_mass_error": 0.0,
            "measured_kl_divergence_vs_g7": 0.0,
        },
        h4={
            "winning_eviction_strategy": "frequency",
            "final_context_intervention_delta": 0.9,
            "peak_unk_absorbed_mass_percentage": 0.03,
            "gate_status": "PASS",
        },
        h5={
            "optimal_physical_cores": 2,
            "unpinned_tokens_per_second": 1000.0,
            "true_os_pinned_tokens_per_second": 1100.0,
            "measured_latency_variance_reduction_rate": 0.2,
        },
        final_reproducibility_suite_score="198/198",
    )

    assert submission["phase_h_submission_token"] == "TOKEN"
    assert submission["experiment_h4"]["gate_status"] == "PASS"
    assert submission["experiment_h5"]["optimal_physical_cores"] == 2
    assert submission["final_reproducibility_suite_score"] == "198/198"


def test_phase_h_result_files_written(tmp_path: Path) -> None:
    submission = {"phase_h_submission_token": "TOKEN"}
    detail = {"h1": {"ok": True}}
    summary_path, full_path = PhaseHLab.write_results(
        submission=submission,
        detail=detail,
        output_dir=tmp_path,
    )

    assert summary_path.exists()
    assert full_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == submission
    assert json.loads(full_path.read_text(encoding="utf-8")) == detail


def test_phase_h_config_hash_matches_frozen_profile() -> None:
    lab = PhaseHLab()
    actual = lab.config_hash_sha256(lab.config)
    expected = lab.frozen_profile["frozen_config_hash_sha256"]
    assert actual == expected


def test_phase_h_config_hash_lock_rejects_mutation() -> None:
    lab = PhaseHLab(config={**PhaseHLab().config, "phase_h_submission_token": "CHANGED"})
    with pytest.raises(RuntimeError, match="frozen profile violated"):
        lab._enforce_frozen_config()


def test_phase_h_hard_gates_reject_failed_submission() -> None:
    lab = PhaseHLab()
    submission = PhaseHLab.build_submission(
        submission_token="SOVEREIGN_CACHE_VERIFIED_2026",
        h1={
            "final_ingested_tokens": 100000,
            "peak_logical_model_size_bytes": 2000,
            "peak_process_rss_bytes": 3000,
            "memory_scaling_profile": "sublinear",
        },
        h2={
            "selected_tokenizer_profile": "online_bpe",
            "final_vocabulary_size": 1024,
            "average_compression_ratio": 2.51,
        },
        h3={
            "winning_precision_format": "fp16",
            "measured_mass_error": 0.0,
            "measured_kl_divergence_vs_g7": 0.0,
        },
        h4={
            "winning_eviction_strategy": "frequency",
            "final_context_intervention_delta": 0.9,
            "peak_unk_absorbed_mass_percentage": 0.03,
            "gate_status": "PASS",
        },
        h5={
            "optimal_physical_cores": 2,
            "unpinned_tokens_per_second": 1000.0,
            "true_os_pinned_tokens_per_second": 1100.0,
            "measured_latency_variance_reduction_rate": 0.1,
        },
        final_reproducibility_suite_score="200/200",
    )
    with pytest.raises(RuntimeError, match="H5 measured_latency_variance_reduction_rate below minimum"):
        lab.enforce_hard_gates(submission)


def test_phase_h_hard_gates_accept_valid_submission() -> None:
    lab = PhaseHLab()
    submission = PhaseHLab.build_submission(
        submission_token="SOVEREIGN_CACHE_VERIFIED_2026",
        h1={
            "final_ingested_tokens": 100000,
            "peak_logical_model_size_bytes": 2000,
            "peak_process_rss_bytes": 3000,
            "memory_scaling_profile": "sublinear",
        },
        h2={
            "selected_tokenizer_profile": "online_bpe",
            "final_vocabulary_size": 1024,
            "average_compression_ratio": 2.51,
        },
        h3={
            "winning_precision_format": "fp16",
            "measured_mass_error": 0.0,
            "measured_kl_divergence_vs_g7": 0.0,
        },
        h4={
            "winning_eviction_strategy": "frequency",
            "final_context_intervention_delta": 0.9,
            "peak_unk_absorbed_mass_percentage": 0.03,
            "gate_status": "PASS",
        },
        h5={
            "optimal_physical_cores": 2,
            "unpinned_tokens_per_second": 1000.0,
            "true_os_pinned_tokens_per_second": 1100.0,
            "measured_latency_variance_reduction_rate": 0.2,
        },
        final_reproducibility_suite_score="200/200",
    )
    lab.enforce_hard_gates(submission)
