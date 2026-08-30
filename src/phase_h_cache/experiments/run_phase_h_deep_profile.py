from __future__ import annotations

import json
from pathlib import Path

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.experiments.h6_locality import run as run_h6
from src.phase_h_cache.experiments.h7_brutal_scaling import run as run_h7
from src.phase_h_cache.experiments.h8_long_context import run as run_h8
from src.phase_h_cache.experiments.h9_safety_audit import run as run_h9
from src.phase_h_cache.experiments.h10_acceptance import run as run_h10


def main() -> None:
    cfg = PhaseHConfig.load_default().raw
    h6 = run_h6()
    h7 = run_h7()
    h8 = run_h8()
    h9 = run_h9()
    h10 = run_h10(h6_result=h6, h7_result=h7, h8_result=h8, h9_result=h9)

    summary = {
        "phase_h_deep_profile_token": cfg["phase_h_deep_profile_token"],
        "experiment_h6": {
            "measured_cache_spillover_threshold_mb": float(h6["measured_cache_spillover_threshold_mb"]),
            "latency_increase_rate_post_spillover": float(h6["latency_increase_rate_post_spillover"]),
            "cycles_per_token_at_optimal_size": float(h6["cycles_per_token_at_optimal_size"]),
        },
        "experiment_h7": {
            "max_brutal_ingested_tokens": int(h7["max_brutal_ingested_tokens"]),
            "terminal_process_rss_bytes": int(h7["terminal_process_rss_bytes"]),
            "terminal_unk_mass_percentage": float(h7["terminal_unk_mass_percentage"]),
            "unbounded_ingestion_stability_status": str(h7["unbounded_ingestion_stability_status"]),
        },
        "experiment_h8": {
            "maximum_graceful_context_sequence_limit": int(h8["maximum_graceful_context_sequence_limit"]),
            "context_smearing_detected_at_length": int(h8["context_smearing_detected_at_length"]),
            "nested_structure_tracking_rate": float(h8["nested_structure_tracking_rate"]),
        },
        "experiment_h9": {
            "baseline_repetition_run_length": int(h9["baseline_repetition_run_length"]),
            "governed_repetition_run_length": int(h9["governed_repetition_run_length"]),
            "safety_layer_mass_loss_error": float(h9["safety_layer_mass_loss_error"]),
            "sequence_entropy_shift_rate": float(h9["sequence_entropy_shift_rate"]),
        },
        "experiment_h10": {
            "runtime_bridge_integration_score": str(h10["runtime_bridge_integration_score"]),
            "final_defensible_model_status": str(h10["final_defensible_model_status"]),
        },
    }
    detail = {"h6": h6, "h7": h7, "h8": h8, "h9": h9, "h10": h10}

    output_dir = Path("experiments") / "phase_h" / "deep_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_h_deep_profile_submission.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "phase_h_deep_profile_full_results.json").write_text(
        json.dumps(detail, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
