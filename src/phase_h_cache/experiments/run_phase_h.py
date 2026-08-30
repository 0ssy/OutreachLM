from __future__ import annotations

import json
from pathlib import Path

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.experiments.h1_memory import run as run_h1
from src.phase_h_cache.experiments.h2_vocab import run as run_h2
from src.phase_h_cache.experiments.h3_quantization import run as run_h3
from src.phase_h_cache.experiments.h4_eviction import run as run_h4
from src.phase_h_cache.experiments.h5_execution import run as run_h5


def main() -> None:
    config = PhaseHConfig.load_default().raw
    h1 = run_h1()
    h2 = run_h2()
    h3 = run_h3()
    h4 = run_h4()
    h5 = run_h5()

    payload = {
        "phase_h_submission_token": config["phase_h_submission_token"],
        "experiment_h1": {
            "final_ingested_tokens": h1["final_ingested_tokens"],
            "peak_logical_model_size_bytes": h1["peak_logical_model_size_bytes"],
            "peak_process_rss_bytes": h1["peak_process_rss_bytes"],
            "memory_scaling_profile": h1["memory_scaling_profile"],
        },
        "experiment_h2": {
            "selected_tokenizer_profile": h2["selected_tokenizer_profile"],
            "final_vocabulary_size": h2["final_vocabulary_size"],
            "average_compression_ratio": round(float(h2["average_compression_ratio"]), 6),
        },
        "experiment_h3": {
            "winning_precision_format": h3["winning_precision_format"],
            "measured_mass_error": round(float(h3["measured_mass_error"]), 8),
            "measured_kl_divergence_vs_g7": round(float(h3["measured_kl_divergence_vs_g7"]), 6),
        },
        "experiment_h4": {
            "winning_eviction_strategy": h4["winning_eviction_strategy"],
            "final_context_intervention_delta": round(float(h4["final_context_intervention_delta"]), 6),
            "peak_unk_absorbed_mass_percentage": round(float(h4["peak_unk_absorbed_mass_percentage"]), 6),
            "gate_status": h4["gate_status"],
        },
        "experiment_h5": {
            "optimal_physical_cores": h5["optimal_physical_cores"],
            "unpinned_tokens_per_second": round(float(h5["unpinned_tokens_per_second"]), 2),
            "true_os_pinned_tokens_per_second": round(float(h5["true_os_pinned_tokens_per_second"]), 2),
            "measured_latency_variance_reduction_rate": round(
                float(h5["measured_latency_variance_reduction_rate"]), 6
            ),
        },
        "final_reproducibility_suite_score": "dirty",
    }

    output_dir = Path("experiments") / "phase_h" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "phase_h_full_results.json"
    detail_path.write_text(
        json.dumps(
            {
                "h1": h1,
                "h2": h2,
                "h3": h3,
                "h4": h4,
                "h5": h5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = output_dir / "phase_h_submission.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
