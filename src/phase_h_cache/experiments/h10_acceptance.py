from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.experiments.h6_locality import run as run_h6
from src.phase_h_cache.experiments.h7_brutal_scaling import run as run_h7
from src.phase_h_cache.experiments.h8_long_context import run as run_h8
from src.phase_h_cache.experiments.h9_safety_audit import run as run_h9


def run(
    *,
    h6_result: dict[str, Any] | None = None,
    h7_result: dict[str, Any] | None = None,
    h8_result: dict[str, Any] | None = None,
    h9_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw
    h6 = h6_result if h6_result is not None else run_h6()
    h7 = h7_result if h7_result is not None else run_h7()
    h8 = h8_result if h8_result is not None else run_h8()
    h9 = h9_result if h9_result is not None else run_h9()

    gates = {
        "h6_spillover_measured": float(h6["measured_cache_spillover_threshold_mb"]) > 0.0,
        "h7_stability_pass": str(h7["unbounded_ingestion_stability_status"]) == "PASS",
        "h8_context_retention": int(h8["maximum_graceful_context_sequence_limit"]) >= 512,
        "h9_safety_mass": float(h9["safety_layer_mass_loss_error"]) <= 1e-6,
        "h9_repetition_improved": int(h9["governed_repetition_run_length"]) <= int(h9["baseline_repetition_run_length"]),
    }
    pass_count = sum(1 for value in gates.values() if value)
    total = len(gates)
    score = int(round((pass_count / total) * 100))
    all_pass = pass_count == total

    return {
        "experiment_id": "h10_acceptance",
        "config_token": config["phase_h_deep_profile_token"],
        "gates": gates,
        "runtime_bridge_integration_score": "100_PERCENT_GREEN" if all_pass else f"{score}_PERCENT_GREEN",
        "final_defensible_model_status": "SCIENTIFICALLY_LOCKED" if all_pass else "NEEDS_REMEDIATION",
        "h6": h6,
        "h7": h7,
        "h8": h8,
        "h9": h9,
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "deep_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h10_acceptance.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
