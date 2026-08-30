from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.experiments.h6_locality import run as run_h6
from src.phase_h_cache.experiments.h7_brutal_scaling import run as run_h7
from src.phase_h_cache.experiments.h8_long_context import run as run_h8
from src.phase_h_cache.experiments.h9_safety_audit import run as run_h9
from src.phase_h_cache.experiments.h10_acceptance import run as run_h10


class PhaseHDeepProfileLab:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or PhaseHConfig.load_default().raw

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @staticmethod
    def build_submission(
        *,
        token: str,
        h6: dict[str, Any],
        h7: dict[str, Any],
        h8: dict[str, Any],
        h9: dict[str, Any],
        h10: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "phase_h_deep_profile_token": token,
            "experiment_h6": {
                "measured_cache_spillover_threshold_mb": round(float(h6["measured_cache_spillover_threshold_mb"]), 6),
                "latency_increase_rate_post_spillover": round(float(h6["latency_increase_rate_post_spillover"]), 6),
                "cycles_per_token_at_optimal_size": round(float(h6["cycles_per_token_at_optimal_size"]), 6),
            },
            "experiment_h7": {
                "max_brutal_ingested_tokens": int(h7["max_brutal_ingested_tokens"]),
                "terminal_process_rss_bytes": int(h7["terminal_process_rss_bytes"]),
                "terminal_unk_mass_percentage": round(float(h7["terminal_unk_mass_percentage"]), 6),
                "unbounded_ingestion_stability_status": str(h7["unbounded_ingestion_stability_status"]),
            },
            "experiment_h8": {
                "maximum_graceful_context_sequence_limit": int(h8["maximum_graceful_context_sequence_limit"]),
                "context_smearing_detected_at_length": int(h8["context_smearing_detected_at_length"]),
                "nested_structure_tracking_rate": round(float(h8["nested_structure_tracking_rate"]), 6),
            },
            "experiment_h9": {
                "baseline_repetition_run_length": int(h9["baseline_repetition_run_length"]),
                "governed_repetition_run_length": int(h9["governed_repetition_run_length"]),
                "safety_layer_mass_loss_error": round(float(h9["safety_layer_mass_loss_error"]), 8),
                "sequence_entropy_shift_rate": round(float(h9["sequence_entropy_shift_rate"]), 6),
            },
            "experiment_h10": {
                "runtime_bridge_integration_score": str(h10["runtime_bridge_integration_score"]),
                "final_defensible_model_status": str(h10["final_defensible_model_status"]),
            },
        }

    def run_suite(self) -> tuple[dict[str, Any], dict[str, Any]]:
        h6 = run_h6()
        h7 = run_h7()
        h8 = run_h8()
        h9 = run_h9()
        h10 = run_h10(h6_result=h6, h7_result=h7, h8_result=h8, h9_result=h9)
        submission = self.build_submission(
            token=str(self.config["phase_h_deep_profile_token"]),
            h6=h6,
            h7=h7,
            h8=h8,
            h9=h9,
            h10=h10,
        )
        detail = {"h6": h6, "h7": h7, "h8": h8, "h9": h9, "h10": h10}
        return submission, detail

    @staticmethod
    def write_results(
        *,
        submission: dict[str, Any],
        detail: dict[str, Any],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "phase_h_deep_profile_submission.json"
        full_path = output_dir / "phase_h_deep_profile_full_results.json"
        summary_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
        full_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
        return summary_path, full_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase H deep-profiling suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "phase_h" / "deep_profile",
        help="Directory for deep profile artifacts.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    lab = PhaseHDeepProfileLab()
    submission, detail = lab.run_suite()
    PhaseHDeepProfileLab.write_results(
        submission=submission,
        detail=detail,
        output_dir=args.output_dir,
    )
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
