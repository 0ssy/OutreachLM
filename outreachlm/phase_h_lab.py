from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.experiments.h1_memory import run as run_h1
from src.phase_h_cache.experiments.h2_vocab import run as run_h2
from src.phase_h_cache.experiments.h3_quantization import run as run_h3
from src.phase_h_cache.experiments.h4_eviction import run as run_h4
from src.phase_h_cache.experiments.h5_execution import run as run_h5


class PhaseHLab:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or PhaseHConfig.load_default().raw
        self._frozen_profile = self._load_frozen_profile()

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def frozen_profile(self) -> dict[str, Any]:
        return self._frozen_profile

    @staticmethod
    def _load_frozen_profile() -> dict[str, Any]:
        profile_path = Path(__file__).resolve().parent / "phase_h_frozen_profile.json"
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Frozen Phase H profile must be a JSON object.")
        return payload

    @staticmethod
    def config_hash_sha256(config: dict[str, Any]) -> str:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _enforce_frozen_config(self) -> None:
        expected = str(self.frozen_profile["frozen_config_hash_sha256"])
        actual = self.config_hash_sha256(self.config)
        if actual != expected:
            raise RuntimeError(
                "Phase H config hash mismatch; frozen profile violated. "
                f"expected={expected}, actual={actual}"
            )
        token_expected = str(self.frozen_profile["submission_token"])
        token_actual = str(self.config.get("phase_h_submission_token"))
        if token_actual != token_expected:
            raise RuntimeError(
                "Phase H submission token mismatch; frozen profile violated. "
                f"expected={token_expected}, actual={token_actual}"
            )

    @staticmethod
    def _assert_gate(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"Phase H frozen gate failed: {message}")

    def enforce_hard_gates(self, submission: dict[str, Any]) -> None:
        gates = self.frozen_profile["required_gates"]
        h1 = submission["experiment_h1"]
        h2 = submission["experiment_h2"]
        h3 = submission["experiment_h3"]
        h4 = submission["experiment_h4"]
        h5 = submission["experiment_h5"]

        self._assert_gate(
            int(h1["final_ingested_tokens"]) >= int(gates["h1_min_final_ingested_tokens"]),
            "H1 final_ingested_tokens below minimum",
        )
        self._assert_gate(
            str(h1["memory_scaling_profile"]) == str(gates["h1_memory_scaling_profile"]),
            "H1 memory_scaling_profile mismatch",
        )
        self._assert_gate(
            str(h2["selected_tokenizer_profile"]) == str(gates["h2_selected_tokenizer_profile"]),
            "H2 selected_tokenizer_profile mismatch",
        )
        self._assert_gate(
            float(h2["average_compression_ratio"]) >= float(gates["h2_min_average_compression_ratio"]),
            "H2 average_compression_ratio below minimum",
        )
        self._assert_gate(
            int(h2["final_vocabulary_size"]) <= int(gates["h2_max_final_vocabulary_size"]),
            "H2 final_vocabulary_size above maximum",
        )
        self._assert_gate(
            float(h3["measured_mass_error"]) <= float(gates["h3_max_mass_error"]),
            "H3 measured_mass_error above maximum",
        )
        self._assert_gate(
            float(h3["measured_kl_divergence_vs_g7"]) <= float(gates["h3_max_kl_divergence_vs_g7"]),
            "H3 measured_kl_divergence_vs_g7 above maximum",
        )
        self._assert_gate(
            str(h4["gate_status"]) == str(gates["h4_required_gate_status"]),
            "H4 gate_status mismatch",
        )
        self._assert_gate(
            float(h4["final_context_intervention_delta"]) >= float(gates["h4_min_context_intervention_delta"]),
            "H4 final_context_intervention_delta below minimum",
        )
        self._assert_gate(
            float(h4["peak_unk_absorbed_mass_percentage"])
            <= float(gates["h4_max_unk_absorbed_mass_percentage"]),
            "H4 peak_unk_absorbed_mass_percentage above maximum",
        )
        self._assert_gate(
            float(h5["measured_latency_variance_reduction_rate"])
            >= float(gates["h5_min_latency_variance_reduction_rate"]),
            "H5 measured_latency_variance_reduction_rate below minimum",
        )

    @staticmethod
    def build_submission(
        *,
        submission_token: str,
        h1: dict[str, Any],
        h2: dict[str, Any],
        h3: dict[str, Any],
        h4: dict[str, Any],
        h5: dict[str, Any],
        final_reproducibility_suite_score: str,
    ) -> dict[str, Any]:
        return {
            "phase_h_submission_token": submission_token,
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
                "optimal_physical_cores": int(h5["optimal_physical_cores"]),
                "unpinned_tokens_per_second": round(float(h5["unpinned_tokens_per_second"]), 2),
                "true_os_pinned_tokens_per_second": round(float(h5["true_os_pinned_tokens_per_second"]), 2),
                "measured_latency_variance_reduction_rate": round(
                    float(h5["measured_latency_variance_reduction_rate"]), 6
                ),
            },
            "final_reproducibility_suite_score": final_reproducibility_suite_score,
        }

    def run_suite(
        self,
        *,
        final_reproducibility_suite_score: str = "dirty",
        enforce_frozen_lock: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if enforce_frozen_lock:
            self._enforce_frozen_config()
        h1 = run_h1()
        h2 = run_h2()
        h3 = run_h3()
        h4 = run_h4()
        h5 = run_h5()
        submission = self.build_submission(
            submission_token=str(self.config["phase_h_submission_token"]),
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            h5=h5,
            final_reproducibility_suite_score=final_reproducibility_suite_score,
        )
        if enforce_frozen_lock:
            self.enforce_hard_gates(submission)
        detail = {"h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5}
        return submission, detail

    @staticmethod
    def write_results(
        *,
        submission: dict[str, Any],
        detail: dict[str, Any],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "phase_h_submission.json"
        full_path = output_dir / "phase_h_full_results.json"
        summary_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
        full_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
        return summary_path, full_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase H operational lab suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "phase_h" / "results",
        help="Directory for phase_h_submission.json and phase_h_full_results.json",
    )
    parser.add_argument(
        "--final-reproducibility-suite-score",
        type=str,
        default="dirty",
        help="Score string to include in final_reproducibility_suite_score.",
    )
    parser.add_argument(
        "--skip-frozen-lock",
        action="store_true",
        help="Run suite without enforcing frozen config hash and hard gates.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    lab = PhaseHLab()
    submission, detail = lab.run_suite(
        final_reproducibility_suite_score=args.final_reproducibility_suite_score,
        enforce_frozen_lock=not args.skip_frozen_lock,
    )
    PhaseHLab.write_results(
        submission=submission,
        detail=detail,
        output_dir=args.output_dir,
    )
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
