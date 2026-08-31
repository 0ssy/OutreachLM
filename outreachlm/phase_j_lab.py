from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outreachlm.phase_j_runtime import ReasoningRuntime


class PhaseJLab:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {
            "phase_j_capability_token": "REASONING_DEGRADATION_INITIAL_2026",
            "reasoning_runtime": {
                "max_hop_depth": 32,
                "min_deductive_accuracy": 0.98,
                "min_8_hop_accuracy": 0.90,
                "max_dependency_violations": 0,
                "min_self_verification_intercept_rate": 0.95,
                "counterfactual_guard_required": True,
            },
        }
        self._frozen_profile = self._load_frozen_profile()

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def frozen_profile(self) -> dict[str, Any]:
        return self._frozen_profile

    @staticmethod
    def _load_frozen_profile() -> dict[str, Any]:
        profile_path = Path(__file__).resolve().parent / "phase_j_frozen_profile.json"
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Frozen Phase J profile must be a JSON object.")
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
                "Phase J config hash mismatch; frozen profile violated. "
                f"expected={expected}, actual={actual}"
            )
        token_expected = str(self.frozen_profile["capability_token"])
        token_actual = str(self.config.get("phase_j_capability_token"))
        if token_actual != token_expected:
            raise RuntimeError(
                "Phase J capability token mismatch; frozen profile violated. "
                f"expected={token_expected}, actual={token_actual}"
            )

    @staticmethod
    def _assert_gate(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"Phase J frozen gate failed: {message}")

    def enforce_hard_gates(self, submission: dict[str, Any]) -> None:
        gates = self.frozen_profile["required_gates"]
        j1 = submission["experiment_j1_deduction"]
        j2 = submission["experiment_j2_multi_hop_curve"]
        j3 = submission["experiment_j3_constraints"]
        j4 = submission["experiment_j4_causal"]
        j5 = submission["experiment_j5_temporal"]
        j6 = submission["experiment_j6_counterfactual"]
        j7 = submission["experiment_j7_planning"]
        j8 = submission["experiment_j8_decomposition"]
        j9 = submission["experiment_j9_self_verification"]

        self._assert_gate(
            float(j1["baseline_deductive_accuracy"]) >= float(gates["j1_min_deductive_accuracy"]),
            "J1 baseline_deductive_accuracy below minimum",
        )
        self._assert_gate(
            float(j2["accuracy_8_hops"]) >= float(gates["j2_min_8_hop_accuracy"]),
            "J2 accuracy_8_hops below minimum",
        )
        self._assert_gate(
            float(j3["paradox_detection_rate"]) >= float(gates["j3_min_paradox_detection_rate"]),
            "J3 paradox_detection_rate below minimum",
        )
        self._assert_gate(
            int(j3["blind_generation_error_count"]) <= int(gates["j3_max_blind_generation_error_count"]),
            "J3 blind_generation_error_count above maximum",
        )
        self._assert_gate(
            float(j4["correlation_vs_causation_separation_rate"]) >= float(gates["j4_min_causal_separation_rate"]),
            "J4 correlation_vs_causation_separation_rate below minimum",
        )
        self._assert_gate(
            float(j5["chronological_ordering_accuracy"]) >= float(gates["j5_min_chronological_ordering_accuracy"]),
            "J5 chronological_ordering_accuracy below minimum",
        )
        self._assert_gate(
            str(j6["factual_state_corruption_detected"]).upper() == str(gates["j6_required_factual_corruption"]).upper(),
            "J6 factual_state_corruption_detected mismatch",
        )
        self._assert_gate(
            int(j7["dependency_violation_count"]) <= int(gates["j7_max_dependency_violations"]),
            "J7 dependency_violation_count above maximum",
        )
        self._assert_gate(
            float(j8["sub_goal_extraction_accuracy"]) >= float(gates["j8_min_sub_goal_accuracy"]),
            "J8 sub_goal_extraction_accuracy below minimum",
        )
        self._assert_gate(
            float(j9["candidate_checking_intercept_rate"]) >= float(gates["j9_min_candidate_checking_intercept_rate"]),
            "J9 candidate_checking_intercept_rate below minimum",
        )
        self._assert_gate(
            str(submission["final_diagnostic_gate_status"]).upper() == str(gates["final_required_gate_status"]).upper(),
            "final_diagnostic_gate_status mismatch",
        )

    @staticmethod
    def build_submission(
        *,
        token: str,
        j1: dict[str, Any],
        j2: dict[str, Any],
        j3: dict[str, Any],
        j4: dict[str, Any],
        j5: dict[str, Any],
        j6: dict[str, Any],
        j7: dict[str, Any],
        j8: dict[str, Any],
        j9: dict[str, Any],
        final_status: str,
    ) -> dict[str, Any]:
        return {
            "phase_j_diagnostic_token": token,
            "experiment_j1_deduction": {
                "baseline_deductive_accuracy": round(float(j1["baseline_deductive_accuracy"]), 6),
                "inference_versus_retrieval_ratio": round(float(j1["inference_versus_retrieval_ratio"]), 6),
            },
            "experiment_j2_multi_hop_curve": {
                "accuracy_1_hop": round(float(j2["accuracy_1_hop"]), 6),
                "accuracy_2_hops": round(float(j2["accuracy_2_hops"]), 6),
                "accuracy_4_hops": round(float(j2["accuracy_4_hops"]), 6),
                "accuracy_8_hops": round(float(j2["accuracy_8_hops"]), 6),
                "accuracy_16_hops": round(float(j2["accuracy_16_hops"]), 6),
                "accuracy_32_hops": round(float(j2["accuracy_32_hops"]), 6),
                "critical_decay_break_point_hops": int(j2["critical_decay_break_point_hops"]),
            },
            "experiment_j3_constraints": {
                "paradox_detection_rate": round(float(j3["paradox_detection_rate"]), 6),
                "blind_generation_error_count": int(j3["blind_generation_error_count"]),
            },
            "experiment_j4_causal": {
                "correlation_vs_causation_separation_rate": round(float(j4["correlation_vs_causation_separation_rate"]), 6),
            },
            "experiment_j5_temporal": {
                "chronological_ordering_accuracy": round(float(j5["chronological_ordering_accuracy"]), 6),
                "temporal_reversal_pass_rate": round(float(j5["temporal_reversal_pass_rate"]), 6),
            },
            "experiment_j6_counterfactual": {
                "hypothetical_world_isolation_rate": round(float(j6["hypothetical_world_isolation_rate"]), 6),
                "factual_state_corruption_detected": str(j6["factual_state_corruption_detected"]),
            },
            "experiment_j7_planning": {
                "valid_action_sequence_rate": round(float(j7["valid_action_sequence_rate"]), 6),
                "dependency_violation_count": int(j7["dependency_violation_count"]),
            },
            "experiment_j8_decomposition": {
                "sub_goal_extraction_accuracy": round(float(j8["sub_goal_extraction_accuracy"]), 6),
                "linear_dependency_pass_rate": round(float(j8["linear_dependency_pass_rate"]), 6),
            },
            "experiment_j9_self_verification": {
                "candidate_checking_intercept_rate": round(float(j9["candidate_checking_intercept_rate"]), 6),
                "uncaught_false_state_count": int(j9["uncaught_false_state_count"]),
            },
            "final_diagnostic_gate_status": final_status,
        }

    def run_suite(self, *, enforce_frozen_lock: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        if enforce_frozen_lock:
            self._enforce_frozen_config()
        runtime = ReasoningRuntime()
        submission, detail = runtime.run_suite()
        if enforce_frozen_lock:
            self.enforce_hard_gates(submission)
        return submission, detail

    @staticmethod
    def write_results(*, submission: dict[str, Any], detail: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "phase_j_submission.json"
        full_path = output_dir / "phase_j_full_results.json"
        summary_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
        full_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
        return summary_path, full_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen Phase J reasoning and planning diagnostic suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "phase_j" / "results",
        help="Directory for phase_j_submission.json and phase_j_full_results.json",
    )
    parser.add_argument(
        "--skip-frozen-lock",
        action="store_true",
        help="Run the suite without enforcing the frozen config hash and hard gates.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    lab = PhaseJLab()
    submission, detail = lab.run_suite(enforce_frozen_lock=not args.skip_frozen_lock)
    PhaseJLab.write_results(submission=submission, detail=detail, output_dir=args.output_dir)
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
