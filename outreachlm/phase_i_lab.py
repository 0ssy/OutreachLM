from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.phase_i_semantic.config_loader import load_phase_i_config
from src.phase_i_semantic.experiments.i1_long_context import run as run_i1
from src.phase_i_semantic.experiments.i2_entity_tracking import run as run_i2
from src.phase_i_semantic.experiments.i3_compositional import run as run_i3
from src.phase_i_semantic.experiments.i4_syntax_nesting import run as run_i4
from src.phase_i_semantic.experiments.i5_semantic_state import run as run_i5
from src.phase_i_semantic.experiments.i6_retrieval_recall import run as run_i6
from src.phase_i_semantic.experiments.i7_control_generation import run as run_i7
from src.phase_i_semantic.experiments.i8_multilingual import run as run_i8
from src.phase_i_semantic.experiments.i9_adversarial import run as run_i9


class PhaseILab:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or load_phase_i_config()
        self._frozen_profile = self._load_frozen_profile()

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def frozen_profile(self) -> dict[str, Any]:
        return self._frozen_profile

    @staticmethod
    def _load_frozen_profile() -> dict[str, Any]:
        profile_path = Path(__file__).resolve().parent / "phase_i_frozen_profile.json"
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Frozen Phase I profile must be a JSON object.")
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
                "Phase I config hash mismatch; frozen profile violated. "
                f"expected={expected}, actual={actual}"
            )
        token_expected = str(self.frozen_profile["capability_token"])
        token_actual = str(self.config.get("phase_i_capability_token"))
        if token_actual != token_expected:
            raise RuntimeError(
                "Phase I capability token mismatch; frozen profile violated. "
                f"expected={token_expected}, actual={token_actual}"
            )

    @staticmethod
    def _assert_gate(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"Phase I frozen gate failed: {message}")

    def enforce_hard_gates(self, submission: dict[str, Any]) -> None:
        gates = self.frozen_profile["required_gates"]
        i1 = submission["experiment_i1_memory"]
        i2 = submission["experiment_i2_entities"]
        i3 = submission["experiment_i3_composition"]
        i4 = submission["experiment_i4_syntax"]
        i5 = submission["experiment_i5_semantics"]
        i6 = submission["experiment_i6_retrieval"]
        i7 = submission["experiment_i7_generation"]
        i8 = submission["experiment_i8_multilingual"]
        i9 = submission["experiment_i9_adversarial"]

        self._assert_gate(
            int(i1["max_tested_context_tokens"]) >= int(gates["i1_min_max_tested_context_tokens"]),
            "I1 max_tested_context_tokens below minimum",
        )
        self._assert_gate(
            float(i1["needle_recall_accuracy_at_16k"]) >= float(gates["i1_min_needle_recall_accuracy_at_16k"]),
            "I1 needle_recall_accuracy_at_16k below minimum",
        )
        self._assert_gate(
            str(i1["long_context_smearing_detected"]).upper() == str(gates["i1_required_long_context_smearing"]).upper(),
            "I1 long_context_smearing_detected mismatch",
        )
        self._assert_gate(
            float(i2["coreference_resolution_accuracy_rate"]) >= float(gates["i2_min_coreference_resolution_accuracy_rate"]),
            "I2 coreference_resolution_accuracy_rate below minimum",
        )
        self._assert_gate(
            int(i2["transactional_state_errors_detected"]) <= int(gates["i2_max_transactional_state_errors_detected"]),
            "I2 transactional_state_errors_detected above maximum",
        )
        self._assert_gate(
            str(i3["structural_composition_status"]) == str(gates["i3_required_structural_composition_status"]),
            "I3 structural_composition_status mismatch",
        )
        self._assert_gate(
            float(i3["unseen_compositional_generalization_rate"])
            >= float(gates["i3_min_unseen_compositional_generalization_rate"]),
            "I3 unseen_compositional_generalization_rate below minimum",
        )
        self._assert_gate(
            float(i4["closure_validation_accuracy_rate"]) >= float(gates["i4_min_closure_validation_accuracy_rate"]),
            "I4 closure_validation_accuracy_rate below minimum",
        )
        self._assert_gate(
            float(i5["paraphrase_matching_confidence_score"]) >= float(gates["i5_min_paraphrase_matching_confidence_score"]),
            "I5 paraphrase_matching_confidence_score below minimum",
        )
        self._assert_gate(
            float(i6["memory_retrieval_accuracy_rate"]) >= float(gates["i6_min_memory_retrieval_accuracy_rate"]),
            "I6 memory_retrieval_accuracy_rate below minimum",
        )
        self._assert_gate(
            float(i6["irrelevant_context_contamination_rate"]) <= float(gates["i6_max_irrelevant_context_contamination_rate"]),
            "I6 irrelevant_context_contamination_rate above maximum",
        )
        self._assert_gate(
            float(i7["factual_consistency_maintenance_rate"])
            >= float(gates["i7_min_factual_consistency_maintenance_rate"]),
            "I7 factual_consistency_maintenance_rate below minimum",
        )
        self._assert_gate(
            float(i7["structural_output_validity_score"]) >= float(gates["i7_min_structural_output_validity_score"]),
            "I7 structural_output_validity_score below minimum",
        )
        self._assert_gate(
            int(i7["degenerate_repetition_run_count"]) <= int(gates["i7_max_degenerate_repetition_run_count"]),
            "I7 degenerate_repetition_run_count above maximum",
        )
        self._assert_gate(
            str(i8["cross_lingual_relation_invariance_status"])
            == str(gates["i8_required_cross_lingual_relation_invariance_status"]),
            "I8 cross_lingual_relation_invariance_status mismatch",
        )
        self._assert_gate(
            float(i9["misleading_distractor_bypass_rate"]) >= float(gates["i9_min_misleading_distractor_bypass_rate"]),
            "I9 misleading_distractor_bypass_rate below minimum",
        )
        self._assert_gate(
            float(i9["contradictory_fact_rejection_rate"]) >= float(gates["i9_min_contradictory_fact_rejection_rate"]),
            "I9 contradictory_fact_rejection_rate below minimum",
        )
        self._assert_gate(
            str(i9["primary_architectural_break_point_log"]) == str(gates["i9_required_primary_architectural_break_point_log"]),
            "I9 primary_architectural_break_point_log mismatch",
        )
        self._assert_gate(
            str(submission["final_scientific_gate_status"]) == str(gates["final_required_scientific_gate_status"]),
            "final_scientific_gate_status mismatch",
        )

    @staticmethod
    def build_submission(
        *,
        token: str,
        i1: dict[str, Any],
        i2: dict[str, Any],
        i3: dict[str, Any],
        i4: dict[str, Any],
        i5: dict[str, Any],
        i6: dict[str, Any],
        i7: dict[str, Any],
        i8: dict[str, Any],
        i9: dict[str, Any],
        final_status: str,
    ) -> dict[str, Any]:
        return {
            "phase_i_capability_token": token,
            "experiment_i1_memory": {
                "max_tested_context_tokens": int(i1["max_tested_context_tokens"]),
                "needle_recall_accuracy_at_16k": round(float(i1["needle_recall_accuracy_at_16k"]), 6),
                "long_context_smearing_detected": "TRUE" if bool(i1["long_context_smearing_detected"]) else "FALSE",
            },
            "experiment_i2_entities": {
                "active_tracked_relationships": int(i2["active_tracked_relationships"]),
                "coreference_resolution_accuracy_rate": round(float(i2["coreference_resolution_accuracy_rate"]), 6),
                "transactional_state_errors_detected": int(i2["transactional_state_errors_detected"]),
            },
            "experiment_i3_composition": {
                "memorized_sequence_reproduction_rate": round(float(i3["memorized_sequence_reproduction_rate"]), 6),
                "unseen_compositional_generalization_rate": round(float(i3["unseen_compositional_generalization_rate"]), 6),
                "structural_composition_status": str(i3["structural_composition_status"]),
            },
            "experiment_i4_syntax": {
                "max_nested_bracket_depth_achieved": int(i4["max_nested_bracket_depth_achieved"]),
                "closure_validation_accuracy_rate": round(float(i4["closure_validation_accuracy_rate"]), 6),
                "long_distance_dependency_failures": int(i4["long_distance_dependency_failures"]),
            },
            "experiment_i5_semantics": {
                "extracted_tuple_invariance_rate": round(float(i5["extracted_tuple_invariance_rate"]), 6),
                "paraphrase_matching_confidence_score": round(float(i5["paraphrase_matching_confidence_score"]), 6),
            },
            "experiment_i6_retrieval": {
                "memory_retrieval_accuracy_rate": round(float(i6["memory_retrieval_accuracy_rate"]), 6),
                "irrelevant_context_contamination_rate": round(float(i6["irrelevant_context_contamination_rate"]), 6),
                "stale_memory_collision_count": int(i6["stale_memory_collision_count"]),
            },
            "experiment_i7_generation": {
                "factual_consistency_maintenance_rate": round(float(i7["factual_consistency_maintenance_rate"]), 6),
                "degenerate_repetition_run_count": int(i7["degenerate_repetition_run_count"]),
                "structural_output_validity_score": round(float(i7["structural_output_validity_score"]), 6),
            },
            "experiment_i8_multilingual": {
                "english_to_swahili_state_match_rate": round(float(i8["english_to_swahili_state_match_rate"]), 6),
                "japanese_to_mandarin_state_match_rate": round(float(i8["japanese_to_mandarin_state_match_rate"]), 6),
                "cross_lingual_relation_invariance_status": str(i8["cross_lingual_relation_invariance_status"]),
            },
            "experiment_i9_adversarial": {
                "misleading_distractor_bypass_rate": round(float(i9["misleading_distractor_bypass_rate"]), 6),
                "contradictory_fact_rejection_rate": round(float(i9["contradictory_fact_rejection_rate"]), 6),
                "primary_architectural_break_point_log": str(i9["primary_architectural_break_point_log"]),
            },
            "final_scientific_gate_status": final_status,
        }

    def run_suite(self, *, enforce_frozen_lock: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
        if enforce_frozen_lock:
            self._enforce_frozen_config()

        i1 = run_i1()
        i2 = run_i2()
        i3 = run_i3()
        i4 = run_i4()
        i5 = run_i5()
        i6 = run_i6()
        i7 = run_i7()
        i8 = run_i8()
        i9 = run_i9()

        final_status = "PASS"
        if i3["structural_composition_status"] != "PASS":
            final_status = "INCOMPLETE"
        if i8["cross_lingual_relation_invariance_status"] != "PASS":
            final_status = "INCOMPLETE"
        if i9["primary_architectural_break_point_log"] != "none":
            final_status = "INCOMPLETE"
        if i4["closure_validation_accuracy_rate"] < 0.95:
            final_status = "INCOMPLETE"
        if i5["paraphrase_matching_confidence_score"] < float(self.config["implicit_semantics"]["min_paraphrase_similarity"]):
            final_status = "INCOMPLETE"
        if i7["factual_consistency_maintenance_rate"] < 0.8:
            final_status = "INCOMPLETE"

        submission = self.build_submission(
            token=str(self.config["phase_i_capability_token"]),
            i1=i1,
            i2=i2,
            i3=i3,
            i4=i4,
            i5=i5,
            i6=i6,
            i7=i7,
            i8=i8,
            i9=i9,
            final_status=final_status,
        )
        if enforce_frozen_lock:
            self.enforce_hard_gates(submission)
        detail = {"i1": i1, "i2": i2, "i3": i3, "i4": i4, "i5": i5, "i6": i6, "i7": i7, "i8": i8, "i9": i9}
        return submission, detail

    @staticmethod
    def write_results(
        *,
        submission: dict[str, Any],
        detail: dict[str, Any],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "phase_i_submission.json"
        full_path = output_dir / "phase_i_full_results.json"
        summary_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
        full_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
        return summary_path, full_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase I semantic capability lab suite.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments") / "phase_i" / "results",
        help="Directory for phase_i_submission.json and phase_i_full_results.json",
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
    lab = PhaseILab()
    submission, detail = lab.run_suite(enforce_frozen_lock=not args.skip_frozen_lock)
    PhaseILab.write_results(
        submission=submission,
        detail=detail,
        output_dir=args.output_dir,
    )
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
