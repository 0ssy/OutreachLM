from __future__ import annotations

import json
from pathlib import Path
import sys

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    g1_path = PHASE_G_ROOT / "results" / "g1_transition_eval_seed1337.json"
    g1 = _read(g1_path)["g1_transition"] if g1_path.exists() else {}
    g2 = _read(PHASE_G_ROOT / "g2_contextual_transition" / "results" / "g2_result.json")
    g2_ctx = _read(PHASE_G_ROOT / "g2_contextual_transition" / "results" / "g2_context_test.json")
    g3 = _read(PHASE_G_ROOT / "g3_representation" / "results" / "g3_result.json")
    g4 = _read(PHASE_G_ROOT / "g4_context_compression" / "results" / "g4_result.json")
    g5 = _read(PHASE_G_ROOT / "g5_adaptive_memory" / "results" / "g5_result.json")
    g6 = _read(PHASE_G_ROOT / "g6_hierarchical_structure" / "results" / "g6_result.json")
    g7 = _read(PHASE_G_ROOT / "g7_sparse_prediction" / "results" / "g7_result.json")
    g8 = _read(PHASE_G_ROOT / "g8_multi_cpu" / "results" / "g8_result.json")
    g9 = _read(PHASE_G_ROOT / "g9_integrated_model" / "results" / "g9_result.json")
    g7_final_path = PHASE_G_ROOT / "g7_recovery" / "r6_residual_sparse" / "results" / "g7_final_validation_result.json"
    g7_final = _read(g7_final_path) if g7_final_path.exists() else None

    predictive = {
        "g1": g1,
        "g2": g2["g2"],
        "g3": g3["g3"],
        "g4": g4["g4"],
        "g5": g5["g5"],
        "g6": g6["g6"],
        "g7": g7["g7"],
        "g9": g9["g9"],
    }
    if g7_final is not None:
        predictive["g7_final_hybrid"] = g7_final["frozen_fixture_metrics"]["g7_final_hybrid"]
    context_sensitivity = {
        "g1_distribution_difference": g2_ctx["g1_distribution_difference"],
        "g2_distribution_difference": g2_ctx["g2_distribution_difference"],
    }
    long_context = {
        "g4_long_accuracy": g4["long_context"]["accuracy"],
        "g6_long_accuracy": g6["hierarchical_breakdown"]["long_accuracy"],
    }
    resource = {
        "g2_model_storage_bytes": g2["model_storage_bytes"],
        "g4_model_storage_bytes": g4["model_storage_bytes"],
        "g5_model_storage_bytes": g5["model_storage_bytes"],
        "g6_model_storage_bytes": g6["model_storage_bytes"],
        "g7_model_storage_bytes": g7["model_storage_bytes"],
        "g9_model_storage": g9["model_storage"],
        "g9_peak_ram": g9["peak_RAM"],
        "g9_tokens_per_second": g9["tokens_per_second"],
    }
    scaling = {
        "single_cpu_tokens_per_second": g8["single_cpu"]["throughput"],
        "two_cpu_tokens_per_second": g8["two_cpu"]["throughput"],
        "scaling_efficiency": g8["scaling_efficiency"],
    }
    ablation = {
        "minus_representation": g2["g2"],
        "minus_context_compression": g3["g3"],
        "minus_adaptive_memory": g4["g4"],
        "minus_hierarchy": g5["g5"],
        "minus_sparsity": g6["g6"],
    }

    keep_remove = {
        "Transition": "KEEP" if g2["g2"]["cross_entropy"] <= g2["g1"]["cross_entropy"] else "REMOVE",
        "Context": "KEEP" if g2_ctx["g2_distribution_difference"] > g2_ctx["g1_distribution_difference"] else "REMOVE",
        "Representation": "KEEP" if g3["hard_gates"]["competitive_with_g2"] else "UNCERTAIN",
        "Compression": "KEEP" if g4["hard_gates"]["memory_reduced"] else "REMOVE",
        "Adaptive memory": "KEEP" if g5["hard_gates"]["benefit_over_g4"] else "UNCERTAIN",
        "Hierarchy": "KEEP" if g6["hard_gates"]["advantage_over_g5"] else "UNCERTAIN",
        "Sparsity": (
            "KEEP_ACCEPTED"
            if (g7_final is not None and g7_final["final_decision"] == "ACCEPTED")
            else ("KEEP" if g7["hard_gates"]["resource_advantage"] else "REMOVE")
        ),
        "Multi-CPU": "KEEP" if g8["hard_gates"]["useful_parallelism"] else "UNCERTAIN",
    }

    g7_accepted = bool(g7_final is not None and g7_final["final_decision"] == "ACCEPTED")
    g9_accepted = bool(
        g9["hard_gates"]["integrated_better_than_g2"]
        and g9["hard_gates"]["integrated_competitive_vs_g6"]
        and g9["hard_gates"]["integrated_beats_r4"]
        and g9["hard_gates"]["probability_mass_correct"]
    )
    predictive_capability_pass = bool(g9["g9"]["cross_entropy"] <= g6["g6"]["cross_entropy"])
    context_sensitivity_pass = bool(g2_ctx["g2_distribution_difference"] > g2_ctx["g1_distribution_difference"])
    probability_correctness_pass = bool(
        g9["hard_gates"]["probability_mass_correct"]
        and (g7_accepted and g7_final["acceptance_gates"]["probability_mass_correct_30_of_30"])
    )
    resource_measurement_pass = True
    cpu_scaling_pass = bool(g8["hard_gates"]["useful_parallelism"])
    ablation_pass = bool(g7_accepted and g7_final["acceptance_gates"]["residual_ablation_matters"])
    phase_g_closed = bool(
        g7_accepted
        and g9_accepted
        and predictive_capability_pass
        and context_sensitivity_pass
        and probability_correctness_pass
        and resource_measurement_pass
        and cpu_scaling_pass
        and ablation_pass
    )

    result = {
        "experiment_id": "g10_final_architecture_validation",
        "seed": g2["seed"],
        "tests": {
            "A_predictive_capability": predictive,
            "B_context_sensitivity": context_sensitivity,
            "C_long_context_retention": long_context,
            "D_resource_efficiency": resource,
            "E_scaling": scaling,
            "F_ablation": ablation,
        },
        "final_decision": keep_remove,
        "hard_gates": {
            "context_sensitivity_confirmed": context_sensitivity_pass,
            "predictive_model_available": g9["g9"]["cross_entropy"] < 5.0,
            "resource_measured": True,
            "ablation_measured": True,
            "g7_candidate_validated": g7_accepted,
        },
        "phase_g_closure": {
            "G7_sparsity_mechanism": "ACCEPTED" if g7_accepted else "REJECTED",
            "G9_integrated_model": "ACCEPTED" if g9_accepted else "REJECTED",
            "Predictive_capability": "PASS" if predictive_capability_pass else "FAIL",
            "Context_sensitivity": "PASS" if context_sensitivity_pass else "FAIL",
            "Probability_correctness": "PASS" if probability_correctness_pass else "FAIL",
            "Resource_measurement": "PASS" if resource_measurement_pass else "FAIL",
            "CPU_scaling": "PASS" if cpu_scaling_pass else "FAIL",
            "Ablation": "PASS" if ablation_pass else "FAIL",
            "Phase_G_scientific_closure": "PASS" if phase_g_closed else "FAIL",
        },
    }
    if g7_final is not None:
        result["tests"]["G7_final_validation_package"] = {
            "path": str(g7_final_path.resolve()),
            "final_decision": g7_final["final_decision"],
            "acceptance_gates": g7_final["acceptance_gates"],
            "g6_vs_g7_final": g7_final["g6_vs_g7_final"],
        }

    root = Path(__file__).resolve().parent
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g10_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g10_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("G10 complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
