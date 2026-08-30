from __future__ import annotations

import json
from pathlib import Path

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


def main() -> None:
    cfg = load_phase_i_config()
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
    if i5["paraphrase_matching_confidence_score"] < float(cfg["implicit_semantics"]["min_paraphrase_similarity"]):
        final_status = "INCOMPLETE"
    if i7["factual_consistency_maintenance_rate"] < 0.8:
        final_status = "INCOMPLETE"

    submission = {
        "phase_i_capability_token": cfg["phase_i_capability_token"],
        "experiment_i1_memory": i1,
        "experiment_i2_entities": i2,
        "experiment_i3_composition": i3,
        "experiment_i4_syntax": i4,
        "experiment_i5_semantics": i5,
        "experiment_i6_retrieval": i6,
        "experiment_i7_generation": i7,
        "experiment_i8_multilingual": i8,
        "experiment_i9_adversarial": i9,
        "final_scientific_gate_status": final_status,
    }

    output_dir = Path("experiments") / "phase_i" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_i_submission.json").write_text(json.dumps(submission, indent=2), encoding="utf-8")
    print(json.dumps(submission, indent=2))


if __name__ == "__main__":
    main()
