from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


PHASE_G_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(script_path: Path) -> None:
    subprocess.run([PYTHON, str(script_path)], check=True)


def main() -> None:
    _run(PHASE_G_ROOT / "g2_contextual_transition" / "train.py")
    _run(PHASE_G_ROOT / "g2_contextual_transition" / "evaluate.py")
    _run(PHASE_G_ROOT / "g2_contextual_transition" / "context_test.py")
    _run(PHASE_G_ROOT / "g3_representation" / "run.py")
    _run(PHASE_G_ROOT / "g4_context_compression" / "run.py")
    _run(PHASE_G_ROOT / "g5_adaptive_memory" / "run.py")
    _run(PHASE_G_ROOT / "g6_hierarchical_structure" / "run.py")
    _run(PHASE_G_ROOT / "g7_sparse_prediction" / "run.py")
    _run(PHASE_G_ROOT / "g8_multi_cpu" / "run.py")
    _run(PHASE_G_ROOT / "g9_integrated_model" / "run.py")
    _run(PHASE_G_ROOT / "g10_final_validation" / "run.py")

    summary = {
        "g2": json.loads((PHASE_G_ROOT / "g2_contextual_transition" / "results" / "g2_result.json").read_text(encoding="utf-8")),
        "g3": json.loads((PHASE_G_ROOT / "g3_representation" / "results" / "g3_result.json").read_text(encoding="utf-8")),
        "g4": json.loads((PHASE_G_ROOT / "g4_context_compression" / "results" / "g4_result.json").read_text(encoding="utf-8")),
        "g5": json.loads((PHASE_G_ROOT / "g5_adaptive_memory" / "results" / "g5_result.json").read_text(encoding="utf-8")),
        "g6": json.loads((PHASE_G_ROOT / "g6_hierarchical_structure" / "results" / "g6_result.json").read_text(encoding="utf-8")),
        "g7": json.loads((PHASE_G_ROOT / "g7_sparse_prediction" / "results" / "g7_result.json").read_text(encoding="utf-8")),
        "g8": json.loads((PHASE_G_ROOT / "g8_multi_cpu" / "results" / "g8_result.json").read_text(encoding="utf-8")),
        "g9": json.loads((PHASE_G_ROOT / "g9_integrated_model" / "results" / "g9_result.json").read_text(encoding="utf-8")),
        "g10": json.loads((PHASE_G_ROOT / "g10_final_validation" / "results" / "g10_result.json").read_text(encoding="utf-8")),
    }
    (PHASE_G_ROOT / "results").mkdir(parents=True, exist_ok=True)
    (PHASE_G_ROOT / "results" / "g2_to_g10_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("G2 to G10 pipeline complete")


if __name__ == "__main__":
    main()
