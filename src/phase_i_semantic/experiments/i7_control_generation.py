from __future__ import annotations

from typing import Any

import numpy as np

from src.phase_i_semantic.config_loader import load_phase_i_config
from src.phase_i_semantic.experiments.common import load_runtime
from src.phase_i_semantic.generation.controlled_core import run_controlled_generation


def run() -> dict[str, Any]:
    cfg = load_phase_i_config()
    runtime = load_runtime()
    prompts = [
        "Joseph builds systems with careful testing",
        "The bank approved the loan after review",
        "Nested structures should remain valid",
    ]
    outputs = [
        run_controlled_generation(
            runtime,
            prompt=prompt,
            max_tokens=80,
            max_depth=int(cfg["structural_syntax"]["max_nesting_depth"]),
        )
        for prompt in prompts
    ]

    factual_hits = 0
    repetition_run_count = 0
    structural_scores: list[float] = []
    for out in outputs:
        factual_hits += 1 if out["anchor_hits"] >= max(1, out["anchor_target"] - 1) else 0
        repetition_run_count += 1 if out["max_repetition_run"] >= 20 else 0
        structural_scores.append(1.0 if out["closure_valid"] else 0.0)
    return {
        "factual_consistency_maintenance_rate": factual_hits / len(prompts),
        "degenerate_repetition_run_count": repetition_run_count,
        "structural_output_validity_score": float(np.mean(np.asarray(structural_scores, dtype=np.float64))),
    }
