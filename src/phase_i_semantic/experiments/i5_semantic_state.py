from __future__ import annotations

from typing import Any

import numpy as np

from src.phase_i_semantic.config_loader import load_phase_i_config
from src.phase_i_semantic.semantics.state_representation import extract_tuple, tuple_similarity


def run() -> dict[str, Any]:
    cfg = load_phase_i_config()
    pairs = [
        ("Joseph is building OutreachLM.", "OutreachLM is currently being developed by Joseph."),
        ("Maya is testing software.", "Software is currently being tested by Maya."),
        ("Nina is building platform.", "Platform is currently being developed by Nina."),
    ]
    scores: list[float] = []
    for left, right in pairs:
        left_tuple = extract_tuple(left)
        right_tuple = extract_tuple(right)
        if left_tuple is None or right_tuple is None:
            scores.append(0.0)
            continue
        scores.append(tuple_similarity(left_tuple, right_tuple))

    invariance_rate = sum(1 for score in scores if score >= float(cfg["implicit_semantics"]["min_paraphrase_similarity"])) / len(scores)
    confidence = float(np.mean(np.asarray(scores, dtype=np.float64))) if scores else 0.0
    return {
        "extracted_tuple_invariance_rate": invariance_rate,
        "paraphrase_matching_confidence_score": confidence,
    }
