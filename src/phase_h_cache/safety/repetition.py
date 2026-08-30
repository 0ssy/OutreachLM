from __future__ import annotations

from collections import Counter

import numpy as np


def apply_repetition_penalty(
    probabilities: np.ndarray,
    recent_tokens: list[int],
    *,
    decay: float = 0.85,
    floor: float = 0.25,
) -> np.ndarray:
    row = np.asarray(probabilities, dtype=np.float64).copy()
    if not recent_tokens:
        return row / row.sum()
    counts = Counter(recent_tokens)
    for token_id, repeats in counts.items():
        if 0 <= token_id < len(row):
            multiplier = max(floor, decay**repeats)
            row[token_id] *= multiplier
    row = np.clip(row, 1e-12, 1.0)
    row = row / row.sum()
    return row

