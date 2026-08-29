from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def distribution_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def evaluate_predictions(probability_rows: Iterable[np.ndarray], targets: Iterable[int]) -> dict[str, float]:
    total = 0
    correct = 0
    nll_sum = 0.0
    for probs, target in zip(probability_rows, targets):
        prediction = int(np.argmax(probs))
        if prediction == int(target):
            correct += 1
        p = float(probs[int(target)])
        nll_sum += -math.log(max(p, 1e-12))
        total += 1
    if total == 0:
        raise ValueError("No samples provided for evaluation.")
    cross_entropy = nll_sum / total
    return {
        "accuracy": correct / total,
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy),
        "count": float(total),
    }
