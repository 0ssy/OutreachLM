from __future__ import annotations

import numpy as np


def normalize_fp32(probabilities: np.ndarray) -> np.ndarray:
    row = np.asarray(probabilities, dtype=np.float32)
    row = np.clip(row, np.float32(1e-12), np.float32(1.0))
    row = row / row.sum(dtype=np.float32)
    return row.astype(np.float64)

