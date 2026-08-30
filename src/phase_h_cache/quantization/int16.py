from __future__ import annotations

import numpy as np


def quantize_int16(probabilities: np.ndarray) -> np.ndarray:
    row = np.asarray(probabilities, dtype=np.float64)
    row = np.clip(row, 1e-12, 1.0)
    row = row / row.sum()
    max_amplitude = float(np.max(row))
    if max_amplitude <= 0.0:
        raise ValueError("Cannot quantize zero-mass vector.")
    quantized = np.floor((row / max_amplitude) * 32767.0).astype(np.int16)
    restored = quantized.astype(np.float64) * (max_amplitude / 32767.0)
    restored = np.clip(restored, 1e-12, 1.0)
    restored = restored / restored.sum()
    return restored

