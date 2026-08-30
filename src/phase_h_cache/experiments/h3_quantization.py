from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from outreachlm.phase_g_bridge import PhaseGHybridRuntime
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.quantization.fp32_reference import normalize_fp32
from src.phase_h_cache.quantization.int16 import quantize_int16
from src.phase_h_cache.quantization.int8 import quantize_int8


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_clip = np.clip(p, 1e-12, 1.0)
    q_clip = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p_clip * np.log(p_clip / q_clip)))


def _quantize_fp16(row: np.ndarray) -> np.ndarray:
    out = np.asarray(row, dtype=np.float16).astype(np.float64)
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum()
    return out


def run() -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw["h3"]
    runtime, train_lines, eval_lines = PhaseGHybridRuntime.from_corpus_path(
        CORPUS_PATH,
        validation_split=VALIDATION_SPLIT,
        max_train_lines=int(config["train_lines"]),
        max_eval_lines=int(config["eval_lines"]),
    )

    max_rows = int(config["max_rows"])
    contexts: list[list[int]] = []
    for line in eval_lines:
        seq = runtime.tokenizer.encode(line, add_bos=True, add_eos=True)
        for pos in range(len(seq) - 1):
            contexts.append(seq[: pos + 1])
            if len(contexts) >= max_rows:
                break
        if len(contexts) >= max_rows:
            break
    if not contexts:
        raise ValueError("No contexts available for quantization evaluation.")

    rows_fp32 = [normalize_fp32(runtime.distribution(context)) for context in contexts]

    engines = {
        "fp16": _quantize_fp16,
        "int16": quantize_int16,
        "int8": quantize_int8,
    }
    engine_metrics: dict[str, dict[str, float]] = {}
    for name, quantizer in engines.items():
        mass_errors: list[float] = []
        kls: list[float] = []
        for row in rows_fp32:
            q = quantizer(row)
            mass_errors.append(abs(float(q.sum()) - 1.0))
            kls.append(_kl_divergence(row, q))
        engine_metrics[name] = {
            "mass_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
            "kl_mean": float(np.mean(np.asarray(kls, dtype=np.float64))),
            "kl_p95": float(np.percentile(np.asarray(kls, dtype=np.float64), 95.0)),
        }

    mass_gate = float(config["mass_error_gate"])
    kl_gate = float(config["kl_gate"])
    candidates = [
        (name, metrics)
        for name, metrics in engine_metrics.items()
        if metrics["mass_error_max"] <= mass_gate and metrics["kl_mean"] <= kl_gate
    ]
    if candidates:
        winning = min(candidates, key=lambda item: item[1]["kl_mean"])[0]
    else:
        winning = "fp32"

    winning_metrics = engine_metrics.get(winning, {"mass_error_max": 0.0, "kl_mean": 0.0})
    return {
        "experiment_id": "h3_quantization_divergence",
        "config": config,
        "sample_count": len(rows_fp32),
        "engine_metrics": engine_metrics,
        "gates": {
            "mass_retention_threshold": mass_gate,
            "semantic_preservation_kl_threshold": kl_gate,
        },
        "winning_precision_format": winning,
        "measured_mass_error": float(winning_metrics["mass_error_max"]),
        "measured_kl_divergence_vs_g7": float(winning_metrics["kl_mean"]),
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h3_quantization.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
