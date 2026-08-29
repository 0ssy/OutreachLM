from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))

from sweep_r4 import _build_samples, _entropy, _kl_divergence, _load_fixture, _r4_variant  # noqa: E402

PHASE_G_ROOT = Path(__file__).resolve().parents[2]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.metrics import evaluate_predictions  # noqa: E402


def _support(probabilities: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(probabilities > threshold))


def _classify(
    *,
    gate_quality: bool,
    gate_sparsity: bool,
    gate_probability: bool,
    ce: float,
    g6_ce: float,
) -> str:
    if not (gate_quality and gate_sparsity and gate_probability):
        return "REJECT"
    if ce <= g6_ce:
        return "COMPETITIVE"
    return "RECOVERED"


def main() -> None:
    fixture, tokenizer, train_sequences, eval_sequences = _load_fixture()
    support_threshold = float(fixture["support_threshold"])
    vocab_size = len(tokenizer.token_to_id)

    top_k_grid = [2, 4, 6, 8, 10, 12, 16, 24, 32, 46]
    floor_mix_grid = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]

    prep_start = time.perf_counter()
    samples = _build_samples(tokenizer, train_sequences, eval_sequences)
    prep_time = time.perf_counter() - prep_start

    g6_rows = [sample["g6"] for sample in samples]
    targets = [int(sample["target"]) for sample in samples]
    g6_metrics = evaluate_predictions(g6_rows, targets)
    g6_ce = float(g6_metrics["cross_entropy"])
    g6_threshold = g6_ce + 0.10
    dense_ops = float(vocab_size)

    rows: list[dict[str, Any]] = []
    eval_start = time.perf_counter()
    for top_k in top_k_grid:
        for floor_mix in floor_mix_grid:
            preds: list[np.ndarray] = []
            entropies: list[float] = []
            supports: list[float] = []
            kls: list[float] = []
            mass_errors: list[float] = []

            inference_start = time.perf_counter()
            for sample in samples:
                pred = _r4_variant(sample["g6"], sample["g2"], top_k=top_k, floor_mix=floor_mix)
                preds.append(pred)
                entropies.append(_entropy(pred))
                supports.append(float(_support(pred, support_threshold)))
                kls.append(_kl_divergence(sample["g6"], pred))
                mass_errors.append(abs(float(pred.sum()) - 1.0))
            inference_time = time.perf_counter() - inference_start

            metrics = evaluate_predictions(preds, targets)
            tokens_per_second = float(metrics["count"] / max(inference_time, 1e-12))
            ops_per_token = float(top_k + floor_mix * vocab_size)
            operation_ratio = float(ops_per_token / dense_ops)

            ce = float(metrics["cross_entropy"])
            max_mass_error = float(np.max(np.asarray(mass_errors, dtype=np.float64)))

            gate_quality = ce <= g6_threshold
            gate_sparsity = ops_per_token < dense_ops
            gate_probability = max_mass_error <= 1e-6
            classification = _classify(
                gate_quality=gate_quality,
                gate_sparsity=gate_sparsity,
                gate_probability=gate_probability,
                ce=ce,
                g6_ce=g6_ce,
            )

            rows.append(
                {
                    "top_k": int(top_k),
                    "floor_mix": float(floor_mix),
                    "accuracy": float(metrics["accuracy"]),
                    "cross_entropy": ce,
                    "perplexity": float(metrics["perplexity"]),
                    "entropy_mean": float(np.mean(np.asarray(entropies, dtype=np.float64))),
                    "support_mean": float(np.mean(np.asarray(supports, dtype=np.float64))),
                    "kl_g6_to_candidate_mean": float(np.mean(np.asarray(kls, dtype=np.float64))),
                    "mass_error_max": max_mass_error,
                    "operations_per_token": ops_per_token,
                    "operation_ratio": operation_ratio,
                    "inference_time_seconds": float(inference_time),
                    "tokens_per_second": tokens_per_second,
                    "gates": {
                        "gate_1_quality": bool(gate_quality),
                        "gate_2_actual_sparsity": bool(gate_sparsity),
                        "gate_3_probability_correctness": bool(gate_probability),
                    },
                    "classification": classification,
                }
            )
    evaluation_time = time.perf_counter() - eval_start

    accepted_rows = [row for row in rows if row["classification"] != "REJECT"]
    best_row = min(accepted_rows if accepted_rows else rows, key=lambda row: float(row["cross_entropy"]))

    for row in rows:
        row["is_best"] = bool(
            row["top_k"] == best_row["top_k"]
            and abs(float(row["floor_mix"]) - float(best_row["floor_mix"])) < 1e-12
            and abs(float(row["cross_entropy"]) - float(best_row["cross_entropy"])) < 1e-12
        )

    summary_counts = {
        "REJECT": int(sum(1 for row in rows if row["classification"] == "REJECT")),
        "RECOVERED": int(sum(1 for row in rows if row["classification"] == "RECOVERED")),
        "COMPETITIVE": int(sum(1 for row in rows if row["classification"] == "COMPETITIVE")),
        "BEST": int(sum(1 for row in rows if row["is_best"])),
    }

    result = {
        "experiment_id": "g7_r4_protocol60_sweep",
        "seed": int(fixture["seed"]),
        "fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "protocol_grid": {
            "top_k": top_k_grid,
            "floor_mix": floor_mix_grid,
            "config_count": len(top_k_grid) * len(floor_mix_grid),
        },
        "g6_reference": {
            "accuracy": float(g6_metrics["accuracy"]),
            "cross_entropy": g6_ce,
            "perplexity": float(g6_metrics["perplexity"]),
            "count": int(g6_metrics["count"]),
            "quality_gate_threshold": float(g6_threshold),
            "dense_operations_per_token": dense_ops,
        },
        "resource_usage": {
            "prep_time_seconds": float(prep_time),
            "evaluation_time_seconds": float(evaluation_time),
            "eval_token_count": len(samples),
        },
        "summary_counts": summary_counts,
        "best_configuration": best_row,
        "rows": rows,
    }

    summary = {
        "experiment_id": "g7_r4_protocol60_sweep_summary",
        "seed": int(fixture["seed"]),
        "summary_counts": summary_counts,
        "best_configuration": best_row,
        "g6_reference": result["g6_reference"],
        "resource_usage": result["resource_usage"],
        "rows": rows,
    }

    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "g7_r4_protocol60_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (results_dir / "g7_r4_protocol60_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r4_protocol60_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r4_protocol60_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g7_r4_protocol60_sweep complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
