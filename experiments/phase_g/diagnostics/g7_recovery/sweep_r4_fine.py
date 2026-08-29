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

from sweep_r4 import (  # noqa: E402
    _build_samples,
    _entropy,
    _kl_divergence,
    _load_fixture,
    _measure_scaling,
    _r4_variant,
    _support,
)

PHASE_G_ROOT = Path(__file__).resolve().parents[2]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.metrics import evaluate_predictions  # noqa: E402


def main() -> None:
    fixture, tokenizer, train_sequences, eval_sequences = _load_fixture()
    support_threshold = float(fixture["support_threshold"])
    vocab_size = len(tokenizer.token_to_id)

    top_k = 4
    floor_mix_values = [round(x, 2) for x in np.arange(0.45, 0.651, 0.01)]

    prep_start = time.perf_counter()
    samples = _build_samples(tokenizer, train_sequences, eval_sequences)
    prep_time = time.perf_counter() - prep_start

    g6_rows = [sample["g6"] for sample in samples]
    targets = [int(sample["target"]) for sample in samples]
    g6_metrics = evaluate_predictions(g6_rows, targets)

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    eval_start = time.perf_counter()
    for floor_mix in floor_mix_values:
        preds = []
        entropies = []
        supports = []
        top1 = []
        mass_errors = []
        kls = []
        for sample in samples:
            pred = _r4_variant(sample["g6"], sample["g2"], top_k=top_k, floor_mix=float(floor_mix))
            preds.append(pred)
            entropies.append(_entropy(pred))
            supports.append(float(_support(pred, support_threshold)))
            top1.append(float(np.max(pred)))
            mass_errors.append(abs(float(pred.sum()) - 1.0))
            kls.append(_kl_divergence(sample["g6"], pred))

        metrics = evaluate_predictions(preds, targets)
        scaling = _measure_scaling(samples, top_k=top_k, floor_mix=float(floor_mix))
        row = {
            "top_k": top_k,
            "floor_mix": float(floor_mix),
            "accuracy": metrics["accuracy"],
            "cross_entropy": metrics["cross_entropy"],
            "perplexity": metrics["perplexity"],
            "delta_cross_entropy_vs_g6": float(metrics["cross_entropy"] - g6_metrics["cross_entropy"]),
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "support_mean": float(np.mean(supports)),
            "top1_probability_mean": float(np.mean(top1)),
            "kl_g6_to_r4_mean": float(np.mean(kls)),
            "kl_g6_to_r4_p95": float(np.percentile(np.asarray(kls), 95)),
            "mass_error_max": float(np.max(mass_errors)),
            "ops_dense": int(vocab_size),
            "ops_sparse_estimate": float(top_k + floor_mix * vocab_size),
            "ops_reduction_ratio": float((top_k + floor_mix * vocab_size) / max(vocab_size, 1)),
            "passes_non_catastrophic_gate": bool((metrics["cross_entropy"] - g6_metrics["cross_entropy"]) <= 1.0),
            "strict_no_increase": bool(metrics["cross_entropy"] <= g6_metrics["cross_entropy"]),
            "single_cpu_throughput": scaling["single_cpu_throughput"],
            "two_cpu_throughput": scaling["two_cpu_throughput"],
            "scaling_efficiency": scaling["scaling_efficiency"],
            "useful_parallelism": scaling["useful_parallelism"],
        }
        rows.append(row)
        details.append({"floor_mix": float(floor_mix), "row": row, "scaling": scaling})

    eval_time = time.perf_counter() - eval_start

    best_by_ce = min(rows, key=lambda item: float(item["cross_entropy"]))
    best_strict = min(
        [item for item in rows if item["strict_no_increase"]],
        key=lambda item: float(item["cross_entropy"]),
        default=None,
    )

    result = {
        "experiment_id": "g7_r4_fine_sweep",
        "seed": int(fixture["seed"]),
        "fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "grid": {"top_k": top_k, "floor_mix_values": floor_mix_values},
        "g6_reference": {
            "accuracy": g6_metrics["accuracy"],
            "cross_entropy": g6_metrics["cross_entropy"],
            "perplexity": g6_metrics["perplexity"],
            "count": int(g6_metrics["count"]),
        },
        "best": {
            "best_by_cross_entropy": best_by_ce,
            "best_strict_no_increase": best_strict,
        },
        "resource_usage": {
            "prep_time_seconds": prep_time,
            "evaluation_time_seconds": eval_time,
            "grid_size": len(rows),
        },
        "summary_rows": rows,
        "details": details,
    }

    summary = {
        "experiment_id": "g7_r4_fine_sweep_summary",
        "seed": int(fixture["seed"]),
        "grid": result["grid"],
        "g6_reference": result["g6_reference"],
        "best": result["best"],
        "resource_usage": result["resource_usage"],
        "summary_rows": rows,
    }

    results_dir = HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "g7_r4_fine_sweep_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (results_dir / "g7_r4_fine_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r4_fine_sweep_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r4_fine_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g7_r4_fine_sweep complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
