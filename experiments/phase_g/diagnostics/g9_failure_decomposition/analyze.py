from __future__ import annotations

import json
from pathlib import Path
import statistics


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def main() -> None:
    root = Path(__file__).resolve().parent
    result_path = root / "results" / "g9_failure_decomposition.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected result file at: {result_path}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    stage_metrics = result["stage_metrics"]
    stage_deltas = result["stage_deltas"]
    kl = result["kl_divergence"]["summary"]
    traces = result["sample_traces"]

    analysis = {
        "experiment_id": "g9_failure_decomposition_analysis",
        "evaluated_samples": len(traces),
        "stages_by_cross_entropy_ascending": sorted(
            (
                {
                    "stage": stage,
                    "cross_entropy": metrics["cross_entropy"],
                    "perplexity": metrics["perplexity"],
                    "entropy_mean": metrics["entropy_mean"],
                    "support_mean": metrics["support_mean"],
                    "top1_probability_mean": metrics["top1_probability_mean"],
                }
                for stage, metrics in stage_metrics.items()
            ),
            key=lambda item: item["cross_entropy"],
        ),
        "largest_ce_increase_edge": max(
            stage_deltas["cross_entropy"].items(),
            key=lambda item: item[1],
        ),
        "largest_entropy_shift_edge": max(
            stage_deltas["entropy_mean"].items(),
            key=lambda item: abs(item[1]),
        ),
        "largest_support_shift_edge": max(
            stage_deltas["support_mean"].items(),
            key=lambda item: abs(item[1]),
        ),
        "largest_mean_kl_edge": result["kl_divergence"]["largest_mean_kl_edge"],
        "largest_p95_kl_edge": result["kl_divergence"]["largest_p95_kl_edge"],
        "kl_table": kl,
        "mean_sample_kl_chain": {
            "g2_g3": _mean([trace["kl_chain"]["g2_g3"] for trace in traces]),
            "g3_g4": _mean([trace["kl_chain"]["g3_g4"] for trace in traces]),
            "g4_g5": _mean([trace["kl_chain"]["g4_g5"] for trace in traces]),
            "g5_g6": _mean([trace["kl_chain"]["g5_g6"] for trace in traces]),
            "g6_g7": _mean([trace["kl_chain"]["g6_g7"] for trace in traces]),
            "g7_g9": _mean([trace["kl_chain"]["g7_g9"] for trace in traces]),
        },
    }

    output_path = root / "results" / "g9_failure_decomposition_analysis.json"
    output_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print("g9_failure_decomposition analysis complete")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
