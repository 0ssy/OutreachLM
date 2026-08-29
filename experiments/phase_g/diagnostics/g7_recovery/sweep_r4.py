from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PHASE_G_ROOT = Path(__file__).resolve().parents[2]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.phase_components import AdaptiveMemoryModel, CompressedContextModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402


def _cooccurrence_embeddings(sequences: list[list[int]], vocab_size: int, window: int = 2) -> np.ndarray:
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for sequence in sequences:
        for index, token in enumerate(sequence):
            start = max(0, index - window)
            end = min(len(sequence), index + window + 1)
            for j in range(start, end):
                if index == j:
                    continue
                matrix[token, sequence[j]] += 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _entropy(probabilities: np.ndarray, epsilon: float = 1e-12) -> float:
    p = np.clip(probabilities, epsilon, 1.0)
    return float(-(p * np.log(p)).sum())


def _support(probabilities: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(probabilities > threshold))


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _load_fixture() -> tuple[dict[str, Any], StupidTokenizer, list[list[int]], list[list[int]]]:
    fixture_path = PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tokenizer = StupidTokenizer(token_to_id=fixture["token_to_id"])
    train_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["train_lines"]]
    eval_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["eval_lines"]]
    return fixture, tokenizer, train_sequences, eval_sequences


def _build_samples(
    tokenizer: StupidTokenizer,
    train_sequences: list[list[int]],
    eval_sequences: list[list[int]],
) -> list[dict[str, Any]]:
    vocab_size = len(tokenizer.token_to_id)
    g2_model = SparseNGramModel(vocab_size=vocab_size, order=2, alpha=0.1)
    g2_model.fit(train_sequences)
    embeddings = _cooccurrence_embeddings(train_sequences, vocab_size=vocab_size, window=2)
    g4_model = CompressedContextModel(vocab_size=vocab_size, context_length=4, bucket_count=32, alpha=0.1)
    g4_model.fit(train_sequences)
    g5_model = AdaptiveMemoryModel(vocab_size=vocab_size, memory_size=3, window_size=6, alpha=0.1)
    g5_model.fit(train_sequences)
    local = SparseNGramModel(vocab_size=vocab_size, order=2, alpha=0.1)
    medium = SparseNGramModel(vocab_size=vocab_size, order=4, alpha=0.1)
    global_model = SparseNGramModel(vocab_size=vocab_size, order=1, alpha=0.1)
    local.fit(train_sequences)
    medium.fit(train_sequences)
    global_model.fit(train_sequences)

    samples: list[dict[str, Any]] = []
    for sequence in eval_sequences:
        for pos in range(len(sequence) - 1):
            context = sequence[: pos + 1]
            target = int(sequence[pos + 1])

            g2 = g2_model.distribution(context)

            if len(context) < 2:
                g3 = g2.copy()
            else:
                vector = embeddings[int(context[-2])] + embeddings[int(context[-1])]
                sims = embeddings @ vector
                sims = np.maximum(sims, 0.0)
                if sims.sum() <= 0.0:
                    g3 = g2.copy()
                else:
                    smooth = (sims / sims.sum()) * 0.3
                    g3 = 0.7 * g2 + smooth
                    g3 = g3 / g3.sum()

            g4 = g4_model.distribution(context)
            g5 = g5_model.distribution(context)
            g6 = 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)
            g6 = g6 / g6.sum()

            samples.append(
                {
                    "target": target,
                    "g2": g2,
                    "g3": g3,
                    "g4": g4,
                    "g5": g5,
                    "g6": g6,
                }
            )
    return samples


def _r4_variant(g6: np.ndarray, g2: np.ndarray, top_k: int, floor_mix: float) -> np.ndarray:
    if top_k >= len(g6):
        sparse = g6.copy()
    else:
        idx = np.argpartition(g6, -top_k)[-top_k:]
        sparse = np.zeros_like(g6)
        sparse[idx] = g6[idx]
        sparse = sparse / sparse.sum()
    out = (1.0 - floor_mix) * sparse + floor_mix * g2
    return out / out.sum()


def _scaling_worker(payload: tuple[list[dict[str, Any]], int, float]) -> int:
    batch, top_k, floor_mix = payload
    for sample in batch:
        _ = _r4_variant(sample["g6"], sample["g2"], top_k=top_k, floor_mix=floor_mix)
    return len(batch)


def _measure_scaling(samples: list[dict[str, Any]], top_k: int, floor_mix: float) -> dict[str, float]:
    expanded = samples * 2000
    single_start = time.perf_counter()
    single_count = _scaling_worker((expanded, top_k, floor_mix))
    single_elapsed = time.perf_counter() - single_start

    mid = len(expanded) // 2
    a = expanded[:mid]
    b = expanded[mid:]
    multi_start = time.perf_counter()
    with mp.get_context("spawn").Pool(processes=2) as pool:
        out = pool.map(_scaling_worker, [(a, top_k, floor_mix), (b, top_k, floor_mix)])
    multi_elapsed = time.perf_counter() - multi_start
    multi_count = int(out[0] + out[1])

    single_tps = single_count / max(single_elapsed, 1e-12)
    multi_tps = multi_count / max(multi_elapsed, 1e-12)
    return {
        "single_cpu_throughput": float(single_tps),
        "two_cpu_throughput": float(multi_tps),
        "single_cpu_elapsed_seconds": float(single_elapsed),
        "two_cpu_elapsed_seconds": float(multi_elapsed),
        "latency_ratio_two_over_one": float(multi_elapsed / max(single_elapsed, 1e-12)),
        "scaling_efficiency": float(multi_tps / max(2.0 * single_tps, 1e-12)),
        "useful_parallelism": bool(multi_tps > single_tps),
    }


def main() -> None:
    fixture, tokenizer, train_sequences, eval_sequences = _load_fixture()
    support_threshold = float(fixture["support_threshold"])
    vocab_size = len(tokenizer.token_to_id)

    top_k_grid = [2, 4, 8, 16, 32, vocab_size]
    floor_mix_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    prep_start = time.perf_counter()
    samples = _build_samples(tokenizer, train_sequences, eval_sequences)
    prep_time = time.perf_counter() - prep_start

    g6_rows = [sample["g6"] for sample in samples]
    targets = [int(sample["target"]) for sample in samples]
    g6_metrics = evaluate_predictions(g6_rows, targets)

    sweep_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    eval_start = time.perf_counter()
    for top_k in top_k_grid:
        for floor_mix in floor_mix_grid:
            rows: list[np.ndarray] = []
            entropies: list[float] = []
            supports: list[float] = []
            top1: list[float] = []
            mass_errors: list[float] = []
            kl_values: list[float] = []
            for sample in samples:
                pred = _r4_variant(sample["g6"], sample["g2"], top_k=top_k, floor_mix=floor_mix)
                rows.append(pred)
                entropies.append(_entropy(pred))
                supports.append(float(_support(pred, support_threshold)))
                top1.append(float(np.max(pred)))
                mass_errors.append(abs(float(pred.sum()) - 1.0))
                kl_values.append(_kl_divergence(sample["g6"], pred))

            metrics = evaluate_predictions(rows, targets)
            scaling = _measure_scaling(samples, top_k=top_k, floor_mix=floor_mix)

            row = {
                "top_k": int(top_k),
                "floor_mix": float(floor_mix),
                "sparsity_ratio": float(1.0 - (min(top_k, vocab_size) / vocab_size)),
                "accuracy": metrics["accuracy"],
                "cross_entropy": metrics["cross_entropy"],
                "perplexity": metrics["perplexity"],
                "delta_cross_entropy_vs_g6": float(metrics["cross_entropy"] - g6_metrics["cross_entropy"]),
                "entropy_mean": float(np.mean(entropies)),
                "entropy_std": float(np.std(entropies)),
                "support_mean": float(np.mean(supports)),
                "support_min": float(np.min(supports)),
                "support_max": float(np.max(supports)),
                "top1_probability_mean": float(np.mean(top1)),
                "kl_g6_to_r4_mean": float(np.mean(kl_values)),
                "kl_g6_to_r4_p95": float(np.percentile(np.asarray(kl_values), 95)),
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
            sweep_rows.append(row)
            details.append(
                {
                    "config": {"top_k": top_k, "floor_mix": floor_mix},
                    "metrics": row,
                    "distribution_diagnostics": {
                        "entropy": _summary(entropies),
                        "support": _summary(supports),
                        "top1_probability": _summary(top1),
                        "kl_g6_to_r4": _summary(kl_values),
                        "probability_mass_error": _summary(mass_errors),
                    },
                    "scaling": scaling,
                }
            )
    eval_time = time.perf_counter() - eval_start

    best_by_ce = min(sweep_rows, key=lambda x: float(x["cross_entropy"]))
    best_with_gate = min(
        [row for row in sweep_rows if row["passes_non_catastrophic_gate"]],
        key=lambda x: float(x["cross_entropy"]),
    )
    best_strict = min(
        [row for row in sweep_rows if row["strict_no_increase"]],
        key=lambda x: float(x["cross_entropy"]),
        default=None,
    )

    result = {
        "experiment_id": "g7_r4_parameter_sweep",
        "seed": int(fixture["seed"]),
        "fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "grid": {"top_k": top_k_grid, "floor_mix": floor_mix_grid},
        "g6_reference": {
            "accuracy": g6_metrics["accuracy"],
            "cross_entropy": g6_metrics["cross_entropy"],
            "perplexity": g6_metrics["perplexity"],
            "count": int(g6_metrics["count"]),
        },
        "resource_usage": {
            "prep_time_seconds": prep_time,
            "evaluation_time_seconds": eval_time,
            "grid_size": len(sweep_rows),
        },
        "best": {
            "best_by_cross_entropy": best_by_ce,
            "best_with_non_catastrophic_gate": best_with_gate,
            "best_strict_no_increase": best_strict,
        },
        "summary_rows": sweep_rows,
        "details": details,
    }

    summary = {
        "experiment_id": "g7_r4_parameter_sweep_summary",
        "seed": int(fixture["seed"]),
        "grid": result["grid"],
        "g6_reference": result["g6_reference"],
        "best": result["best"],
        "resource_usage": result["resource_usage"],
        "summary_rows": sweep_rows,
    }

    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "g7_r4_sweep_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (results_dir / "g7_r4_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r4_sweep_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r4_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g7_r4_sweep complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
