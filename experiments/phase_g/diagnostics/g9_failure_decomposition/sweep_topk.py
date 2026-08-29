from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import tracemalloc

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


def _support_size(probabilities: np.ndarray, threshold: float = 1e-4) -> int:
    return int(np.count_nonzero(probabilities > threshold))


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _summary_stats(values: list[float]) -> dict[str, float]:
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


def _stage_stats(rows: list[np.ndarray], targets: list[int], threshold: float) -> dict[str, float]:
    metrics = evaluate_predictions(rows, targets)
    entropies = [_entropy(row) for row in rows]
    supports = [_support_size(row, threshold) for row in rows]
    top1 = [float(np.max(row)) for row in rows]
    return {
        "accuracy": metrics["accuracy"],
        "cross_entropy": metrics["cross_entropy"],
        "perplexity": metrics["perplexity"],
        "count": int(metrics["count"]),
        "entropy": _summary_stats(entropies),
        "support": _summary_stats([float(x) for x in supports]),
        "top1_probability": _summary_stats(top1),
    }


def _top_tokens(probabilities: np.ndarray, tokenizer: StupidTokenizer, top_n: int = 8) -> list[dict[str, object]]:
    id_to_token = tokenizer.id_to_token
    idx = np.argsort(probabilities)[-top_n:][::-1]
    return [
        {
            "token_id": int(i),
            "token": id_to_token[int(i)],
            "probability": float(probabilities[int(i)]),
        }
        for i in idx
    ]


def _load_fixture() -> tuple[dict[str, object], StupidTokenizer, list[list[int]], list[list[int]]]:
    fixture_path = PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tokenizer = StupidTokenizer(token_to_id=fixture["token_to_id"])

    train_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["train_lines"]]
    eval_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["eval_lines"]]
    return fixture, tokenizer, train_sequences, eval_sequences


def main() -> None:
    fixture, tokenizer, train_sequences, eval_sequences = _load_fixture()
    vocab_size = len(tokenizer.token_to_id)
    sweep_topk = [2, 4, 8, 16, vocab_size]
    support_threshold = float(fixture["support_threshold"])

    tracemalloc.start()
    train_start = time.perf_counter()

    g2_model = SparseNGramModel(vocab_size=vocab_size, order=2, alpha=0.1)
    g2_model.fit(train_sequences)
    embeddings = _cooccurrence_embeddings(train_sequences, vocab_size=vocab_size, window=2)

    g4_model = CompressedContextModel(vocab_size=vocab_size, context_length=int(fixture["context_length"]), bucket_count=32, alpha=0.1)
    g4_model.fit(train_sequences)

    g5_model = AdaptiveMemoryModel(vocab_size=vocab_size, memory_size=3, window_size=6, alpha=0.1)
    g5_model.fit(train_sequences)

    local = SparseNGramModel(vocab_size=vocab_size, order=2, alpha=0.1)
    medium = SparseNGramModel(vocab_size=vocab_size, order=4, alpha=0.1)
    global_model = SparseNGramModel(vocab_size=vocab_size, order=1, alpha=0.1)
    local.fit(train_sequences)
    medium.fit(train_sequences)
    global_model.fit(train_sequences)

    training_time = time.perf_counter() - train_start

    def p_g2(context: list[int]) -> np.ndarray:
        return g2_model.distribution(context)

    def p_g3(context: list[int], base: np.ndarray) -> np.ndarray:
        if len(context) < 2:
            return base.copy()
        vector = embeddings[int(context[-2])] + embeddings[int(context[-1])]
        sims = embeddings @ vector
        sims = np.maximum(sims, 0.0)
        if sims.sum() <= 0.0:
            return base.copy()
        smooth = (sims / sims.sum()) * 0.3
        out = (0.7 * base) + smooth
        return out / out.sum()

    def p_g6(context: list[int]) -> np.ndarray:
        out = 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)
        return out / out.sum()

    start_infer = time.perf_counter()
    all_results: list[dict[str, object]] = []
    for top_k in sweep_topk:
        g2_rows: list[np.ndarray] = []
        g3_rows: list[np.ndarray] = []
        g4_rows: list[np.ndarray] = []
        g5_rows: list[np.ndarray] = []
        g6_rows: list[np.ndarray] = []
        g7_rows: list[np.ndarray] = []
        g9_rows: list[np.ndarray] = []
        targets: list[int] = []
        kl_g6_g7: list[float] = []
        kl_g7_g9: list[float] = []
        per_sample: list[dict[str, object]] = []

        sample_index = 0
        for sequence_id, sequence in enumerate(eval_sequences):
            for pos in range(len(sequence) - 1):
                context = sequence[: pos + 1]
                target = int(sequence[pos + 1])

                s_g2 = p_g2(context)
                s_g3 = p_g3(context, s_g2)
                s_g4 = g4_model.distribution(context)
                s_g5 = g5_model.distribution(context)
                s_g6 = p_g6(context)

                if top_k >= vocab_size:
                    s_g7 = s_g6.copy()
                else:
                    idx = np.argpartition(s_g6, -top_k)[-top_k:]
                    sparse = np.zeros_like(s_g6)
                    sparse[idx] = s_g6[idx]
                    s_g7 = sparse / sparse.sum()

                s_g9 = (0.2 * s_g2) + (0.15 * s_g3) + (0.15 * s_g4) + (0.2 * s_g5) + (0.1 * s_g6) + (0.2 * s_g7)
                s_g9 = s_g9 / s_g9.sum()

                g2_rows.append(s_g2)
                g3_rows.append(s_g3)
                g4_rows.append(s_g4)
                g5_rows.append(s_g5)
                g6_rows.append(s_g6)
                g7_rows.append(s_g7)
                g9_rows.append(s_g9)
                targets.append(target)

                k67 = _kl_divergence(s_g6, s_g7)
                k79 = _kl_divergence(s_g7, s_g9)
                kl_g6_g7.append(k67)
                kl_g7_g9.append(k79)

                per_sample.append(
                    {
                        "sample_index": sample_index,
                        "sequence_id": sequence_id,
                        "position": pos,
                        "context_token_ids": [int(x) for x in context],
                        "target_token_id": target,
                        "target_token": tokenizer.id_to_token[target],
                        "g6_entropy": _entropy(s_g6),
                        "g7_entropy": _entropy(s_g7),
                        "g9_entropy": _entropy(s_g9),
                        "g6_support": _support_size(s_g6, support_threshold),
                        "g7_support": _support_size(s_g7, support_threshold),
                        "g9_support": _support_size(s_g9, support_threshold),
                        "kl_g6_g7": k67,
                        "kl_g7_g9": k79,
                        "g6_top_tokens": _top_tokens(s_g6, tokenizer),
                        "g7_top_tokens": _top_tokens(s_g7, tokenizer),
                        "g9_top_tokens": _top_tokens(s_g9, tokenizer),
                    }
                )
                sample_index += 1

        stage_metrics = {
            "g2": _stage_stats(g2_rows, targets, support_threshold),
            "g3": _stage_stats(g3_rows, targets, support_threshold),
            "g4": _stage_stats(g4_rows, targets, support_threshold),
            "g5": _stage_stats(g5_rows, targets, support_threshold),
            "g6": _stage_stats(g6_rows, targets, support_threshold),
            "g7": _stage_stats(g7_rows, targets, support_threshold),
            "g9": _stage_stats(g9_rows, targets, support_threshold),
        }

        top_k_result = {
            "top_k": int(top_k),
            "sparsity_ratio": float(1.0 - (min(top_k, vocab_size) / vocab_size)),
            "eval_token_count": len(targets),
            "stage_metrics": stage_metrics,
            "deltas": {
                "cross_entropy_g6_to_g7": stage_metrics["g7"]["cross_entropy"] - stage_metrics["g6"]["cross_entropy"],
                "cross_entropy_g7_to_g9": stage_metrics["g9"]["cross_entropy"] - stage_metrics["g7"]["cross_entropy"],
                "entropy_mean_g6_to_g7": stage_metrics["g7"]["entropy"]["mean"] - stage_metrics["g6"]["entropy"]["mean"],
                "support_mean_g6_to_g7": stage_metrics["g7"]["support"]["mean"] - stage_metrics["g6"]["support"]["mean"],
            },
            "kl": {
                "g6_g7": _summary_stats(kl_g6_g7),
                "g7_g9": _summary_stats(kl_g7_g9),
            },
            "per_sample": per_sample,
        }
        all_results.append(top_k_result)

    inference_time = time.perf_counter() - start_infer
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary_rows = []
    for item in all_results:
        summary_rows.append(
            {
                "top_k": item["top_k"],
                "sparsity_ratio": item["sparsity_ratio"],
                "g6_cross_entropy": item["stage_metrics"]["g6"]["cross_entropy"],
                "g7_cross_entropy": item["stage_metrics"]["g7"]["cross_entropy"],
                "g9_cross_entropy": item["stage_metrics"]["g9"]["cross_entropy"],
                "g6_entropy_mean": item["stage_metrics"]["g6"]["entropy"]["mean"],
                "g7_entropy_mean": item["stage_metrics"]["g7"]["entropy"]["mean"],
                "g9_entropy_mean": item["stage_metrics"]["g9"]["entropy"]["mean"],
                "g6_support_mean": item["stage_metrics"]["g6"]["support"]["mean"],
                "g7_support_mean": item["stage_metrics"]["g7"]["support"]["mean"],
                "g9_support_mean": item["stage_metrics"]["g9"]["support"]["mean"],
                "delta_ce_g6_to_g7": item["deltas"]["cross_entropy_g6_to_g7"],
                "delta_ce_g7_to_g9": item["deltas"]["cross_entropy_g7_to_g9"],
                "kl_g6_g7_mean": item["kl"]["g6_g7"]["mean"],
                "kl_g6_g7_p95": item["kl"]["g6_g7"]["p95"],
                "kl_g7_g9_mean": item["kl"]["g7_g9"]["mean"],
            }
        )

    result = {
        "experiment_id": "g9_failure_decomposition_topk_sweep",
        "seed": int(fixture["seed"]),
        "fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "vocab_size": vocab_size,
        "support_threshold": support_threshold,
        "sweep_topk": sweep_topk,
        "resource_usage": {
            "training_time_seconds": training_time,
            "inference_time_seconds": inference_time,
            "peak_process_ram_bytes": int(peak),
            "tokens_per_second_sweep": sum(row["eval_token_count"] for row in all_results) / max(inference_time, 1e-12),
        },
        "summary_rows": summary_rows,
        "details": all_results,
    }

    summary = {
        "experiment_id": "g9_failure_decomposition_topk_sweep_summary",
        "seed": int(fixture["seed"]),
        "sweep_topk": sweep_topk,
        "resource_usage": result["resource_usage"],
        "summary_rows": summary_rows,
    }

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "g9_failure_topk_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (results_dir / "g9_failure_topk_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (PHASE_G_ROOT / "results" / "g9_failure_topk_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g9_failure_topk_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g9_failure_topk_sweep complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
