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

from common.datasets import BASE_CORPUS, CONTEXT_AMBIGUITY_CORPUS, LONG_CONTEXT_CORPUS, build_train_eval_split  # noqa: E402
from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.phase_components import AdaptiveMemoryModel, CompressedContextModel  # noqa: E402
from common.tokenizer import StupidTokenizer, build_stupid_tokenizer_from_lines  # noqa: E402


def _encode_lines(tokenizer: StupidTokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


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


def _stage_summary(rows: list[np.ndarray], targets: list[int], threshold: float) -> dict[str, float]:
    metrics = evaluate_predictions(rows, targets)
    entropies = [_entropy(row) for row in rows]
    supports = [_support_size(row, threshold) for row in rows]
    top1 = [float(np.max(row)) for row in rows]
    l2_norms = [float(np.linalg.norm(row)) for row in rows]
    l1_norms = [float(np.linalg.norm(row, ord=1)) for row in rows]
    return {
        "accuracy": metrics["accuracy"],
        "cross_entropy": metrics["cross_entropy"],
        "perplexity": metrics["perplexity"],
        "count": int(metrics["count"]),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "entropy_min": float(np.min(entropies)),
        "entropy_max": float(np.max(entropies)),
        "support_mean": float(np.mean(supports)),
        "support_min": float(np.min(supports)),
        "support_max": float(np.max(supports)),
        "top1_probability_mean": float(np.mean(top1)),
        "top1_probability_min": float(np.min(top1)),
        "top1_probability_max": float(np.max(top1)),
        "state_l2_mean": float(np.mean(l2_norms)),
        "state_l2_std": float(np.std(l2_norms)),
        "state_l1_mean": float(np.mean(l1_norms)),
    }


def _distribution_record(
    probabilities: np.ndarray,
    target: int,
    tokenizer: StupidTokenizer,
    top_n: int = 8,
) -> dict[str, object]:
    id_to_token = tokenizer.id_to_token
    top_indices = np.argsort(probabilities)[-top_n:][::-1]
    return {
        "target_id": int(target),
        "target_token": id_to_token[int(target)],
        "top_prediction_id": int(np.argmax(probabilities)),
        "top_prediction_token": id_to_token[int(np.argmax(probabilities))],
        "top_prediction_probability": float(np.max(probabilities)),
        "entropy": _entropy(probabilities),
        "support_size@1e-4": _support_size(probabilities, threshold=1e-4),
        "distribution": [float(x) for x in probabilities.tolist()],
        "top_tokens": [
            {
                "token_id": int(index),
                "token": id_to_token[int(index)],
                "probability": float(probabilities[int(index)]),
            }
            for index in top_indices
        ],
    }


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    seed = 1337
    eval_ratio = 0.3
    context_length = 4
    sparsity_top_k = 4
    support_threshold = 1e-4

    corpus = BASE_CORPUS + CONTEXT_AMBIGUITY_CORPUS + LONG_CONTEXT_CORPUS
    train_lines, eval_lines = build_train_eval_split(corpus, seed=seed, eval_ratio=eval_ratio)
    tokenizer = build_stupid_tokenizer_from_lines(corpus)
    train_sequences = _encode_lines(tokenizer, train_lines)
    eval_sequences = _encode_lines(tokenizer, eval_lines)

    root = Path(__file__).resolve().parent
    frozen_eval_dir = root.parent / "frozen_eval"
    results_dir = root / "results"
    frozen_eval_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    fixture = {
        "seed": seed,
        "eval_ratio": eval_ratio,
        "context_length": context_length,
        "sparsity_top_k": sparsity_top_k,
        "support_threshold": support_threshold,
        "corpus_size": len(corpus),
        "train_size": len(train_lines),
        "eval_size": len(eval_lines),
        "vocab_size": len(tokenizer.token_to_id),
        "corpus": corpus,
        "train_lines": train_lines,
        "eval_lines": eval_lines,
        "token_to_id": tokenizer.token_to_id,
    }
    (frozen_eval_dir / "fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    tokenizer.save(frozen_eval_dir / "tokenizer.json")

    tracemalloc.start()
    train_start = time.perf_counter()

    g2_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
    g2_model.fit(train_sequences)

    embeddings = _cooccurrence_embeddings(train_sequences, len(tokenizer.token_to_id), window=2)

    g4_model = CompressedContextModel(
        vocab_size=len(tokenizer.token_to_id),
        context_length=context_length,
        bucket_count=32,
        alpha=0.1,
    )
    g4_model.fit(train_sequences)

    g5_model = AdaptiveMemoryModel(
        vocab_size=len(tokenizer.token_to_id),
        memory_size=3,
        window_size=6,
        alpha=0.1,
    )
    g5_model.fit(train_sequences)

    local = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
    medium = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=4, alpha=0.1)
    global_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=1, alpha=0.1)
    local.fit(train_sequences)
    medium.fit(train_sequences)
    global_model.fit(train_sequences)

    training_time = time.perf_counter() - train_start

    inference_start = time.perf_counter()

    stage_rows = {
        "g2": [],
        "g3": [],
        "g4": [],
        "g5": [],
        "g6": [],
        "g7": [],
        "g9": [],
    }
    targets: list[int] = []
    sample_traces: list[dict[str, object]] = []
    kl_chain = {
        "g2_g3": [],
        "g3_g4": [],
        "g4_g5": [],
        "g5_g6": [],
        "g6_g7": [],
        "g7_g9": [],
    }

    id_to_token = tokenizer.id_to_token
    sample_index = 0
    for sequence_id, sequence in enumerate(eval_sequences):
        for pos in range(len(sequence) - 1):
            target = int(sequence[pos + 1])
            context = sequence[: pos + 1]
            history_tokens = [id_to_token[token_id] for token_id in context]

            p_g2 = g2_model.distribution(context)

            if len(context) < 2:
                p_g3 = p_g2.copy()
            else:
                vector = embeddings[int(context[-2])] + embeddings[int(context[-1])]
                sims = embeddings @ vector
                sims = np.maximum(sims, 0.0)
                if sims.sum() <= 0.0:
                    p_g3 = p_g2.copy()
                else:
                    smooth = (sims / sims.sum()) * 0.3
                    p_g3 = 0.7 * p_g2 + smooth
                    p_g3 = p_g3 / p_g3.sum()

            p_g4 = g4_model.distribution(context)
            p_g5 = g5_model.distribution(context)
            p_g6 = 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)
            p_g6 = p_g6 / p_g6.sum()

            g7_indices = np.argpartition(p_g6, -sparsity_top_k)[-sparsity_top_k:]
            p_g7 = np.zeros_like(p_g6)
            p_g7[g7_indices] = p_g6[g7_indices]
            p_g7 = p_g7 / p_g7.sum()

            p_g9 = (0.2 * p_g2) + (0.15 * p_g3) + (0.15 * p_g4) + (0.2 * p_g5) + (0.1 * p_g6) + (0.2 * p_g7)
            p_g9 = p_g9 / p_g9.sum()

            stage_rows["g2"].append(p_g2)
            stage_rows["g3"].append(p_g3)
            stage_rows["g4"].append(p_g4)
            stage_rows["g5"].append(p_g5)
            stage_rows["g6"].append(p_g6)
            stage_rows["g7"].append(p_g7)
            stage_rows["g9"].append(p_g9)
            targets.append(target)

            kl_chain["g2_g3"].append(_kl_divergence(p_g2, p_g3))
            kl_chain["g3_g4"].append(_kl_divergence(p_g3, p_g4))
            kl_chain["g4_g5"].append(_kl_divergence(p_g4, p_g5))
            kl_chain["g5_g6"].append(_kl_divergence(p_g5, p_g6))
            kl_chain["g6_g7"].append(_kl_divergence(p_g6, p_g7))
            kl_chain["g7_g9"].append(_kl_divergence(p_g7, p_g9))

            sample_traces.append(
                {
                    "sample_index": sample_index,
                    "sequence_id": sequence_id,
                    "position": pos,
                    "context_token_ids": [int(x) for x in context],
                    "context_tokens": history_tokens,
                    "target_token_id": target,
                    "target_token": id_to_token[target],
                    "stages": {
                        "g2": _distribution_record(p_g2, target, tokenizer),
                        "g3": _distribution_record(p_g3, target, tokenizer),
                        "g4": _distribution_record(p_g4, target, tokenizer),
                        "g5": _distribution_record(p_g5, target, tokenizer),
                        "g6": _distribution_record(p_g6, target, tokenizer),
                        "g7": _distribution_record(p_g7, target, tokenizer),
                        "g9": _distribution_record(p_g9, target, tokenizer),
                    },
                    "kl_chain": {
                        "g2_g3": kl_chain["g2_g3"][-1],
                        "g3_g4": kl_chain["g3_g4"][-1],
                        "g4_g5": kl_chain["g4_g5"][-1],
                        "g5_g6": kl_chain["g5_g6"][-1],
                        "g6_g7": kl_chain["g6_g7"][-1],
                        "g7_g9": kl_chain["g7_g9"][-1],
                    },
                }
            )
            sample_index += 1

    inference_time = time.perf_counter() - inference_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stage_metrics = {
        stage_name: _stage_summary(rows, targets, threshold=support_threshold)
        for stage_name, rows in stage_rows.items()
    }
    kl_summary = {name: _quantiles(values) for name, values in kl_chain.items()}

    largest_mean_kl_edge = max(kl_summary.items(), key=lambda item: item[1]["mean"])[0]
    largest_p95_kl_edge = max(kl_summary.items(), key=lambda item: item[1]["p95"])[0]
    stage_deltas = {
        "cross_entropy": {
            "g2_to_g3": stage_metrics["g3"]["cross_entropy"] - stage_metrics["g2"]["cross_entropy"],
            "g3_to_g4": stage_metrics["g4"]["cross_entropy"] - stage_metrics["g3"]["cross_entropy"],
            "g4_to_g5": stage_metrics["g5"]["cross_entropy"] - stage_metrics["g4"]["cross_entropy"],
            "g5_to_g6": stage_metrics["g6"]["cross_entropy"] - stage_metrics["g5"]["cross_entropy"],
            "g6_to_g7": stage_metrics["g7"]["cross_entropy"] - stage_metrics["g6"]["cross_entropy"],
            "g7_to_g9": stage_metrics["g9"]["cross_entropy"] - stage_metrics["g7"]["cross_entropy"],
        },
        "entropy_mean": {
            "g2_to_g3": stage_metrics["g3"]["entropy_mean"] - stage_metrics["g2"]["entropy_mean"],
            "g3_to_g4": stage_metrics["g4"]["entropy_mean"] - stage_metrics["g3"]["entropy_mean"],
            "g4_to_g5": stage_metrics["g5"]["entropy_mean"] - stage_metrics["g4"]["entropy_mean"],
            "g5_to_g6": stage_metrics["g6"]["entropy_mean"] - stage_metrics["g5"]["entropy_mean"],
            "g6_to_g7": stage_metrics["g7"]["entropy_mean"] - stage_metrics["g6"]["entropy_mean"],
            "g7_to_g9": stage_metrics["g9"]["entropy_mean"] - stage_metrics["g7"]["entropy_mean"],
        },
        "support_mean": {
            "g2_to_g3": stage_metrics["g3"]["support_mean"] - stage_metrics["g2"]["support_mean"],
            "g3_to_g4": stage_metrics["g4"]["support_mean"] - stage_metrics["g3"]["support_mean"],
            "g4_to_g5": stage_metrics["g5"]["support_mean"] - stage_metrics["g4"]["support_mean"],
            "g5_to_g6": stage_metrics["g6"]["support_mean"] - stage_metrics["g5"]["support_mean"],
            "g6_to_g7": stage_metrics["g7"]["support_mean"] - stage_metrics["g6"]["support_mean"],
            "g7_to_g9": stage_metrics["g9"]["support_mean"] - stage_metrics["g7"]["support_mean"],
        },
    }

    result = {
        "experiment_id": "g9_failure_decomposition",
        "seed": seed,
        "frozen_fixture": {
            "path": str((frozen_eval_dir / "fixture.json").resolve()),
            "corpus_size": len(corpus),
            "train_size": len(train_lines),
            "eval_size": len(eval_lines),
            "eval_token_count": len(targets),
            "vocab_size": len(tokenizer.token_to_id),
            "context_length": context_length,
            "sparsity_top_k": sparsity_top_k,
            "support_threshold": support_threshold,
        },
        "resource_usage": {
            "training_time_seconds": training_time,
            "inference_time_seconds": inference_time,
            "tokens_per_second": len(targets) / max(inference_time, 1e-12),
            "peak_process_ram_bytes": int(peak),
            "model_storage_bytes": {
                "g2": g2_model.model_storage_bytes,
                "g3_embeddings": int(embeddings.nbytes),
                "g4": g4_model.model_storage_bytes,
                "g5": g5_model.model_storage_bytes,
                "g6_local": local.model_storage_bytes,
                "g6_medium": medium.model_storage_bytes,
                "g6_global": global_model.model_storage_bytes,
            },
        },
        "stage_metrics": stage_metrics,
        "stage_deltas": stage_deltas,
        "kl_divergence": {
            "summary": kl_summary,
            "largest_mean_kl_edge": largest_mean_kl_edge,
            "largest_p95_kl_edge": largest_p95_kl_edge,
        },
        "sample_traces": sample_traces,
    }

    summary = {
        "experiment_id": "g9_failure_decomposition_summary",
        "seed": seed,
        "eval_token_count": len(targets),
        "stage_metrics": stage_metrics,
        "stage_deltas": stage_deltas,
        "kl_divergence": result["kl_divergence"],
        "resource_usage": result["resource_usage"],
    }

    (results_dir / "g9_failure_decomposition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (results_dir / "g9_failure_decomposition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g9_failure_decomposition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g9_failure_decomposition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g9_failure_decomposition complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
