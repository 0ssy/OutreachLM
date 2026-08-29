from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Callable

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
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _top_tokens(probabilities: np.ndarray, tokenizer: StupidTokenizer, top_n: int = 8) -> list[dict[str, object]]:
    id_to_token = tokenizer.id_to_token
    indices = np.argsort(probabilities)[-top_n:][::-1]
    return [
        {
            "token_id": int(index),
            "token": id_to_token[int(index)],
            "probability": float(probabilities[int(index)]),
        }
        for index in indices
    ]


def _load_fixture() -> tuple[dict[str, object], StupidTokenizer, list[list[int]], list[list[int]]]:
    fixture_path = PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tokenizer = StupidTokenizer(token_to_id=fixture["token_to_id"])
    train_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["train_lines"]]
    eval_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["eval_lines"]]
    return fixture, tokenizer, train_sequences, eval_sequences


def _build_stage_arrays(
    tokenizer: StupidTokenizer,
    train_sequences: list[list[int]],
    eval_sequences: list[list[int]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
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

    prior_tail = g2_model.global_counts + 0.1
    prior_tail = prior_tail / prior_tail.sum()

    samples: list[dict[str, object]] = []
    id_to_token = tokenizer.id_to_token
    for sequence_id, sequence in enumerate(eval_sequences):
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
                    g3 = (0.7 * g2) + smooth
                    g3 = g3 / g3.sum()

            g4 = g4_model.distribution(context)
            g5 = g5_model.distribution(context)
            g6 = 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)
            g6 = g6 / g6.sum()

            samples.append(
                {
                    "sequence_id": sequence_id,
                    "position": pos,
                    "context_token_ids": [int(x) for x in context],
                    "context_tokens": [id_to_token[int(x)] for x in context],
                    "target_token_id": target,
                    "target_token": id_to_token[target],
                    "g2": g2,
                    "g3": g3,
                    "g4": g4,
                    "g5": g5,
                    "g6": g6,
                    "prior": prior_tail.copy(),
                }
            )

    storage = {
        "g2_model_storage_bytes": g2_model.model_storage_bytes,
        "g3_embedding_storage_bytes": int(embeddings.nbytes),
        "g4_model_storage_bytes": g4_model.model_storage_bytes,
        "g5_model_storage_bytes": g5_model.model_storage_bytes,
        "g6_local_storage_bytes": local.model_storage_bytes,
        "g6_medium_storage_bytes": medium.model_storage_bytes,
        "g6_global_storage_bytes": global_model.model_storage_bytes,
    }
    return samples, storage


def _variant_r1_topk_renorm(g6: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(g6, -k)[-k:]
    out = np.zeros_like(g6)
    out[idx] = g6[idx]
    return out / out.sum()


def _variant_r2_residual_bucket(g6: np.ndarray, prior: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(g6, -k)[-k:]
    out = np.zeros_like(g6)
    out[idx] = g6[idx]
    kept_mass = float(out.sum())
    residual = max(0.0, 1.0 - kept_mass)
    mask = np.ones_like(g6, dtype=bool)
    mask[idx] = False
    tail_prior = prior[mask]
    if tail_prior.sum() > 0.0:
        out[mask] = residual * (tail_prior / tail_prior.sum())
    return out / out.sum()


def _variant_r3_adaptive_k(g6: np.ndarray, mass_threshold: float, min_k: int, max_k: int) -> tuple[np.ndarray, int]:
    indices = np.argsort(g6)[::-1]
    cumulative = np.cumsum(g6[indices])
    k = int(np.searchsorted(cumulative, mass_threshold, side="left") + 1)
    k = max(min_k, min(max_k, k))
    selected = indices[:k]
    out = np.zeros_like(g6)
    out[selected] = g6[selected]
    out = out / out.sum()
    return out, k


def _variant_r3_adaptive_k_with_meta(g6: np.ndarray, mass_threshold: float, min_k: int, max_k: int) -> tuple[np.ndarray, dict[str, float]]:
    out, k = _variant_r3_adaptive_k(g6, mass_threshold=mass_threshold, min_k=min_k, max_k=max_k)
    return out, {"effective_k": float(k)}


def _variant_r4_sparse_compute_dense_probability(g6: np.ndarray, g2: np.ndarray, k: int, floor_mix: float) -> np.ndarray:
    idx = np.argpartition(g6, -k)[-k:]
    sparse = np.zeros_like(g6)
    sparse[idx] = g6[idx]
    sparse = sparse / sparse.sum()
    out = (1.0 - floor_mix) * sparse + floor_mix * g2
    return out / out.sum()


def _evaluate_variant(
    name: str,
    samples: list[dict[str, object]],
    tokenizer: StupidTokenizer,
    support_threshold: float,
    variant_fn: Callable[[dict[str, object]], tuple[np.ndarray, dict[str, float]]],
    operations_dense: int,
    operations_sparse: float,
) -> dict[str, object]:
    rows: list[np.ndarray] = []
    g6_rows: list[np.ndarray] = []
    targets: list[int] = []
    entropies: list[float] = []
    supports: list[float] = []
    mass_errors: list[float] = []
    kl_g6_variant: list[float] = []
    extra_values: dict[str, list[float]] = {}
    per_sample: list[dict[str, object]] = []

    for sample_index, sample in enumerate(samples):
        prediction, extra = variant_fn(sample)
        g6 = sample["g6"]
        target = int(sample["target_token_id"])
        rows.append(prediction)
        g6_rows.append(g6)
        targets.append(target)
        entropies.append(_entropy(prediction))
        supports.append(float(_support(prediction, support_threshold)))
        mass_errors.append(abs(float(prediction.sum()) - 1.0))
        kl_g6_variant.append(_kl_divergence(g6, prediction))
        for key, value in extra.items():
            extra_values.setdefault(key, []).append(float(value))

        per_sample.append(
            {
                "sample_index": sample_index,
                "sequence_id": int(sample["sequence_id"]),
                "position": int(sample["position"]),
                "context_tokens": sample["context_tokens"],
                "target_token": sample["target_token"],
                "g6_entropy": _entropy(g6),
                f"{name}_entropy": _entropy(prediction),
                "g6_support": _support(g6, support_threshold),
                f"{name}_support": _support(prediction, support_threshold),
                "kl_g6_variant": _kl_divergence(g6, prediction),
                "g6_top_tokens": _top_tokens(g6, tokenizer),
                f"{name}_top_tokens": _top_tokens(prediction, tokenizer),
                "probability_mass_sum": float(prediction.sum()),
            }
        )

    variant_metrics = evaluate_predictions(rows, targets)
    g6_metrics = evaluate_predictions(g6_rows, targets)

    result: dict[str, object] = {
        "variant": name,
        "metrics": {
            "g6_reference": {k: g6_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
            "variant": {k: variant_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
            "delta_cross_entropy_vs_g6": float(variant_metrics["cross_entropy"] - g6_metrics["cross_entropy"]),
        },
        "distribution_diagnostics": {
            "entropy": _summary(entropies),
            "support": _summary(supports),
            "top1_probability": _summary([float(np.max(row)) for row in rows]),
            "kl_g6_to_variant": _summary(kl_g6_variant),
            "probability_mass_error": _summary(mass_errors),
        },
        "operations_per_token": {
            "dense_reference": int(operations_dense),
            "sparse_variant_estimate": float(operations_sparse),
            "reduction_ratio": float(operations_sparse / max(operations_dense, 1)),
        },
        "gate": {
            "non_catastrophic_ce_increase_threshold": 1.0,
            "passes_non_catastrophic_gate": bool((variant_metrics["cross_entropy"] - g6_metrics["cross_entropy"]) <= 1.0),
            "strict_no_increase": bool(variant_metrics["cross_entropy"] <= g6_metrics["cross_entropy"]),
            "mass_conserved": bool(max(mass_errors) <= 1e-9),
        },
        "extra": {key: _summary(values) for key, values in extra_values.items()},
        "per_sample": per_sample,
    }
    return result


def _scaling_worker(payload: tuple[list[dict[str, object]], str, int, float]) -> int:
    samples, variant_name, top_k, floor_mix = payload
    count = 0
    for sample in samples:
        g6 = sample["g6"]
        prior = sample["prior"]
        g2 = sample["g2"]
        if variant_name == "g7_r1_renormalized_topk":
            _ = _variant_r1_topk_renorm(g6, top_k)
        elif variant_name == "g7_r2_residual_mass_sparse":
            _ = _variant_r2_residual_bucket(g6, prior, top_k)
        elif variant_name == "g7_r3_adaptive_k":
            _ = _variant_r3_adaptive_k(g6, mass_threshold=0.9, min_k=2, max_k=len(g6))[0]
        elif variant_name == "g7_r4_sparse_compute_dense_probability":
            _ = _variant_r4_sparse_compute_dense_probability(g6, g2, top_k, floor_mix=floor_mix)
        else:
            raise ValueError(f"Unknown variant: {variant_name}")
        count += 1
    return count


def _measure_scaling(
    samples: list[dict[str, object]],
    variant_name: str,
    top_k: int,
    floor_mix: float,
) -> dict[str, float]:
    expanded = samples * 2000
    single_start = time.perf_counter()
    single_count = _scaling_worker((expanded, variant_name, top_k, floor_mix))
    single_elapsed = time.perf_counter() - single_start

    mid = len(expanded) // 2
    left = expanded[:mid]
    right = expanded[mid:]
    two_start = time.perf_counter()
    with mp.get_context("spawn").Pool(processes=2) as pool:
        out = pool.map(_scaling_worker, [(left, variant_name, top_k, floor_mix), (right, variant_name, top_k, floor_mix)])
    two_elapsed = time.perf_counter() - two_start
    two_count = int(out[0] + out[1])

    single_tps = single_count / max(single_elapsed, 1e-12)
    two_tps = two_count / max(two_elapsed, 1e-12)
    return {
        "inference_items": int(len(expanded)),
        "single_cpu_elapsed_seconds": float(single_elapsed),
        "single_cpu_throughput": float(single_tps),
        "two_cpu_elapsed_seconds": float(two_elapsed),
        "two_cpu_throughput": float(two_tps),
        "latency_ratio_two_over_one": float(two_elapsed / max(single_elapsed, 1e-12)),
        "scaling_efficiency": float(two_tps / max(2.0 * single_tps, 1e-12)),
        "useful_parallelism": bool(two_tps > single_tps),
    }


def main() -> None:
    fixture, tokenizer, train_sequences, eval_sequences = _load_fixture()
    support_threshold = float(fixture["support_threshold"])
    top_k = 4
    floor_mix = 0.2

    tracemalloc.start()
    prep_start = time.perf_counter()
    samples, storage = _build_stage_arrays(tokenizer, train_sequences, eval_sequences)
    prep_time = time.perf_counter() - prep_start

    eval_start = time.perf_counter()
    r1 = _evaluate_variant(
        name="g7_r1_renormalized_topk",
        samples=samples,
        tokenizer=tokenizer,
        support_threshold=support_threshold,
        variant_fn=lambda sample: (_variant_r1_topk_renorm(sample["g6"], top_k), {"effective_k": float(top_k)}),
        operations_dense=len(tokenizer.token_to_id),
        operations_sparse=float(top_k),
    )
    r2 = _evaluate_variant(
        name="g7_r2_residual_mass_sparse",
        samples=samples,
        tokenizer=tokenizer,
        support_threshold=support_threshold,
        variant_fn=lambda sample: (_variant_r2_residual_bucket(sample["g6"], sample["prior"], top_k), {"effective_k": float(top_k)}),
        operations_dense=len(tokenizer.token_to_id),
        operations_sparse=float(top_k + 1),
    )
    r3 = _evaluate_variant(
        name="g7_r3_adaptive_k",
        samples=samples,
        tokenizer=tokenizer,
        support_threshold=support_threshold,
        variant_fn=lambda sample: _variant_r3_adaptive_k_with_meta(sample["g6"], mass_threshold=0.9, min_k=2, max_k=len(tokenizer.token_to_id)),
        operations_dense=len(tokenizer.token_to_id),
        operations_sparse=-1.0,  # replaced below with measured effective_k.
    )
    if "effective_k" in r3["extra"]:
        r3["operations_per_token"]["sparse_variant_estimate"] = r3["extra"]["effective_k"]["mean"]
        r3["operations_per_token"]["reduction_ratio"] = r3["extra"]["effective_k"]["mean"] / max(len(tokenizer.token_to_id), 1)

    r4 = _evaluate_variant(
        name="g7_r4_sparse_compute_dense_probability",
        samples=samples,
        tokenizer=tokenizer,
        support_threshold=support_threshold,
        variant_fn=lambda sample: (_variant_r4_sparse_compute_dense_probability(sample["g6"], sample["g2"], k=top_k, floor_mix=floor_mix), {"effective_k": float(top_k), "floor_mix": float(floor_mix)}),
        operations_dense=len(tokenizer.token_to_id),
        operations_sparse=float(top_k + 4),
    )
    eval_time = time.perf_counter() - eval_start

    scaling = {
        "g7_r1_renormalized_topk": _measure_scaling(samples, "g7_r1_renormalized_topk", top_k=top_k, floor_mix=floor_mix),
        "g7_r2_residual_mass_sparse": _measure_scaling(samples, "g7_r2_residual_mass_sparse", top_k=top_k, floor_mix=floor_mix),
        "g7_r3_adaptive_k": _measure_scaling(samples, "g7_r3_adaptive_k", top_k=top_k, floor_mix=floor_mix),
        "g7_r4_sparse_compute_dense_probability": _measure_scaling(samples, "g7_r4_sparse_compute_dense_probability", top_k=top_k, floor_mix=floor_mix),
    }

    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    variants = [r1, r2, r3, r4]
    best = min(variants, key=lambda item: float(item["metrics"]["variant"]["cross_entropy"]))
    r5 = {
        "variant": "g7_r5_best_recovered_sparse_mechanism",
        "selected_variant": best["variant"],
        "selection_criterion": "minimum cross entropy on frozen fixture",
        "selected_metrics": best["metrics"],
        "selected_distribution_diagnostics": best["distribution_diagnostics"],
        "selected_operations_per_token": best["operations_per_token"],
        "selected_gate": best["gate"],
        "selected_scaling": scaling[best["variant"]],
    }

    summary_rows = []
    for item in variants:
        name = item["variant"]
        summary_rows.append(
            {
                "variant": name,
                "accuracy": item["metrics"]["variant"]["accuracy"],
                "cross_entropy": item["metrics"]["variant"]["cross_entropy"],
                "perplexity": item["metrics"]["variant"]["perplexity"],
                "delta_cross_entropy_vs_g6": item["metrics"]["delta_cross_entropy_vs_g6"],
                "entropy_mean": item["distribution_diagnostics"]["entropy"]["mean"],
                "support_mean": item["distribution_diagnostics"]["support"]["mean"],
                "kl_g6_to_variant_mean": item["distribution_diagnostics"]["kl_g6_to_variant"]["mean"],
                "mass_error_max": item["distribution_diagnostics"]["probability_mass_error"]["max"],
                "ops_dense": item["operations_per_token"]["dense_reference"],
                "ops_sparse": item["operations_per_token"]["sparse_variant_estimate"],
                "ops_reduction_ratio": item["operations_per_token"]["reduction_ratio"],
                "passes_non_catastrophic_gate": item["gate"]["passes_non_catastrophic_gate"],
                "strict_no_increase": item["gate"]["strict_no_increase"],
                "two_cpu_throughput": scaling[name]["two_cpu_throughput"],
                "single_cpu_throughput": scaling[name]["single_cpu_throughput"],
                "scaling_efficiency": scaling[name]["scaling_efficiency"],
            }
        )

    result = {
        "experiment_id": "g7_recovery_sparse_mechanisms",
        "seed": int(fixture["seed"]),
        "frozen_fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "config": {
            "support_threshold": support_threshold,
            "r1_top_k": top_k,
            "r2_top_k": top_k,
            "r3_mass_threshold": 0.9,
            "r3_min_k": 2,
            "r3_max_k": len(tokenizer.token_to_id),
            "r4_top_k": top_k,
            "r4_floor_mix": floor_mix,
        },
        "resource_usage": {
            "prep_time_seconds": prep_time,
            "evaluation_time_seconds": eval_time,
            "peak_process_ram_bytes": int(peak_ram),
            "model_storage_bytes": storage,
            "eval_token_count": len(samples),
        },
        "variants": {
            "g7_r1_renormalized_topk": r1,
            "g7_r2_residual_mass_sparse": r2,
            "g7_r3_adaptive_k": r3,
            "g7_r4_sparse_compute_dense_probability": r4,
            "g7_r5_best_recovered_sparse_mechanism": r5,
        },
        "scaling": scaling,
        "summary_rows": summary_rows,
    }

    summary = {
        "experiment_id": "g7_recovery_sparse_mechanisms_summary",
        "seed": int(fixture["seed"]),
        "summary_rows": summary_rows,
        "selected_best_variant": r5["selected_variant"],
        "resource_usage": result["resource_usage"],
    }

    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "g7_recovery_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (results_dir / "g7_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_recovery_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_recovery_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g7_recovery complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
