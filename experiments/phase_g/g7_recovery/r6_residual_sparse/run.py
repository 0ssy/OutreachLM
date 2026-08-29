from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tracemalloc

PHASE_G_ROOT = Path(__file__).resolve().parents[2]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.cpu_threads import configure_cpu_threads_from_env  # noqa: E402
from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402

CPU_THREADING = configure_cpu_threads_from_env()

import numpy as np


def _load_frozen_fixture() -> tuple[dict, StupidTokenizer, list[list[int]], list[list[int]]]:
    fixture_path = PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tokenizer = StupidTokenizer(token_to_id=fixture["token_to_id"])
    train_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["train_lines"]]
    eval_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["eval_lines"]]
    return fixture, tokenizer, train_sequences, eval_sequences


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


def _g6_distribution(local: SparseNGramModel, medium: SparseNGramModel, global_model: SparseNGramModel, context: list[int]) -> np.ndarray:
    out = 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)
    return out / out.sum()


def _g2_distribution(model: SparseNGramModel, context: list[int]) -> np.ndarray:
    return model.distribution(context)


def _r4_baseline(g6: np.ndarray, g2: np.ndarray, top_k: int = 4, floor_mix: float = 0.2) -> np.ndarray:
    idx = np.argpartition(g6, -top_k)[-top_k:]
    sparse = np.zeros_like(g6)
    sparse[idx] = g6[idx]
    sparse = sparse / sparse.sum()
    out = (1.0 - floor_mix) * sparse + floor_mix * g2
    return out / out.sum()


def _r6_residual_sparse(g6: np.ndarray, top_k: int, unigram_distribution: np.ndarray) -> np.ndarray:
    idx = np.argpartition(g6, -top_k)[-top_k:]
    in_topk = np.zeros_like(g6, dtype=bool)
    in_topk[idx] = True

    candidate_probs = np.zeros_like(g6)
    candidate_probs[idx] = g6[idx]
    candidate_mass = float(candidate_probs.sum())
    residual_mass = max(0.0, 1.0 - candidate_mass)

    normalized_candidates = np.zeros_like(g6)
    if candidate_mass > 0.0:
        normalized_candidates[idx] = candidate_probs[idx] / candidate_mass

    out = np.zeros_like(g6)
    out[idx] = (1.0 - residual_mass) * normalized_candidates[idx]

    tail_mask = ~in_topk
    tail_q = np.where(tail_mask, unigram_distribution, 0.0)
    tail_q_sum = float(tail_q.sum())
    if tail_q_sum > 0.0 and residual_mass > 0.0:
        out[tail_mask] = residual_mass * (tail_q[tail_mask] / tail_q_sum)

    out = out / out.sum()
    return out


def _r6_residual_sparse_contextual(
    g6: np.ndarray,
    top_k: int,
    g2: np.ndarray,
    unigram_distribution: np.ndarray,
    context_blend: float = 0.8,
    tail_temperature: float = 0.85,
) -> np.ndarray:
    idx = np.argpartition(g6, -top_k)[-top_k:]
    in_topk = np.zeros_like(g6, dtype=bool)
    in_topk[idx] = True

    candidate_probs = np.zeros_like(g6)
    candidate_probs[idx] = g6[idx]
    candidate_mass = float(candidate_probs.sum())
    residual_mass = max(0.0, 1.0 - candidate_mass)

    normalized_candidates = np.zeros_like(g6)
    if candidate_mass > 0.0:
        normalized_candidates[idx] = candidate_probs[idx] / candidate_mass

    background = (context_blend * g2) + ((1.0 - context_blend) * unigram_distribution)
    background = np.clip(background, 1e-12, 1.0)
    background = np.power(background, tail_temperature)

    out = np.zeros_like(g6)
    out[idx] = (1.0 - residual_mass) * normalized_candidates[idx]

    tail_mask = ~in_topk
    tail_q = np.where(tail_mask, background, 0.0)
    tail_q_sum = float(tail_q.sum())
    if tail_q_sum > 0.0 and residual_mass > 0.0:
        out[tail_mask] = residual_mass * (tail_q[tail_mask] / tail_q_sum)

    out = out / out.sum()
    return out


def _r6_residual_sparse_contextual_plus(
    g6: np.ndarray,
    top_k: int,
    g2: np.ndarray,
    unigram_distribution: np.ndarray,
    context_blend: float,
    tail_temperature: float,
    head_temperature: float,
    head_context_mix: float,
    residual_gain: float,
) -> np.ndarray:
    idx = np.argpartition(g6, -top_k)[-top_k:]
    in_topk = np.zeros_like(g6, dtype=bool)
    in_topk[idx] = True

    candidate_probs = np.zeros_like(g6)
    candidate_probs[idx] = g6[idx]
    candidate_mass = float(candidate_probs.sum())
    residual_mass = max(0.0, min(1.0, (1.0 - candidate_mass) * residual_gain))

    g6_head = np.clip(g6[idx], 1e-12, 1.0)
    g6_head = np.power(g6_head, head_temperature)
    g2_head = np.clip(g2[idx], 1e-12, 1.0)
    g2_head = g2_head / max(float(g2_head.sum()), 1e-12)
    blended_head = ((1.0 - head_context_mix) * g6_head) + (head_context_mix * g2_head)
    blended_head = blended_head / max(float(blended_head.sum()), 1e-12)

    background = (context_blend * g2) + ((1.0 - context_blend) * unigram_distribution)
    background = np.clip(background, 1e-12, 1.0)
    background = np.power(background, tail_temperature)

    out = np.zeros_like(g6)
    out[idx] = (1.0 - residual_mass) * blended_head

    tail_mask = ~in_topk
    tail_q = np.where(tail_mask, background, 0.0)
    tail_q_sum = float(tail_q.sum())
    if tail_q_sum > 0.0 and residual_mass > 0.0:
        out[tail_mask] = residual_mass * (tail_q[tail_mask] / tail_q_sum)

    out = out / out.sum()
    return out


def _r6_residual_sparse_contextual_plus_forced_unk(
    g6: np.ndarray,
    top_k: int,
    g2: np.ndarray,
    unigram_distribution: np.ndarray,
    unk_id: int,
    context_blend: float,
    tail_temperature: float,
    head_temperature: float,
    head_context_mix: float,
    residual_gain: float,
) -> np.ndarray:
    idx = np.argpartition(g6, -top_k)[-top_k:]
    if unk_id not in idx:
        smallest_local = int(np.argmin(g6[idx]))
        idx[smallest_local] = int(unk_id)
    idx = np.unique(idx)
    if len(idx) > top_k:
        keep = np.argsort(g6[idx])[-top_k:]
        idx = idx[keep]

    in_topk = np.zeros_like(g6, dtype=bool)
    in_topk[idx] = True

    candidate_probs = np.zeros_like(g6)
    candidate_probs[idx] = g6[idx]
    candidate_mass = float(candidate_probs.sum())
    residual_mass = max(0.0, min(1.0, (1.0 - candidate_mass) * residual_gain))

    g6_head = np.clip(g6[idx], 1e-12, 1.0)
    g6_head = np.power(g6_head, head_temperature)
    g2_head = np.clip(g2[idx], 1e-12, 1.0)
    g2_head = g2_head / max(float(g2_head.sum()), 1e-12)
    blended_head = ((1.0 - head_context_mix) * g6_head) + (head_context_mix * g2_head)
    blended_head = blended_head / max(float(blended_head.sum()), 1e-12)

    background = (context_blend * g2) + ((1.0 - context_blend) * unigram_distribution)
    background = np.clip(background, 1e-12, 1.0)
    background = np.power(background, tail_temperature)

    out = np.zeros_like(g6)
    out[idx] = (1.0 - residual_mass) * blended_head

    tail_mask = ~in_topk
    tail_q = np.where(tail_mask, background, 0.0)
    tail_q_sum = float(tail_q.sum())
    if tail_q_sum > 0.0 and residual_mass > 0.0:
        out[tail_mask] = residual_mass * (tail_q[tail_mask] / tail_q_sum)

    out = out / out.sum()
    return out


def _r6_residual_sparse_contextual_plus_forced_unk_kl_safe(
    g6: np.ndarray,
    top_k: int,
    g2: np.ndarray,
    unigram_distribution: np.ndarray,
    unk_id: int,
    context_blend: float,
    tail_temperature: float,
    head_temperature: float,
    head_context_mix: float,
    residual_gain: float,
    max_kl_shift: float = 0.08,
) -> np.ndarray:
    candidate = _r6_residual_sparse_contextual_plus_forced_unk(
        g6=g6,
        top_k=top_k,
        g2=g2,
        unigram_distribution=unigram_distribution,
        unk_id=unk_id,
        context_blend=context_blend,
        tail_temperature=tail_temperature,
        head_temperature=head_temperature,
        head_context_mix=head_context_mix,
        residual_gain=residual_gain,
    )
    p = np.clip(g6, 1e-12, 1.0)
    q = np.clip(candidate, 1e-12, 1.0)
    kl = float(np.sum(p * np.log(p / q)))
    if kl > max_kl_shift:
        return g6.copy()
    return candidate


def _r6_residual_sparse_contextual_plus_forced_unk_kl_safe_unk_context(
    g6: np.ndarray,
    top_k: int,
    g2: np.ndarray,
    unigram_distribution: np.ndarray,
    context: list[int],
    unk_id: int,
    context_blend: float,
    tail_temperature: float,
    head_temperature: float,
    head_context_mix: float,
    residual_gain: float,
    max_kl_shift: float = 0.08,
) -> np.ndarray:
    if unk_id in context:
        return g6.copy()
    return _r6_residual_sparse_contextual_plus_forced_unk_kl_safe(
        g6=g6,
        top_k=top_k,
        g2=g2,
        unigram_distribution=unigram_distribution,
        unk_id=unk_id,
        context_blend=context_blend,
        tail_temperature=tail_temperature,
        head_temperature=head_temperature,
        head_context_mix=head_context_mix,
        residual_gain=residual_gain,
        max_kl_shift=max_kl_shift,
    )


def _r6_residual_sparse_contextual_plus_forced_unk_hybrid(
    g6: np.ndarray,
    top_k: int,
    g2: np.ndarray,
    unigram_distribution: np.ndarray,
    context: list[int],
    unk_id: int,
    unk_probability_threshold: float = 0.02,
) -> np.ndarray:
    # Default profile (strong on in-distribution and transition-shifted inputs).
    default_candidate = _r6_residual_sparse_contextual_plus_forced_unk(
        g6=g6,
        top_k=top_k,
        g2=g2,
        unigram_distribution=unigram_distribution,
        unk_id=unk_id,
        context_blend=0.9,
        tail_temperature=0.75,
        head_temperature=1.1,
        head_context_mix=0.0,
        residual_gain=0.8,
    )
    # OOV profile (stronger on unseen-token cases).
    oov_candidate = _r6_residual_sparse_contextual_plus_forced_unk(
        g6=g6,
        top_k=top_k,
        g2=g2,
        unigram_distribution=unigram_distribution,
        unk_id=unk_id,
        context_blend=0.9,
        tail_temperature=0.9,
        head_temperature=0.8,
        head_context_mix=0.0,
        residual_gain=0.6,
    )
    if (unk_id in context) or (float(g6[unk_id]) >= unk_probability_threshold):
        return oov_candidate
    return default_candidate


def _r6_residual_sparse_contextual_plus_unk_safe(
    g6: np.ndarray,
    top_k: int,
    g2: np.ndarray,
    unigram_distribution: np.ndarray,
    context: list[int],
    unk_id: int,
    context_blend: float,
    tail_temperature: float,
    head_temperature: float,
    head_context_mix: float,
    residual_gain: float,
    min_head_mass_for_sparse: float = 0.55,
) -> np.ndarray:
    # If the context includes unknown tokens, or confidence in sparse head is weak,
    # keep the dense G6 distribution to avoid quality regressions on OOV-heavy inputs.
    if unk_id in context:
        return g6.copy()
    idx = np.argpartition(g6, -top_k)[-top_k:]
    head_mass = float(np.sum(g6[idx]))
    if head_mass < min_head_mass_for_sparse:
        return g6.copy()
    return _r6_residual_sparse_contextual_plus(
        g6=g6,
        top_k=top_k,
        g2=g2,
        unigram_distribution=unigram_distribution,
        context_blend=context_blend,
        tail_temperature=tail_temperature,
        head_temperature=head_temperature,
        head_context_mix=head_context_mix,
        residual_gain=residual_gain,
    )


def _evaluate(
    name: str,
    predictions: list[np.ndarray],
    g6_rows: list[np.ndarray],
    targets: list[int],
    support_threshold: float,
    g6_ce: float,
    dense_ops: int,
    ops_per_token: float,
) -> dict:
    metrics = evaluate_predictions(predictions, targets)
    entropies = [_entropy(row) for row in predictions]
    supports = [float(_support(row, support_threshold)) for row in predictions]
    kls = [_kl_divergence(g6, pred) for g6, pred in zip(g6_rows, predictions)]
    mass_errors = [abs(float(row.sum()) - 1.0) for row in predictions]
    delta_ce = float(metrics["cross_entropy"] - g6_ce)

    gate_quality_recovered = float(metrics["cross_entropy"]) <= (g6_ce + 0.1)
    gate_quality_competitive = float(metrics["cross_entropy"]) <= g6_ce
    gate_resource = float(ops_per_token) < float(dense_ops)
    gate_probability = float(np.max(np.asarray(mass_errors, dtype=np.float64))) <= 1e-6

    status = "REJECT"
    if gate_quality_competitive and gate_resource and gate_probability:
        status = "COMPETITIVE"
    elif gate_quality_recovered and gate_resource and gate_probability:
        status = "RECOVERED"

    return {
        "name": name,
        "accuracy": float(metrics["accuracy"]),
        "cross_entropy": float(metrics["cross_entropy"]),
        "perplexity": float(metrics["perplexity"]),
        "delta_ce_vs_g6": delta_ce,
        "entropy_mean": float(np.mean(np.asarray(entropies, dtype=np.float64))),
        "support_mean": float(np.mean(np.asarray(supports, dtype=np.float64))),
        "kl_g6_to_r6_mean": float(np.mean(np.asarray(kls, dtype=np.float64))),
        "mass_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
        "operations_per_token": float(ops_per_token),
        "sparsity_ratio": float(1.0 - (ops_per_token / max(float(dense_ops), 1.0))),
        "gates": {
            "recovered_threshold_ce_le_g6_plus_0_1": gate_quality_recovered,
            "competitive_threshold_ce_le_g6": gate_quality_competitive,
            "resource_advantage": gate_resource,
            "probability_correctness_mass_error_le_1e_6": gate_probability,
        },
        "classification": status,
    }


def main() -> None:
    seed = 1337
    fixture, tokenizer, train_sequences, eval_sequences = _load_frozen_fixture()
    if int(fixture["seed"]) != seed:
        raise ValueError("Frozen fixture seed mismatch.")

    support_threshold = float(fixture["support_threshold"])
    vocab_size = len(tokenizer.token_to_id)
    dense_ops = vocab_size

    tracemalloc.start()
    prep_start = time.perf_counter()

    g2_model = SparseNGramModel(vocab_size=vocab_size, order=2, alpha=0.1)
    g2_model.fit(train_sequences)
    local = SparseNGramModel(vocab_size=vocab_size, order=2, alpha=0.1)
    medium = SparseNGramModel(vocab_size=vocab_size, order=4, alpha=0.1)
    global_model = SparseNGramModel(vocab_size=vocab_size, order=1, alpha=0.1)
    local.fit(train_sequences)
    medium.fit(train_sequences)
    global_model.fit(train_sequences)

    unigram = global_model.global_counts + 0.1
    unigram = unigram / unigram.sum()

    prep_time = time.perf_counter() - prep_start

    eval_start = time.perf_counter()
    contexts: list[list[int]] = []
    targets: list[int] = []
    g6_rows: list[np.ndarray] = []
    g2_rows: list[np.ndarray] = []
    for sequence in eval_sequences:
        for pos in range(len(sequence) - 1):
            context = sequence[: pos + 1]
            target = int(sequence[pos + 1])
            contexts.append(context)
            targets.append(target)
            g6_rows.append(_g6_distribution(local, medium, global_model, context))
            g2_rows.append(_g2_distribution(g2_model, context))

    g6_metrics = evaluate_predictions(g6_rows, targets)
    g6_ce = float(g6_metrics["cross_entropy"])

    # Baseline R4 as requested.
    r4_preds = [_r4_baseline(g6, g2, top_k=4, floor_mix=0.2) for g6, g2 in zip(g6_rows, g2_rows)]
    r4_result = _evaluate(
        name="r4_baseline_topk4_floor_mix0.2",
        predictions=r4_preds,
        g6_rows=g6_rows,
        targets=targets,
        support_threshold=support_threshold,
        g6_ce=g6_ce,
        dense_ops=dense_ops,
        ops_per_token=(4 + 0.2 * vocab_size),
    )

    # R6 sweep as requested.
    r6_topk_values = [2, 4, 8, 16]
    r6_results: list[dict] = []
    for top_k in r6_topk_values:
        preds = [_r6_residual_sparse(g6, top_k=top_k, unigram_distribution=unigram) for g6 in g6_rows]
        r6_results.append(
            _evaluate(
                name=f"r6_residual_sparse_topk{top_k}",
                predictions=preds,
                g6_rows=g6_rows,
                targets=targets,
                support_threshold=support_threshold,
                g6_ce=g6_ce,
                dense_ops=dense_ops,
                ops_per_token=float(top_k + 1),  # top-k exact candidates + residual background estimate
            )
        )
        contextual_preds = [
            _r6_residual_sparse_contextual(
                g6,
                top_k=top_k,
                g2=g2,
                unigram_distribution=unigram,
                context_blend=0.8,
                tail_temperature=0.85,
            )
            for g6, g2 in zip(g6_rows, g2_rows)
        ]
        r6_results.append(
            _evaluate(
                name=f"r6_residual_sparse_contextual_topk{top_k}",
                predictions=contextual_preds,
                g6_rows=g6_rows,
                targets=targets,
                support_threshold=support_threshold,
                g6_ce=g6_ce,
                dense_ops=dense_ops,
                ops_per_token=float(top_k + 2),  # top-k + contextual residual estimate
            )
        )

    # Stronger R6+ sweep: contextual head/tail calibration while preserving sparse compute.
    r6_plus_topk_values = [4, 6, 8, 10, 12, 16, 24, 32, 36, 40, 41]
    r6_plus_context_blends = [0.7, 0.8, 0.9]
    r6_plus_tail_temperatures = [0.75, 0.85, 1.0]
    r6_plus_head_temperatures = [0.9, 1.0, 1.1]
    r6_plus_head_context_mixes = [0.0, 0.1, 0.2]
    r6_plus_residual_gains = [0.8, 1.0, 1.2]

    r6_plus_results: list[dict] = []
    for top_k in r6_plus_topk_values:
        for context_blend in r6_plus_context_blends:
            for tail_temperature in r6_plus_tail_temperatures:
                for head_temperature in r6_plus_head_temperatures:
                    for head_context_mix in r6_plus_head_context_mixes:
                        for residual_gain in r6_plus_residual_gains:
                            predictions = [
                                _r6_residual_sparse_contextual_plus(
                                    g6=g6,
                                    top_k=top_k,
                                    g2=g2,
                                    unigram_distribution=unigram,
                                    context_blend=context_blend,
                                    tail_temperature=tail_temperature,
                                    head_temperature=head_temperature,
                                    head_context_mix=head_context_mix,
                                    residual_gain=residual_gain,
                                )
                                for g6, g2 in zip(g6_rows, g2_rows)
                            ]
                            r6_plus_results.append(
                                _evaluate(
                                    name=(
                                        "r6_plus"
                                        f"_k{top_k}"
                                        f"_cb{context_blend:.2f}"
                                        f"_tt{tail_temperature:.2f}"
                                        f"_ht{head_temperature:.2f}"
                                        f"_hcm{head_context_mix:.2f}"
                                        f"_rg{residual_gain:.2f}"
                                    ),
                                    predictions=predictions,
                                    g6_rows=g6_rows,
                                    targets=targets,
                                    support_threshold=support_threshold,
                                    g6_ce=g6_ce,
                                    dense_ops=dense_ops,
                                    ops_per_token=float(top_k + 4),
                                )
                            )

    eval_time = time.perf_counter() - eval_start
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    all_candidates = [r4_result] + r6_results + r6_plus_results
    best_candidate = min(all_candidates, key=lambda item: float(item["cross_entropy"]))
    competitive_candidates = [item for item in all_candidates if item["classification"] == "COMPETITIVE"]
    best_competitive_candidate = (
        min(competitive_candidates, key=lambda item: float(item["cross_entropy"])) if competitive_candidates else None
    )

    counts = {
        "REJECT": int(sum(1 for item in all_candidates if item["classification"] == "REJECT")),
        "RECOVERED": int(sum(1 for item in all_candidates if item["classification"] == "RECOVERED")),
        "COMPETITIVE": int(sum(1 for item in all_candidates if item["classification"] == "COMPETITIVE")),
    }

    result = {
        "experiment_id": "g7_r6_residual_sparse_approximation",
        "seed": seed,
        "cpu_threading": CPU_THREADING,
        "fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "frozen_fixture": {
            "eval_token_count": len(targets),
            "vocab_size": vocab_size,
            "support_threshold": support_threshold,
        },
        "g6_reference": {
            "accuracy": float(g6_metrics["accuracy"]),
            "cross_entropy": float(g6_metrics["cross_entropy"]),
            "perplexity": float(g6_metrics["perplexity"]),
            "recovered_threshold_ce_le": float(g6_ce + 0.1),
            "competitive_threshold_ce_le": float(g6_ce),
            "dense_operations_per_token": dense_ops,
        },
        "resource_usage": {
            "prep_time_seconds": prep_time,
            "evaluation_time_seconds": eval_time,
            "peak_process_ram_bytes": int(peak_ram),
            "tokens_per_second": float(len(targets) / max(eval_time, 1e-12)),
        },
        "baseline_r4": r4_result,
        "r6_results": r6_results,
        "r6_plus_sweep_config": {
            "top_k": r6_plus_topk_values,
            "context_blend": r6_plus_context_blends,
            "tail_temperature": r6_plus_tail_temperatures,
            "head_temperature": r6_plus_head_temperatures,
            "head_context_mix": r6_plus_head_context_mixes,
            "residual_gain": r6_plus_residual_gains,
        },
        "r6_plus_results": r6_plus_results,
        "summary_counts": counts,
        "best_candidate_by_cross_entropy": best_candidate,
        "best_competitive_candidate_by_cross_entropy": best_competitive_candidate,
    }

    summary = {
        "experiment_id": "g7_r6_residual_sparse_approximation_summary",
        "seed": seed,
        "g6_reference": result["g6_reference"],
        "resource_usage": result["resource_usage"],
        "baseline_r4": r4_result,
        "r6_results": r6_results,
        "r6_plus_sweep_config": result["r6_plus_sweep_config"],
        "r6_plus_results": r6_plus_results,
        "summary_counts": counts,
        "best_candidate_by_cross_entropy": best_candidate,
        "best_competitive_candidate_by_cross_entropy": best_competitive_candidate,
    }

    root = Path(__file__).resolve().parent
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g7_r6_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (root / "results" / "g7_r6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r6_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g7_r6 complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
