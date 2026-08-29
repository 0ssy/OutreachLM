from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable

PHASE_G_ROOT = Path(__file__).resolve().parents[2]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.cpu_threads import configure_cpu_threads_from_env  # noqa: E402
from common.datasets import CONTEXT_AMBIGUITY_CORPUS, LONG_CONTEXT_CORPUS, build_train_eval_split  # noqa: E402
from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402
from run import (
    _r4_baseline,
    _r6_residual_sparse_contextual_plus,
    _r6_residual_sparse_contextual_plus_forced_unk,
    _r6_residual_sparse_contextual_plus_forced_unk_hybrid,
    _r6_residual_sparse_contextual_plus_forced_unk_kl_safe,
    _r6_residual_sparse_contextual_plus_forced_unk_kl_safe_unk_context,
    _r6_residual_sparse_contextual_plus_unk_safe,
)  # noqa: E402

CPU_THREADING = configure_cpu_threads_from_env()

import numpy as np


WINNING_CONFIG = {
    "top_k": 4,
    "context_blend": 0.90,
    "tail_temperature": 0.75,
    "head_temperature": 1.10,
    "head_context_mix": 0.00,
    "residual_gain": 0.80,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    lines: list[str]
    notes: str


def _load_fixture() -> tuple[dict, StupidTokenizer]:
    fixture_path = PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tokenizer = StupidTokenizer(token_to_id=fixture["token_to_id"])
    return fixture, tokenizer


def _entropy(probabilities: np.ndarray, epsilon: float = 1e-12) -> float:
    p = np.clip(probabilities, epsilon, 1.0)
    return float(-(p * np.log(p)).sum())


def _support(probabilities: np.ndarray, threshold: float) -> int:
    return int(np.count_nonzero(probabilities > threshold))


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _g6_distribution(local: SparseNGramModel, medium: SparseNGramModel, global_model: SparseNGramModel, context: list[int]) -> np.ndarray:
    out = 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)
    return out / out.sum()


def _build_unseen_transition_lines(corpus: list[str], limit: int = 64) -> list[str]:
    existing = set(corpus)
    subjects = ["the cat", "the dog", "the boy", "the girl"]
    verbs = ["saw", "fed", "ate", "likes", "will"]
    objects = ["the dog", "the cat", "the food", "the fish", "milk", "bones", "people", "birds"]
    candidates: list[str] = []
    for s, v, o in itertools.product(subjects, verbs, objects):
        line = f"{s} {v} {o}"
        if line not in existing:
            candidates.append(line)
    return candidates[:limit]


def _build_unseen_token_lines(limit: int = 64) -> list[str]:
    lines = [
        "the quokka saw the cat",
        "the dog fed nebula kibble",
        "the boy will teleport",
        "the girl likes quantum milk",
        "the cat ate hyperfish",
        "the dog saw ultrapeople",
        "the boy fed metacat",
        "the girl will hypersleep",
    ]
    repeated: list[str] = []
    while len(repeated) < limit:
        repeated.extend(lines)
    return repeated[:limit]


def _build_scenarios(base_corpus: list[str], eval_lines: list[str]) -> list[Scenario]:
    return [
        Scenario("baseline_split", eval_lines, "Original split distribution."),
        Scenario("larger_eval_corpus", (base_corpus * 8) + eval_lines, "Larger corpus for stronger statistical power."),
        Scenario("context_long_heavy", (LONG_CONTEXT_CORPUS * 40) + eval_lines, "Long-context shifted distribution."),
        Scenario("context_ambiguity_heavy", (CONTEXT_AMBIGUITY_CORPUS * 40) + eval_lines, "Ambiguity-shifted distribution."),
        Scenario("unseen_transitions", _build_unseen_transition_lines(base_corpus, limit=96), "Novel token transitions with known vocabulary."),
        Scenario("unseen_tokens", _build_unseen_token_lines(limit=96), "Out-of-vocabulary words mapped to <UNK>."),
    ]


def _encode_lines(tokenizer: StupidTokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def _extract_rows(
    eval_sequences: list[list[int]],
    g2_model: SparseNGramModel,
    local: SparseNGramModel,
    medium: SparseNGramModel,
    global_model: SparseNGramModel,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[list[int]]]:
    g6_rows: list[np.ndarray] = []
    g2_rows: list[np.ndarray] = []
    targets: list[int] = []
    contexts: list[list[int]] = []
    for sequence in eval_sequences:
        for pos in range(len(sequence) - 1):
            context = sequence[: pos + 1]
            contexts.append(context)
            targets.append(int(sequence[pos + 1]))
            g6_rows.append(_g6_distribution(local, medium, global_model, context))
            g2_rows.append(g2_model.distribution(context))
    return g6_rows, g2_rows, targets, contexts


def _flatten_bigrams(sequences: list[list[int]]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for seq in sequences:
        output.extend((int(seq[i]), int(seq[i + 1])) for i in range(len(seq) - 1))
    return output


def _model_predictions(
    model_name: str,
    g6_rows: list[np.ndarray],
    g2_rows: list[np.ndarray],
    unigram: np.ndarray,
    contexts: list[list[int]],
    unk_id: int,
) -> list[np.ndarray]:
    if model_name == "g6":
        return [row.copy() for row in g6_rows]
    if model_name == "r4":
        return [_r4_baseline(g6, g2, top_k=4, floor_mix=0.2) for g6, g2 in zip(g6_rows, g2_rows)]
    if model_name == "r6_plus":
        return [
            _r6_residual_sparse_contextual_plus(
                g6=g6,
                top_k=WINNING_CONFIG["top_k"],
                g2=g2,
                unigram_distribution=unigram,
                context_blend=WINNING_CONFIG["context_blend"],
                tail_temperature=WINNING_CONFIG["tail_temperature"],
                head_temperature=WINNING_CONFIG["head_temperature"],
                head_context_mix=WINNING_CONFIG["head_context_mix"],
                residual_gain=WINNING_CONFIG["residual_gain"],
            )
            for g6, g2 in zip(g6_rows, g2_rows)
        ]
    if model_name == "r6_plus_unk_safe":
        return [
            _r6_residual_sparse_contextual_plus_unk_safe(
                g6=g6,
                top_k=WINNING_CONFIG["top_k"],
                g2=g2,
                unigram_distribution=unigram,
                context=context,
                unk_id=unk_id,
                context_blend=WINNING_CONFIG["context_blend"],
                tail_temperature=WINNING_CONFIG["tail_temperature"],
                head_temperature=WINNING_CONFIG["head_temperature"],
                head_context_mix=WINNING_CONFIG["head_context_mix"],
                residual_gain=WINNING_CONFIG["residual_gain"],
                min_head_mass_for_sparse=0.55,
            )
            for g6, g2, context in zip(g6_rows, g2_rows, contexts)
        ]
    if model_name == "r6_plus_forced_unk":
        return [
            _r6_residual_sparse_contextual_plus_forced_unk(
                g6=g6,
                top_k=WINNING_CONFIG["top_k"],
                g2=g2,
                unigram_distribution=unigram,
                unk_id=unk_id,
                context_blend=WINNING_CONFIG["context_blend"],
                tail_temperature=WINNING_CONFIG["tail_temperature"],
                head_temperature=WINNING_CONFIG["head_temperature"],
                head_context_mix=WINNING_CONFIG["head_context_mix"],
                residual_gain=WINNING_CONFIG["residual_gain"],
            )
            for g6, g2 in zip(g6_rows, g2_rows)
        ]
    if model_name == "r6_plus_forced_unk_kl_safe":
        return [
            _r6_residual_sparse_contextual_plus_forced_unk_kl_safe(
                g6=g6,
                top_k=WINNING_CONFIG["top_k"],
                g2=g2,
                unigram_distribution=unigram,
                unk_id=unk_id,
                context_blend=WINNING_CONFIG["context_blend"],
                tail_temperature=WINNING_CONFIG["tail_temperature"],
                head_temperature=WINNING_CONFIG["head_temperature"],
                head_context_mix=WINNING_CONFIG["head_context_mix"],
                residual_gain=WINNING_CONFIG["residual_gain"],
                max_kl_shift=0.08,
            )
            for g6, g2 in zip(g6_rows, g2_rows)
        ]
    if model_name == "r6_plus_forced_unk_kl_safe_unk_context":
        return [
            _r6_residual_sparse_contextual_plus_forced_unk_kl_safe_unk_context(
                g6=g6,
                top_k=WINNING_CONFIG["top_k"],
                g2=g2,
                unigram_distribution=unigram,
                context=context,
                unk_id=unk_id,
                context_blend=WINNING_CONFIG["context_blend"],
                tail_temperature=WINNING_CONFIG["tail_temperature"],
                head_temperature=WINNING_CONFIG["head_temperature"],
                head_context_mix=WINNING_CONFIG["head_context_mix"],
                residual_gain=WINNING_CONFIG["residual_gain"],
                max_kl_shift=0.08,
            )
            for g6, g2, context in zip(g6_rows, g2_rows, contexts)
        ]
    if model_name == "r6_plus_forced_unk_hybrid":
        return [
            _r6_residual_sparse_contextual_plus_forced_unk_hybrid(
                g6=g6,
                top_k=WINNING_CONFIG["top_k"],
                g2=g2,
                unigram_distribution=unigram,
                context=context,
                unk_id=unk_id,
                unk_probability_threshold=0.02,
            )
            for g6, g2, context in zip(g6_rows, g2_rows, contexts)
        ]
    raise ValueError(f"Unknown model_name: {model_name}")


def _predict_parallel(
    model_name: str,
    g6_rows: list[np.ndarray],
    g2_rows: list[np.ndarray],
    unigram: np.ndarray,
    contexts: list[list[int]],
    unk_id: int,
    workers: int,
) -> list[np.ndarray]:
    if workers <= 1:
        return _model_predictions(model_name, g6_rows, g2_rows, unigram, contexts, unk_id)

    chunk_size = max(1, len(g6_rows) // workers)
    chunks = []
    for i in range(0, len(g6_rows), chunk_size):
        chunks.append((g6_rows[i : i + chunk_size], g2_rows[i : i + chunk_size], contexts[i : i + chunk_size]))

    def _run_chunk(chunk: tuple[list[np.ndarray], list[np.ndarray], list[list[int]]]) -> list[np.ndarray]:
        c_g6, c_g2, c_ctx = chunk
        return _model_predictions(model_name, c_g6, c_g2, unigram, c_ctx, unk_id)

    out: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for rows in executor.map(_run_chunk, chunks):
            out.extend(rows)
    return out


def _bootstrap_variance(
    predictions: list[np.ndarray],
    targets: list[int],
    repeats: int,
    rng_seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    count = len(targets)
    ces: list[float] = []
    accs: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, count, size=count)
        sampled_preds = [predictions[int(i)] for i in indices]
        sampled_targets = [targets[int(i)] for i in indices]
        metrics = evaluate_predictions(sampled_preds, sampled_targets)
        ces.append(float(metrics["cross_entropy"]))
        accs.append(float(metrics["accuracy"]))
    return {
        "repeat_count": repeats,
        "ce_mean": float(np.mean(ces)),
        "ce_std": float(np.std(ces)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
    }


def _summarize_model(
    model_name: str,
    predictions: list[np.ndarray],
    g6_rows: list[np.ndarray],
    targets: list[int],
    support_threshold: float,
) -> dict[str, float | dict | list]:
    metrics = evaluate_predictions(predictions, targets)
    entropies = [_entropy(row) for row in predictions]
    supports = [_support(row, support_threshold) for row in predictions]
    kls = [_kl_divergence(g6, pred) for g6, pred in zip(g6_rows, predictions)]
    mass_errors = [abs(float(row.sum()) - 1.0) for row in predictions]
    return {
        "model": model_name,
        "accuracy": float(metrics["accuracy"]),
        "cross_entropy": float(metrics["cross_entropy"]),
        "perplexity": float(metrics["perplexity"]),
        "entropy_mean": float(np.mean(np.asarray(entropies, dtype=np.float64))),
        "support_mean": float(np.mean(np.asarray(supports, dtype=np.float64))),
        "kl_g6_to_model_mean": float(np.mean(np.asarray(kls, dtype=np.float64))),
        "mass_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
    }


def _cpu_benchmark(
    model_name: str,
    g6_rows: list[np.ndarray],
    g2_rows: list[np.ndarray],
    targets: list[int],
    unigram: np.ndarray,
    contexts: list[list[int]],
    unk_id: int,
    workers: list[int],
    repeats: int = 3,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for worker_count in workers:
        elapsed: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            predictions = _predict_parallel(model_name, g6_rows, g2_rows, unigram, contexts, unk_id, worker_count)
            _ = evaluate_predictions(predictions, targets)
            elapsed.append(time.perf_counter() - start)
        mean_elapsed = float(np.mean(np.asarray(elapsed, dtype=np.float64)))
        result[f"workers_{worker_count}"] = {
            "mean_elapsed_seconds": mean_elapsed,
            "tokens_per_second": float(len(targets) / max(mean_elapsed, 1e-12)),
        }
    return result


def main() -> None:
    fixture, tokenizer = _load_fixture()
    support_threshold = float(fixture["support_threshold"])
    base_corpus = list(fixture["corpus"])
    eval_ratio = float(fixture["eval_ratio"])

    seeds = [1337, 2024, 4242, 9001, 12345]
    cpu_workers = [1, 2, 4]
    bootstrap_repeats = 10

    rows: list[dict] = []
    start = time.perf_counter()
    for seed in seeds:
        train_lines, eval_lines = build_train_eval_split(base_corpus, seed=seed, eval_ratio=eval_ratio)
        train_sequences = _encode_lines(tokenizer, train_lines)
        train_bigrams = set(_flatten_bigrams(train_sequences))

        g2_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
        g2_model.fit(train_sequences)
        local = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
        medium = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=4, alpha=0.1)
        global_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=1, alpha=0.1)
        local.fit(train_sequences)
        medium.fit(train_sequences)
        global_model.fit(train_sequences)
        unigram = global_model.global_counts + 0.1
        unigram = unigram / unigram.sum()

        scenarios = _build_scenarios(base_corpus, eval_lines)
        for scenario in scenarios:
            eval_sequences = _encode_lines(tokenizer, scenario.lines)
            g6_rows, g2_rows, targets, contexts = _extract_rows(eval_sequences, g2_model, local, medium, global_model)
            eval_bigrams = _flatten_bigrams(eval_sequences)
            unseen_bigram_count = sum(1 for bg in eval_bigrams if bg not in train_bigrams)
            unk_token_count = sum(1 for seq in eval_sequences for token in seq if token == tokenizer.unk_id)

            g6_preds = _model_predictions("g6", g6_rows, g2_rows, unigram, contexts, tokenizer.unk_id)
            r4_preds = _model_predictions("r4", g6_rows, g2_rows, unigram, contexts, tokenizer.unk_id)
            r6_preds = _model_predictions("r6_plus", g6_rows, g2_rows, unigram, contexts, tokenizer.unk_id)
            r6_unk_safe_preds = _model_predictions("r6_plus_unk_safe", g6_rows, g2_rows, unigram, contexts, tokenizer.unk_id)
            r6_forced_unk_preds = _model_predictions(
                "r6_plus_forced_unk",
                g6_rows,
                g2_rows,
                unigram,
                contexts,
                tokenizer.unk_id,
            )
            r6_forced_unk_kl_safe_preds = _model_predictions(
                "r6_plus_forced_unk_kl_safe",
                g6_rows,
                g2_rows,
                unigram,
                contexts,
                tokenizer.unk_id,
            )
            r6_forced_unk_kl_safe_unk_context_preds = _model_predictions(
                "r6_plus_forced_unk_kl_safe_unk_context",
                g6_rows,
                g2_rows,
                unigram,
                contexts,
                tokenizer.unk_id,
            )
            r6_forced_unk_hybrid_preds = _model_predictions(
                "r6_plus_forced_unk_hybrid",
                g6_rows,
                g2_rows,
                unigram,
                contexts,
                tokenizer.unk_id,
            )

            g6_summary = _summarize_model("g6", g6_preds, g6_rows, targets, support_threshold)
            r4_summary = _summarize_model("r4", r4_preds, g6_rows, targets, support_threshold)
            r6_summary = _summarize_model("r6_plus", r6_preds, g6_rows, targets, support_threshold)
            r6_unk_safe_summary = _summarize_model("r6_plus_unk_safe", r6_unk_safe_preds, g6_rows, targets, support_threshold)
            r6_forced_unk_summary = _summarize_model(
                "r6_plus_forced_unk",
                r6_forced_unk_preds,
                g6_rows,
                targets,
                support_threshold,
            )
            r6_forced_unk_kl_safe_summary = _summarize_model(
                "r6_plus_forced_unk_kl_safe",
                r6_forced_unk_kl_safe_preds,
                g6_rows,
                targets,
                support_threshold,
            )
            r6_forced_unk_kl_safe_unk_context_summary = _summarize_model(
                "r6_plus_forced_unk_kl_safe_unk_context",
                r6_forced_unk_kl_safe_unk_context_preds,
                g6_rows,
                targets,
                support_threshold,
            )
            r6_forced_unk_hybrid_summary = _summarize_model(
                "r6_plus_forced_unk_hybrid",
                r6_forced_unk_hybrid_preds,
                g6_rows,
                targets,
                support_threshold,
            )

            g6_ce = float(g6_summary["cross_entropy"])
            r4_ce = float(r4_summary["cross_entropy"])
            r6_ce = float(r6_summary["cross_entropy"])
            r6_unk_safe_ce = float(r6_unk_safe_summary["cross_entropy"])
            r6_forced_unk_ce = float(r6_forced_unk_summary["cross_entropy"])
            r6_forced_unk_kl_safe_ce = float(r6_forced_unk_kl_safe_summary["cross_entropy"])
            r6_forced_unk_kl_safe_unk_context_ce = float(r6_forced_unk_kl_safe_unk_context_summary["cross_entropy"])
            r6_forced_unk_hybrid_ce = float(r6_forced_unk_hybrid_summary["cross_entropy"])

            row = {
                "seed": seed,
                "scenario": scenario.name,
                "scenario_notes": scenario.notes,
                "eval_line_count": len(scenario.lines),
                "eval_token_count": len(targets),
                "unseen_transition_rate": float(unseen_bigram_count / max(len(eval_bigrams), 1)),
                "unk_token_rate": float(unk_token_count / max(sum(len(x) for x in eval_sequences), 1)),
                "models": {
                    "g6": g6_summary,
                    "r4": r4_summary,
                    "r6_plus": r6_summary,
                    "r6_plus_unk_safe": r6_unk_safe_summary,
                    "r6_plus_forced_unk": r6_forced_unk_summary,
                    "r6_plus_forced_unk_kl_safe": r6_forced_unk_kl_safe_summary,
                    "r6_plus_forced_unk_kl_safe_unk_context": r6_forced_unk_kl_safe_unk_context_summary,
                    "r6_plus_forced_unk_hybrid": r6_forced_unk_hybrid_summary,
                },
                "comparison": {
                    "r6_minus_g6_ce": float(r6_ce - g6_ce),
                    "r6_minus_r4_ce": float(r6_ce - r4_ce),
                    "r6_competitive_vs_g6": bool(r6_ce <= g6_ce),
                    "r6_recovered_vs_g6_plus_0_1": bool(r6_ce <= (g6_ce + 0.1)),
                    "r6_beats_r4": bool(r6_ce <= r4_ce),
                    "mass_correctness_r6_le_1e_6": bool(float(r6_summary["mass_error_max"]) <= 1e-6),
                    "r6_unk_safe_minus_g6_ce": float(r6_unk_safe_ce - g6_ce),
                    "r6_unk_safe_minus_r4_ce": float(r6_unk_safe_ce - r4_ce),
                    "r6_unk_safe_competitive_vs_g6": bool(r6_unk_safe_ce <= g6_ce),
                    "r6_unk_safe_recovered_vs_g6_plus_0_1": bool(r6_unk_safe_ce <= (g6_ce + 0.1)),
                    "r6_unk_safe_beats_r4": bool(r6_unk_safe_ce <= r4_ce),
                    "mass_correctness_r6_unk_safe_le_1e_6": bool(float(r6_unk_safe_summary["mass_error_max"]) <= 1e-6),
                    "r6_forced_unk_minus_g6_ce": float(r6_forced_unk_ce - g6_ce),
                    "r6_forced_unk_minus_r4_ce": float(r6_forced_unk_ce - r4_ce),
                    "r6_forced_unk_competitive_vs_g6": bool(r6_forced_unk_ce <= g6_ce),
                    "r6_forced_unk_recovered_vs_g6_plus_0_1": bool(r6_forced_unk_ce <= (g6_ce + 0.1)),
                    "r6_forced_unk_beats_r4": bool(r6_forced_unk_ce <= r4_ce),
                    "mass_correctness_r6_forced_unk_le_1e_6": bool(float(r6_forced_unk_summary["mass_error_max"]) <= 1e-6),
                    "r6_forced_unk_kl_safe_minus_g6_ce": float(r6_forced_unk_kl_safe_ce - g6_ce),
                    "r6_forced_unk_kl_safe_minus_r4_ce": float(r6_forced_unk_kl_safe_ce - r4_ce),
                    "r6_forced_unk_kl_safe_competitive_vs_g6": bool(r6_forced_unk_kl_safe_ce <= g6_ce),
                    "r6_forced_unk_kl_safe_recovered_vs_g6_plus_0_1": bool(r6_forced_unk_kl_safe_ce <= (g6_ce + 0.1)),
                    "r6_forced_unk_kl_safe_beats_r4": bool(r6_forced_unk_kl_safe_ce <= r4_ce),
                    "mass_correctness_r6_forced_unk_kl_safe_le_1e_6": bool(
                        float(r6_forced_unk_kl_safe_summary["mass_error_max"]) <= 1e-6
                    ),
                    "r6_forced_unk_kl_safe_unk_context_minus_g6_ce": float(r6_forced_unk_kl_safe_unk_context_ce - g6_ce),
                    "r6_forced_unk_kl_safe_unk_context_minus_r4_ce": float(r6_forced_unk_kl_safe_unk_context_ce - r4_ce),
                    "r6_forced_unk_kl_safe_unk_context_competitive_vs_g6": bool(r6_forced_unk_kl_safe_unk_context_ce <= g6_ce),
                    "r6_forced_unk_kl_safe_unk_context_recovered_vs_g6_plus_0_1": bool(
                        r6_forced_unk_kl_safe_unk_context_ce <= (g6_ce + 0.1)
                    ),
                    "r6_forced_unk_kl_safe_unk_context_beats_r4": bool(r6_forced_unk_kl_safe_unk_context_ce <= r4_ce),
                    "mass_correctness_r6_forced_unk_kl_safe_unk_context_le_1e_6": bool(
                        float(r6_forced_unk_kl_safe_unk_context_summary["mass_error_max"]) <= 1e-6
                    ),
                    "r6_forced_unk_hybrid_minus_g6_ce": float(r6_forced_unk_hybrid_ce - g6_ce),
                    "r6_forced_unk_hybrid_minus_r4_ce": float(r6_forced_unk_hybrid_ce - r4_ce),
                    "r6_forced_unk_hybrid_competitive_vs_g6": bool(r6_forced_unk_hybrid_ce <= g6_ce),
                    "r6_forced_unk_hybrid_recovered_vs_g6_plus_0_1": bool(r6_forced_unk_hybrid_ce <= (g6_ce + 0.1)),
                    "r6_forced_unk_hybrid_beats_r4": bool(r6_forced_unk_hybrid_ce <= r4_ce),
                    "mass_correctness_r6_forced_unk_hybrid_le_1e_6": bool(
                        float(r6_forced_unk_hybrid_summary["mass_error_max"]) <= 1e-6
                    ),
                },
                "variance": {
                    "g6": _bootstrap_variance(g6_preds, targets, repeats=bootstrap_repeats, rng_seed=(seed * 1000) + 11),
                    "r4": _bootstrap_variance(r4_preds, targets, repeats=bootstrap_repeats, rng_seed=(seed * 1000) + 22),
                    "r6_plus": _bootstrap_variance(r6_preds, targets, repeats=bootstrap_repeats, rng_seed=(seed * 1000) + 33),
                    "r6_plus_unk_safe": _bootstrap_variance(
                        r6_unk_safe_preds,
                        targets,
                        repeats=bootstrap_repeats,
                        rng_seed=(seed * 1000) + 44,
                    ),
                    "r6_plus_forced_unk": _bootstrap_variance(
                        r6_forced_unk_preds,
                        targets,
                        repeats=bootstrap_repeats,
                        rng_seed=(seed * 1000) + 55,
                    ),
                    "r6_plus_forced_unk_kl_safe": _bootstrap_variance(
                        r6_forced_unk_kl_safe_preds,
                        targets,
                        repeats=bootstrap_repeats,
                        rng_seed=(seed * 1000) + 66,
                    ),
                    "r6_plus_forced_unk_kl_safe_unk_context": _bootstrap_variance(
                        r6_forced_unk_kl_safe_unk_context_preds,
                        targets,
                        repeats=bootstrap_repeats,
                        rng_seed=(seed * 1000) + 77,
                    ),
                    "r6_plus_forced_unk_hybrid": _bootstrap_variance(
                        r6_forced_unk_hybrid_preds,
                        targets,
                        repeats=bootstrap_repeats,
                        rng_seed=(seed * 1000) + 88,
                    ),
                },
                "cpu_benchmark": {
                    "g6": _cpu_benchmark("g6", g6_rows, g2_rows, targets, unigram, contexts, tokenizer.unk_id, workers=cpu_workers),
                    "r4": _cpu_benchmark("r4", g6_rows, g2_rows, targets, unigram, contexts, tokenizer.unk_id, workers=cpu_workers),
                    "r6_plus": _cpu_benchmark(
                        "r6_plus",
                        g6_rows,
                        g2_rows,
                        targets,
                        unigram,
                        contexts,
                        tokenizer.unk_id,
                        workers=cpu_workers,
                    ),
                    "r6_plus_unk_safe": _cpu_benchmark(
                        "r6_plus_unk_safe",
                        g6_rows,
                        g2_rows,
                        targets,
                        unigram,
                        contexts,
                        tokenizer.unk_id,
                        workers=cpu_workers,
                    ),
                    "r6_plus_forced_unk": _cpu_benchmark(
                        "r6_plus_forced_unk",
                        g6_rows,
                        g2_rows,
                        targets,
                        unigram,
                        contexts,
                        tokenizer.unk_id,
                        workers=cpu_workers,
                    ),
                    "r6_plus_forced_unk_kl_safe": _cpu_benchmark(
                        "r6_plus_forced_unk_kl_safe",
                        g6_rows,
                        g2_rows,
                        targets,
                        unigram,
                        contexts,
                        tokenizer.unk_id,
                        workers=cpu_workers,
                    ),
                    "r6_plus_forced_unk_kl_safe_unk_context": _cpu_benchmark(
                        "r6_plus_forced_unk_kl_safe_unk_context",
                        g6_rows,
                        g2_rows,
                        targets,
                        unigram,
                        contexts,
                        tokenizer.unk_id,
                        workers=cpu_workers,
                    ),
                    "r6_plus_forced_unk_hybrid": _cpu_benchmark(
                        "r6_plus_forced_unk_hybrid",
                        g6_rows,
                        g2_rows,
                        targets,
                        unigram,
                        contexts,
                        tokenizer.unk_id,
                        workers=cpu_workers,
                    ),
                },
            }
            rows.append(row)

    elapsed = time.perf_counter() - start

    total = len(rows)
    r6_competitive_count = sum(1 for row in rows if row["comparison"]["r6_competitive_vs_g6"])
    r6_recovered_count = sum(1 for row in rows if row["comparison"]["r6_recovered_vs_g6_plus_0_1"])
    r6_beats_r4_count = sum(1 for row in rows if row["comparison"]["r6_beats_r4"])
    mass_ok_count = sum(1 for row in rows if row["comparison"]["mass_correctness_r6_le_1e_6"])
    r6_unk_safe_competitive_count = sum(1 for row in rows if row["comparison"]["r6_unk_safe_competitive_vs_g6"])
    r6_unk_safe_recovered_count = sum(1 for row in rows if row["comparison"]["r6_unk_safe_recovered_vs_g6_plus_0_1"])
    r6_unk_safe_beats_r4_count = sum(1 for row in rows if row["comparison"]["r6_unk_safe_beats_r4"])
    r6_unk_safe_mass_ok_count = sum(1 for row in rows if row["comparison"]["mass_correctness_r6_unk_safe_le_1e_6"])
    r6_forced_unk_competitive_count = sum(1 for row in rows if row["comparison"]["r6_forced_unk_competitive_vs_g6"])
    r6_forced_unk_recovered_count = sum(1 for row in rows if row["comparison"]["r6_forced_unk_recovered_vs_g6_plus_0_1"])
    r6_forced_unk_beats_r4_count = sum(1 for row in rows if row["comparison"]["r6_forced_unk_beats_r4"])
    r6_forced_unk_mass_ok_count = sum(1 for row in rows if row["comparison"]["mass_correctness_r6_forced_unk_le_1e_6"])
    r6_forced_unk_kl_safe_competitive_count = sum(
        1 for row in rows if row["comparison"]["r6_forced_unk_kl_safe_competitive_vs_g6"]
    )
    r6_forced_unk_kl_safe_recovered_count = sum(
        1 for row in rows if row["comparison"]["r6_forced_unk_kl_safe_recovered_vs_g6_plus_0_1"]
    )
    r6_forced_unk_kl_safe_beats_r4_count = sum(1 for row in rows if row["comparison"]["r6_forced_unk_kl_safe_beats_r4"])
    r6_forced_unk_kl_safe_mass_ok_count = sum(
        1 for row in rows if row["comparison"]["mass_correctness_r6_forced_unk_kl_safe_le_1e_6"]
    )
    r6_forced_unk_kl_safe_unk_context_competitive_count = sum(
        1 for row in rows if row["comparison"]["r6_forced_unk_kl_safe_unk_context_competitive_vs_g6"]
    )
    r6_forced_unk_kl_safe_unk_context_recovered_count = sum(
        1 for row in rows if row["comparison"]["r6_forced_unk_kl_safe_unk_context_recovered_vs_g6_plus_0_1"]
    )
    r6_forced_unk_kl_safe_unk_context_beats_r4_count = sum(
        1 for row in rows if row["comparison"]["r6_forced_unk_kl_safe_unk_context_beats_r4"]
    )
    r6_forced_unk_kl_safe_unk_context_mass_ok_count = sum(
        1 for row in rows if row["comparison"]["mass_correctness_r6_forced_unk_kl_safe_unk_context_le_1e_6"]
    )
    r6_forced_unk_hybrid_competitive_count = sum(
        1 for row in rows if row["comparison"]["r6_forced_unk_hybrid_competitive_vs_g6"]
    )
    r6_forced_unk_hybrid_recovered_count = sum(
        1 for row in rows if row["comparison"]["r6_forced_unk_hybrid_recovered_vs_g6_plus_0_1"]
    )
    r6_forced_unk_hybrid_beats_r4_count = sum(1 for row in rows if row["comparison"]["r6_forced_unk_hybrid_beats_r4"])
    r6_forced_unk_hybrid_mass_ok_count = sum(
        1 for row in rows if row["comparison"]["mass_correctness_r6_forced_unk_hybrid_le_1e_6"]
    )

    by_scenario: dict[str, dict[str, float]] = {}
    scenario_names = sorted({str(row["scenario"]) for row in rows})
    for scenario_name in scenario_names:
        subset = [row for row in rows if row["scenario"] == scenario_name]
        ce_deltas = np.asarray([row["comparison"]["r6_minus_g6_ce"] for row in subset], dtype=np.float64)
        ce_deltas_unk_safe = np.asarray([row["comparison"]["r6_unk_safe_minus_g6_ce"] for row in subset], dtype=np.float64)
        ce_deltas_forced_unk = np.asarray([row["comparison"]["r6_forced_unk_minus_g6_ce"] for row in subset], dtype=np.float64)
        ce_deltas_forced_unk_kl_safe = np.asarray(
            [row["comparison"]["r6_forced_unk_kl_safe_minus_g6_ce"] for row in subset],
            dtype=np.float64,
        )
        ce_deltas_forced_unk_kl_safe_unk_context = np.asarray(
            [row["comparison"]["r6_forced_unk_kl_safe_unk_context_minus_g6_ce"] for row in subset],
            dtype=np.float64,
        )
        ce_deltas_forced_unk_hybrid = np.asarray(
            [row["comparison"]["r6_forced_unk_hybrid_minus_g6_ce"] for row in subset],
            dtype=np.float64,
        )
        by_scenario[scenario_name] = {
            "count": float(len(subset)),
            "r6_minus_g6_ce_mean": float(np.mean(ce_deltas)),
            "r6_minus_g6_ce_std": float(np.std(ce_deltas)),
            "competitive_rate": float(sum(1 for row in subset if row["comparison"]["r6_competitive_vs_g6"]) / max(len(subset), 1)),
            "beats_r4_rate": float(sum(1 for row in subset if row["comparison"]["r6_beats_r4"]) / max(len(subset), 1)),
            "r6_unk_safe_minus_g6_ce_mean": float(np.mean(ce_deltas_unk_safe)),
            "r6_unk_safe_minus_g6_ce_std": float(np.std(ce_deltas_unk_safe)),
            "r6_unk_safe_competitive_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_unk_safe_competitive_vs_g6"]) / max(len(subset), 1)
            ),
            "r6_unk_safe_beats_r4_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_unk_safe_beats_r4"]) / max(len(subset), 1)
            ),
            "r6_forced_unk_minus_g6_ce_mean": float(np.mean(ce_deltas_forced_unk)),
            "r6_forced_unk_minus_g6_ce_std": float(np.std(ce_deltas_forced_unk)),
            "r6_forced_unk_competitive_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_competitive_vs_g6"]) / max(len(subset), 1)
            ),
            "r6_forced_unk_beats_r4_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_beats_r4"]) / max(len(subset), 1)
            ),
            "r6_forced_unk_kl_safe_minus_g6_ce_mean": float(np.mean(ce_deltas_forced_unk_kl_safe)),
            "r6_forced_unk_kl_safe_minus_g6_ce_std": float(np.std(ce_deltas_forced_unk_kl_safe)),
            "r6_forced_unk_kl_safe_competitive_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_kl_safe_competitive_vs_g6"])
                / max(len(subset), 1)
            ),
            "r6_forced_unk_kl_safe_beats_r4_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_kl_safe_beats_r4"]) / max(len(subset), 1)
            ),
            "r6_forced_unk_kl_safe_unk_context_minus_g6_ce_mean": float(
                np.mean(ce_deltas_forced_unk_kl_safe_unk_context)
            ),
            "r6_forced_unk_kl_safe_unk_context_minus_g6_ce_std": float(
                np.std(ce_deltas_forced_unk_kl_safe_unk_context)
            ),
            "r6_forced_unk_kl_safe_unk_context_competitive_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_kl_safe_unk_context_competitive_vs_g6"])
                / max(len(subset), 1)
            ),
            "r6_forced_unk_kl_safe_unk_context_beats_r4_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_kl_safe_unk_context_beats_r4"])
                / max(len(subset), 1)
            ),
            "r6_forced_unk_hybrid_minus_g6_ce_mean": float(np.mean(ce_deltas_forced_unk_hybrid)),
            "r6_forced_unk_hybrid_minus_g6_ce_std": float(np.std(ce_deltas_forced_unk_hybrid)),
            "r6_forced_unk_hybrid_competitive_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_hybrid_competitive_vs_g6"]) / max(len(subset), 1)
            ),
            "r6_forced_unk_hybrid_beats_r4_rate": float(
                sum(1 for row in subset if row["comparison"]["r6_forced_unk_hybrid_beats_r4"]) / max(len(subset), 1)
            ),
        }

    summary = {
        "experiment_id": "g7_r6_plus_reproducibility",
        "winning_config_frozen": WINNING_CONFIG,
        "cpu_threading": CPU_THREADING,
        "seeds": seeds,
        "scenario_count": len(scenario_names),
        "total_seed_scenario_runs": total,
        "elapsed_seconds": elapsed,
        "aggregate": {
            "r6_competitive_vs_g6_count": r6_competitive_count,
            "r6_competitive_vs_g6_rate": float(r6_competitive_count / max(total, 1)),
            "r6_recovered_vs_g6_plus_0_1_count": r6_recovered_count,
            "r6_recovered_vs_g6_plus_0_1_rate": float(r6_recovered_count / max(total, 1)),
            "r6_beats_r4_count": r6_beats_r4_count,
            "r6_beats_r4_rate": float(r6_beats_r4_count / max(total, 1)),
            "mass_correctness_pass_count": mass_ok_count,
            "mass_correctness_pass_rate": float(mass_ok_count / max(total, 1)),
            "r6_unk_safe_competitive_vs_g6_count": r6_unk_safe_competitive_count,
            "r6_unk_safe_competitive_vs_g6_rate": float(r6_unk_safe_competitive_count / max(total, 1)),
            "r6_unk_safe_recovered_vs_g6_plus_0_1_count": r6_unk_safe_recovered_count,
            "r6_unk_safe_recovered_vs_g6_plus_0_1_rate": float(r6_unk_safe_recovered_count / max(total, 1)),
            "r6_unk_safe_beats_r4_count": r6_unk_safe_beats_r4_count,
            "r6_unk_safe_beats_r4_rate": float(r6_unk_safe_beats_r4_count / max(total, 1)),
            "r6_unk_safe_mass_correctness_pass_count": r6_unk_safe_mass_ok_count,
            "r6_unk_safe_mass_correctness_pass_rate": float(r6_unk_safe_mass_ok_count / max(total, 1)),
            "r6_forced_unk_competitive_vs_g6_count": r6_forced_unk_competitive_count,
            "r6_forced_unk_competitive_vs_g6_rate": float(r6_forced_unk_competitive_count / max(total, 1)),
            "r6_forced_unk_recovered_vs_g6_plus_0_1_count": r6_forced_unk_recovered_count,
            "r6_forced_unk_recovered_vs_g6_plus_0_1_rate": float(r6_forced_unk_recovered_count / max(total, 1)),
            "r6_forced_unk_beats_r4_count": r6_forced_unk_beats_r4_count,
            "r6_forced_unk_beats_r4_rate": float(r6_forced_unk_beats_r4_count / max(total, 1)),
            "r6_forced_unk_mass_correctness_pass_count": r6_forced_unk_mass_ok_count,
            "r6_forced_unk_mass_correctness_pass_rate": float(r6_forced_unk_mass_ok_count / max(total, 1)),
            "r6_forced_unk_kl_safe_competitive_vs_g6_count": r6_forced_unk_kl_safe_competitive_count,
            "r6_forced_unk_kl_safe_competitive_vs_g6_rate": float(r6_forced_unk_kl_safe_competitive_count / max(total, 1)),
            "r6_forced_unk_kl_safe_recovered_vs_g6_plus_0_1_count": r6_forced_unk_kl_safe_recovered_count,
            "r6_forced_unk_kl_safe_recovered_vs_g6_plus_0_1_rate": float(
                r6_forced_unk_kl_safe_recovered_count / max(total, 1)
            ),
            "r6_forced_unk_kl_safe_beats_r4_count": r6_forced_unk_kl_safe_beats_r4_count,
            "r6_forced_unk_kl_safe_beats_r4_rate": float(r6_forced_unk_kl_safe_beats_r4_count / max(total, 1)),
            "r6_forced_unk_kl_safe_mass_correctness_pass_count": r6_forced_unk_kl_safe_mass_ok_count,
            "r6_forced_unk_kl_safe_mass_correctness_pass_rate": float(
                r6_forced_unk_kl_safe_mass_ok_count / max(total, 1)
            ),
            "r6_forced_unk_kl_safe_unk_context_competitive_vs_g6_count": r6_forced_unk_kl_safe_unk_context_competitive_count,
            "r6_forced_unk_kl_safe_unk_context_competitive_vs_g6_rate": float(
                r6_forced_unk_kl_safe_unk_context_competitive_count / max(total, 1)
            ),
            "r6_forced_unk_kl_safe_unk_context_recovered_vs_g6_plus_0_1_count": r6_forced_unk_kl_safe_unk_context_recovered_count,
            "r6_forced_unk_kl_safe_unk_context_recovered_vs_g6_plus_0_1_rate": float(
                r6_forced_unk_kl_safe_unk_context_recovered_count / max(total, 1)
            ),
            "r6_forced_unk_kl_safe_unk_context_beats_r4_count": r6_forced_unk_kl_safe_unk_context_beats_r4_count,
            "r6_forced_unk_kl_safe_unk_context_beats_r4_rate": float(
                r6_forced_unk_kl_safe_unk_context_beats_r4_count / max(total, 1)
            ),
            "r6_forced_unk_kl_safe_unk_context_mass_correctness_pass_count": r6_forced_unk_kl_safe_unk_context_mass_ok_count,
            "r6_forced_unk_kl_safe_unk_context_mass_correctness_pass_rate": float(
                r6_forced_unk_kl_safe_unk_context_mass_ok_count / max(total, 1)
            ),
            "r6_forced_unk_hybrid_competitive_vs_g6_count": r6_forced_unk_hybrid_competitive_count,
            "r6_forced_unk_hybrid_competitive_vs_g6_rate": float(r6_forced_unk_hybrid_competitive_count / max(total, 1)),
            "r6_forced_unk_hybrid_recovered_vs_g6_plus_0_1_count": r6_forced_unk_hybrid_recovered_count,
            "r6_forced_unk_hybrid_recovered_vs_g6_plus_0_1_rate": float(
                r6_forced_unk_hybrid_recovered_count / max(total, 1)
            ),
            "r6_forced_unk_hybrid_beats_r4_count": r6_forced_unk_hybrid_beats_r4_count,
            "r6_forced_unk_hybrid_beats_r4_rate": float(r6_forced_unk_hybrid_beats_r4_count / max(total, 1)),
            "r6_forced_unk_hybrid_mass_correctness_pass_count": r6_forced_unk_hybrid_mass_ok_count,
            "r6_forced_unk_hybrid_mass_correctness_pass_rate": float(
                r6_forced_unk_hybrid_mass_ok_count / max(total, 1)
            ),
        },
        "by_scenario": by_scenario,
        "rows": rows,
    }

    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / "g7_r6_reproducibility_result.json"
    summary_path = results_dir / "g7_r6_reproducibility_summary.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r6_reproducibility_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_r6_reproducibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("g7_r6 reproducibility complete")
    print(json.dumps({k: summary[k] for k in ("experiment_id", "winning_config_frozen", "aggregate", "by_scenario")}, indent=2))


if __name__ == "__main__":
    main()
