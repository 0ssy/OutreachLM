from __future__ import annotations

import json
from pathlib import Path
import sys
import time

PHASE_G_ROOT = Path(__file__).resolve().parents[2]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.cpu_threads import configure_cpu_threads_from_env  # noqa: E402
from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402
from run import (  # noqa: E402
    _g2_distribution,
    _g6_distribution,
    _r4_baseline,
    _r6_residual_sparse_contextual_plus_forced_unk,
    _r6_residual_sparse_contextual_plus_forced_unk_hybrid,
)

CPU_THREADING = configure_cpu_threads_from_env()

import numpy as np


def _load_fixture() -> tuple[dict, StupidTokenizer, list[list[int]], list[list[int]]]:
    fixture_path = PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    tokenizer = StupidTokenizer(token_to_id=fixture["token_to_id"])
    train_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["train_lines"]]
    eval_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in fixture["eval_lines"]]
    return fixture, tokenizer, train_sequences, eval_sequences


def _entropy(probabilities: np.ndarray, epsilon: float = 1e-12) -> float:
    p = np.clip(probabilities, epsilon, 1.0)
    return float(-(p * np.log(p)).sum())


def _kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _build_rows(
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
            g2_rows.append(_g2_distribution(g2_model, context))
    return g6_rows, g2_rows, targets, contexts


def _summary(name: str, predictions: list[np.ndarray], g6_rows: list[np.ndarray], targets: list[int]) -> dict[str, float]:
    metrics = evaluate_predictions(predictions, targets)
    mass_errors = [abs(float(row.sum()) - 1.0) for row in predictions]
    kls = [_kl_divergence(g6, pred) for g6, pred in zip(g6_rows, predictions)]
    entropies = [_entropy(row) for row in predictions]
    return {
        "name": name,
        "accuracy": float(metrics["accuracy"]),
        "cross_entropy": float(metrics["cross_entropy"]),
        "perplexity": float(metrics["perplexity"]),
        "kl_g6_mean": float(np.mean(np.asarray(kls, dtype=np.float64))),
        "entropy_mean": float(np.mean(np.asarray(entropies, dtype=np.float64))),
        "mass_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
    }


def _benchmark(
    g6_rows: list[np.ndarray],
    g2_rows: list[np.ndarray],
    contexts: list[list[int]],
    unigram: np.ndarray,
    unk_id: int,
    repeats: int = 5,
) -> dict[str, dict[str, float]]:
    models = {
        "g6": lambda i: g6_rows[i],
        "r4": lambda i: _r4_baseline(g6_rows[i], g2_rows[i], top_k=4, floor_mix=0.2),
        "g7_final_hybrid": lambda i: _r6_residual_sparse_contextual_plus_forced_unk_hybrid(
            g6=g6_rows[i],
            top_k=4,
            g2=g2_rows[i],
            unigram_distribution=unigram,
            context=contexts[i],
            unk_id=unk_id,
            unk_probability_threshold=0.02,
        ),
    }
    output: dict[str, dict[str, float]] = {}
    for name, fn in models.items():
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = [fn(i) for i in range(len(g6_rows))]
            elapsed.append(time.perf_counter() - start)
        mean_elapsed = float(np.mean(np.asarray(elapsed, dtype=np.float64)))
        output[name] = {
            "mean_elapsed_seconds": mean_elapsed,
            "tokens_per_second": float(len(g6_rows) / max(mean_elapsed, 1e-12)),
        }
    return output


def main() -> None:
    fixture, tokenizer, train_sequences, eval_sequences = _load_fixture()
    reproducibility_path = Path(__file__).resolve().parent / "results" / "g7_r6_reproducibility_summary.json"
    reproducibility = json.loads(reproducibility_path.read_text(encoding="utf-8"))

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

    g6_rows, g2_rows, targets, contexts = _build_rows(eval_sequences, g2_model, local, medium, global_model)

    g6_preds = [row.copy() for row in g6_rows]
    r4_preds = [_r4_baseline(g6, g2, top_k=4, floor_mix=0.2) for g6, g2 in zip(g6_rows, g2_rows)]
    g7_final_preds = [
        _r6_residual_sparse_contextual_plus_forced_unk_hybrid(
            g6=g6,
            top_k=4,
            g2=g2,
            unigram_distribution=unigram,
            context=context,
            unk_id=tokenizer.unk_id,
            unk_probability_threshold=0.02,
        )
        for g6, g2, context in zip(g6_rows, g2_rows, contexts)
    ]
    ablation_no_residual = [
        _r6_residual_sparse_contextual_plus_forced_unk(
            g6=g6,
            top_k=4,
            g2=g2,
            unigram_distribution=unigram,
            unk_id=tokenizer.unk_id,
            context_blend=0.9,
            tail_temperature=0.75,
            head_temperature=1.1,
            head_context_mix=0.0,
            residual_gain=0.0,
        )
        for g6, g2 in zip(g6_rows, g2_rows)
    ]
    ablation_no_hybrid = [
        _r6_residual_sparse_contextual_plus_forced_unk(
            g6=g6,
            top_k=4,
            g2=g2,
            unigram_distribution=unigram,
            unk_id=tokenizer.unk_id,
            context_blend=0.9,
            tail_temperature=0.75,
            head_temperature=1.1,
            head_context_mix=0.0,
            residual_gain=0.8,
        )
        for g6, g2 in zip(g6_rows, g2_rows)
    ]

    g6_summary = _summary("g6", g6_preds, g6_rows, targets)
    r4_summary = _summary("r4", r4_preds, g6_rows, targets)
    g7_final_summary = _summary("g7_final_hybrid", g7_final_preds, g6_rows, targets)
    ablation_no_residual_summary = _summary("ablation_no_residual", ablation_no_residual, g6_rows, targets)
    ablation_no_hybrid_summary = _summary("ablation_no_hybrid", ablation_no_hybrid, g6_rows, targets)
    cpu_summary = _benchmark(g6_rows, g2_rows, contexts, unigram, tokenizer.unk_id)

    profile = {
        "mechanism_name": "g7_final_hybrid",
        "description": "sparse compute + dense probability reconstruction + contextual residual correction + forced-UNK hybrid gating",
        "default_profile": {
            "top_k": 4,
            "context_blend": 0.9,
            "tail_temperature": 0.75,
            "head_temperature": 1.1,
            "head_context_mix": 0.0,
            "residual_gain": 0.8,
        },
        "oov_profile": {
            "top_k": 4,
            "context_blend": 0.9,
            "tail_temperature": 0.9,
            "head_temperature": 0.8,
            "head_context_mix": 0.0,
            "residual_gain": 0.6,
        },
        "hybrid_gate": {
            "trigger_if_unk_in_context": True,
            "trigger_if_g6_unk_probability_gte": 0.02,
        },
        "operations_per_token_estimate": 8.0,
    }

    protocol = {
        "seed_values": [1337, 2024, 4242, 9001, 12345],
        "scenarios": [
            "baseline_split",
            "larger_eval_corpus",
            "context_long_heavy",
            "context_ambiguity_heavy",
            "unseen_transitions",
            "unseen_tokens",
        ],
        "total_runs_expected": 30,
        "required_checks": [
            "competitive_vs_g6",
            "recovered_vs_g6_plus_0.1",
            "mass_correctness",
            "beats_r4",
            "cpu_benchmark",
            "variance_bootstrap",
        ],
    }

    aggregate = reproducibility["aggregate"]
    by_scenario = reproducibility["by_scenario"]

    gates = {
        "reproducibility_30_of_30_competitive_vs_g6": bool(aggregate["r6_forced_unk_hybrid_competitive_vs_g6_count"] == 30),
        "reproducibility_30_of_30_recovered": bool(aggregate["r6_forced_unk_hybrid_recovered_vs_g6_plus_0_1_count"] == 30),
        "unseen_tokens_5_of_5_competitive": bool(by_scenario["unseen_tokens"]["r6_forced_unk_hybrid_competitive_rate"] == 1.0),
        "probability_mass_correct_30_of_30": bool(aggregate["r6_forced_unk_hybrid_mass_correctness_pass_count"] == 30),
        "frozen_fixture_ce_competitive_vs_g6": bool(g7_final_summary["cross_entropy"] <= g6_summary["cross_entropy"]),
        "frozen_fixture_beats_r4": bool(g7_final_summary["cross_entropy"] <= r4_summary["cross_entropy"]),
        "residual_ablation_matters": bool(ablation_no_residual_summary["cross_entropy"] > g7_final_summary["cross_entropy"]),
    }
    accepted = bool(all(gates.values()))

    result = {
        "experiment_id": "g7_final_validation_package",
        "cpu_threading": CPU_THREADING,
        "frozen_fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "mechanism_profile": profile,
        "reproducibility_protocol": protocol,
        "frozen_fixture_metrics": {
            "g6": g6_summary,
            "r4": r4_summary,
            "g7_final_hybrid": g7_final_summary,
        },
        "g6_vs_g7_final": {
            "delta_ce_g7_minus_g6": float(g7_final_summary["cross_entropy"] - g6_summary["cross_entropy"]),
            "delta_ce_g7_minus_r4": float(g7_final_summary["cross_entropy"] - r4_summary["cross_entropy"]),
        },
        "ablation": {
            "ablation_no_residual": ablation_no_residual_summary,
            "ablation_no_hybrid": ablation_no_hybrid_summary,
            "delta_ce_no_residual_minus_final": float(
                ablation_no_residual_summary["cross_entropy"] - g7_final_summary["cross_entropy"]
            ),
            "delta_ce_no_hybrid_minus_final": float(
                ablation_no_hybrid_summary["cross_entropy"] - g7_final_summary["cross_entropy"]
            ),
        },
        "cpu_benchmark_frozen_fixture": cpu_summary,
        "reproducibility_aggregate": aggregate,
        "reproducibility_by_scenario": by_scenario,
        "acceptance_gates": gates,
        "final_decision": "ACCEPTED" if accepted else "REJECTED",
    }

    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "g7_final_mechanism_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    (results_dir / "g7_final_reproducibility_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (results_dir / "g7_final_validation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_final_mechanism_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_final_reproducibility_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g7_final_validation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("g7 final validation complete")
    print(json.dumps({"final_decision": result["final_decision"], "acceptance_gates": gates}, indent=2))


if __name__ == "__main__":
    main()
