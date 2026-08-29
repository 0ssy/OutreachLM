from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from tokenizer import build_stupid_tokenizer_from_lines
from transition_model import TransitionModel


AMBIGUOUS_CORPUS = [
    "the cat will eat",
    "the dog will sleep",
    "the cat will purr",
    "the dog will bark",
    "the cat eats fish",
    "the dog eats food",
    "the cat sees birds",
    "the dog sees people",
]


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "g1_context_failure.json"


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def _top_k_distribution(distribution: np.ndarray, id_to_token: list[str], k: int = 5) -> list[dict[str, float]]:
    top_indices = np.argsort(distribution)[::-1][:k]
    return [
        {
            "token": id_to_token[int(index)],
            "probability": float(distribution[int(index)]),
        }
        for index in top_indices
    ]


def _distribution_for_context(model: TransitionModel, tokenizer, context_text: str) -> tuple[int, np.ndarray]:
    context_ids = tokenizer.encode(context_text, add_bos=False, add_eos=False)
    if not context_ids:
        raise ValueError(f"Context is empty after tokenization: {context_text!r}")
    final_token_id = context_ids[-1]
    return final_token_id, model.predict_next_distribution(final_token_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="G1.2 context failure characterization for the transition model.")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--experiment-id", type=str, default="g1_context_failure")
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()

    tokenizer = build_stupid_tokenizer_from_lines(AMBIGUOUS_CORPUS)
    token_sequences = _encode_lines(tokenizer, AMBIGUOUS_CORPUS)
    model = TransitionModel(vocab_size=len(tokenizer.token_to_id), alpha=args.alpha)
    transition_count = model.fit(token_sequences)

    context_a = "the cat will"
    context_b = "the dog will"
    context_c = "the cat eats"
    context_d = "the dog sees"

    final_token_a, dist_a = _distribution_for_context(model, tokenizer, context_a)
    final_token_b, dist_b = _distribution_for_context(model, tokenizer, context_b)
    final_token_c, dist_c = _distribution_for_context(model, tokenizer, context_c)
    final_token_d, dist_d = _distribution_for_context(model, tokenizer, context_d)

    same_token_max_abs_diff = float(np.max(np.abs(dist_a - dist_b)))
    different_token_max_abs_diff_cd = float(np.max(np.abs(dist_c - dist_d)))
    different_token_max_abs_diff_ac = float(np.max(np.abs(dist_a - dist_c)))

    elapsed = time.perf_counter() - start
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    model_storage_bytes = int(model.counts.nbytes + np.array([model.alpha], dtype=np.float64).nbytes)
    process_peak_ram_bytes = int(peak_ram)

    id_to_token = tokenizer.id_to_token
    result = {
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "alpha": args.alpha,
        "corpus_size": len(AMBIGUOUS_CORPUS),
        "vocab_size": len(tokenizer.token_to_id),
        "parameter_count": int(model.counts.size),
        "transition_count": int(transition_count),
        "training_and_test_time_seconds": float(elapsed),
        "model_storage_bytes": model_storage_bytes,
        "process_peak_ram_bytes": process_peak_ram_bytes,
        "contexts": {
            "a": context_a,
            "b": context_b,
            "c": context_c,
            "d": context_d,
        },
        "final_tokens": {
            "a": id_to_token[final_token_a],
            "b": id_to_token[final_token_b],
            "c": id_to_token[final_token_c],
            "d": id_to_token[final_token_d],
        },
        "distribution_comparisons": {
            "same_final_token_a_vs_b_max_abs_diff": same_token_max_abs_diff,
            "different_final_token_c_vs_d_max_abs_diff": different_token_max_abs_diff_cd,
            "different_final_token_a_vs_c_max_abs_diff": different_token_max_abs_diff_ac,
        },
        "hard_gate": {
            "same_final_token_distributions_identical": bool(same_token_max_abs_diff == 0.0),
            "same_final_token_expected_max_abs_diff": 0.0,
            "same_final_token_observed_max_abs_diff": same_token_max_abs_diff,
        },
        "top_predictions": {
            "a": _top_k_distribution(dist_a, id_to_token, k=5),
            "b": _top_k_distribution(dist_b, id_to_token, k=5),
            "c": _top_k_distribution(dist_c, id_to_token, k=5),
            "d": _top_k_distribution(dist_d, id_to_token, k=5),
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("G1.2 context failure characterization complete")
    print(f"Output: {args.output_path}")
    print(f"Same-final-token max abs diff (expected 0): {same_token_max_abs_diff:.12f}")
    print(f"Different-final-token max abs diff C vs D: {different_token_max_abs_diff_cd:.12f}")
    print(f"Model storage (bytes): {model_storage_bytes}")
    print(f"Process peak RAM (bytes): {process_peak_ram_bytes}")


if __name__ == "__main__":
    main()
