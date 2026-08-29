from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402
from g2_contextual_transition.contextual_transition_model import ContextualTransitionModel  # noqa: E402


def _random_baseline(sequences: list[list[int]], vocab_size: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for sequence in sequences:
        for pos in range(len(sequence) - 1):
            one_hot = np.zeros(vocab_size, dtype=np.float64)
            one_hot[int(rng.integers(0, vocab_size))] = 1.0
            rows.append(one_hot)
            targets.append(int(sequence[pos + 1]))
    metrics = evaluate_predictions(rows, targets)
    metrics["cross_entropy"] = float(math.log(vocab_size))
    metrics["perplexity"] = float(vocab_size)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate G2 against G1 baseline and random.")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    tokenizer = StupidTokenizer.load(artifact_dir / "tokenizer.json")
    g1 = SparseNGramModel.load(artifact_dir / "g1_baseline.pkl")
    g2 = ContextualTransitionModel.load(artifact_dir / "g2_model.pkl")
    split_payload = json.loads((artifact_dir / "split.json").read_text(encoding="utf-8"))
    eval_lines = split_payload["eval_lines"]
    eval_sequences = [tokenizer.encode(line, add_bos=True, add_eos=True) for line in eval_lines]

    tracemalloc.start()
    start = time.perf_counter()
    random_metrics = _random_baseline(eval_sequences, len(tokenizer.token_to_id), seed=args.seed)
    g1_rows, g1_targets = g1.iter_probability_rows(eval_sequences)
    g2_rows, g2_targets = g2.iter_probability_rows(eval_sequences)
    g1_metrics = evaluate_predictions(g1_rows, g1_targets)
    g2_metrics = evaluate_predictions(g2_rows, g2_targets)
    inference_time = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "experiment_id": "g2_contextual_transition",
        "seed": args.seed,
        "corpus_size": len(split_payload["train_lines"]) + len(split_payload["eval_lines"]),
        "vocab_size": len(tokenizer.token_to_id),
        "parameter_count": g2.parameter_count,
        "nonzero_parameters": g2.nonzero_parameters,
        "transition_count": int(g2_metrics["count"]),
        "context_count": len(g2.counts),
        "training_time": json.loads((Path(__file__).resolve().parent / "results" / "g2_train_result.json").read_text(encoding="utf-8"))["training_time"],
        "inference_time": inference_time,
        "tokens_per_second": g2_metrics["count"] / max(inference_time, 1e-12),
        "model_storage_bytes": g2.model_storage_bytes,
        "peak_process_ram": int(peak),
        "random": {k: random_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "g1": {k: g1_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "g2": {k: g2_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
    }

    results_dir = Path(__file__).resolve().parent / "results"
    (results_dir / "g2_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    global_results = PHASE_G_ROOT / "results" / "g2_result.json"
    global_results.parent.mkdir(parents=True, exist_ok=True)
    global_results.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("G2 evaluation complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
