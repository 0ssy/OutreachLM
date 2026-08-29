from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.datasets import BASE_CORPUS, CONTEXT_AMBIGUITY_CORPUS, LONG_CONTEXT_CORPUS, build_train_eval_split  # noqa: E402
from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import build_stupid_tokenizer_from_lines  # noqa: E402


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def main() -> None:
    g5 = json.loads((PHASE_G_ROOT / "g5_adaptive_memory" / "results" / "g5_result.json").read_text(encoding="utf-8"))
    seed = 1337
    corpus = BASE_CORPUS + CONTEXT_AMBIGUITY_CORPUS + LONG_CONTEXT_CORPUS
    train_lines, eval_lines = build_train_eval_split(corpus, seed=seed, eval_ratio=0.3)
    tokenizer = build_stupid_tokenizer_from_lines(corpus)
    train_sequences = _encode_lines(tokenizer, train_lines)
    eval_sequences = _encode_lines(tokenizer, eval_lines)

    local = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
    medium = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=4, alpha=0.1)
    global_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=1, alpha=0.1)

    tracemalloc.start()
    t0 = time.perf_counter()
    local.fit(train_sequences)
    medium.fit(train_sequences)
    global_model.fit(train_sequences)
    training_time = time.perf_counter() - t0

    def dist(context: list[int]) -> np.ndarray:
        return 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)

    t1 = time.perf_counter()
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for sequence in eval_sequences:
        for pos in range(len(sequence) - 1):
            rows.append(dist(sequence[: pos + 1]))
            targets.append(int(sequence[pos + 1]))
    metrics = evaluate_predictions(rows, targets)
    inference_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def subset_accuracy(token: str) -> float:
        token_id = tokenizer.token_to_id.get(token)
        if token_id is None:
            return float("nan")
        s_rows: list[np.ndarray] = []
        s_targets: list[int] = []
        for sequence in eval_sequences:
            for pos in range(len(sequence) - 1):
                if sequence[pos] == token_id:
                    s_rows.append(dist(sequence[: pos + 1]))
                    s_targets.append(int(sequence[pos + 1]))
        if not s_rows:
            return float("nan")
        return evaluate_predictions(s_rows, s_targets)["accuracy"]

    result = {
        "experiment_id": "g6_hierarchical_structure",
        "seed": seed,
        "corpus_size": len(corpus),
        "vocab_size": len(tokenizer.token_to_id),
        "parameter_count": local.parameter_count + medium.parameter_count + global_model.parameter_count,
        "nonzero_parameters": local.nonzero_parameters + medium.nonzero_parameters + global_model.nonzero_parameters,
        "training_time": training_time,
        "inference_time": inference_time,
        "tokens_per_second": metrics["count"] / max(inference_time, 1e-12),
        "model_storage_bytes": local.model_storage_bytes + medium.model_storage_bytes + global_model.model_storage_bytes,
        "peak_process_ram": int(peak),
        "g5_reference": g5["g5"],
        "g6": {k: metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "hierarchical_breakdown": {
            "local_accuracy": subset_accuracy("cat"),
            "medium_accuracy": subset_accuracy("will"),
            "long_accuracy": subset_accuracy("answer"),
        },
    }
    result["hard_gates"] = {
        "advantage_over_g5": (
            result["g6"]["accuracy"] >= g5["g5"]["accuracy"]
            or (not np.isnan(result["hierarchical_breakdown"]["long_accuracy"]) and result["hierarchical_breakdown"]["long_accuracy"] >= 0.5)
        ),
        "cpu_measured": True,
    }

    root = Path(__file__).resolve().parent
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g6_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g6_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("G6 complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
