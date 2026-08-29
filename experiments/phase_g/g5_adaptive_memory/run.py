from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tracemalloc

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.datasets import CONTEXT_AMBIGUITY_CORPUS, LONG_CONTEXT_CORPUS, build_train_eval_split  # noqa: E402
from common.metrics import evaluate_predictions  # noqa: E402
from common.phase_components import AdaptiveMemoryModel  # noqa: E402
from common.tokenizer import build_stupid_tokenizer_from_lines  # noqa: E402


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def main() -> None:
    g4 = json.loads((PHASE_G_ROOT / "g4_context_compression" / "results" / "g4_result.json").read_text(encoding="utf-8"))
    seed = 1337
    corpus = CONTEXT_AMBIGUITY_CORPUS + LONG_CONTEXT_CORPUS
    train_lines, eval_lines = build_train_eval_split(corpus, seed=seed, eval_ratio=0.35)
    tokenizer = build_stupid_tokenizer_from_lines(corpus)
    train_sequences = _encode_lines(tokenizer, train_lines)
    eval_sequences = _encode_lines(tokenizer, eval_lines)

    tracemalloc.start()
    t0 = time.perf_counter()
    model = AdaptiveMemoryModel(vocab_size=len(tokenizer.token_to_id), memory_size=3, window_size=6, alpha=0.1)
    model.fit(train_sequences)
    training_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    rows, targets = model.iter_probability_rows(eval_sequences)
    metrics = evaluate_predictions(rows, targets)
    inference_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "experiment_id": "g5_adaptive_memory",
        "seed": seed,
        "corpus_size": len(corpus),
        "vocab_size": len(tokenizer.token_to_id),
        "memory_size": 3,
        "window_size": 6,
        "parameter_count": len(model.counts) * len(tokenizer.token_to_id),
        "nonzero_parameters": int(sum((row != 0).sum() for row in model.counts.values())),
        "training_time": training_time,
        "inference_time": inference_time,
        "tokens_per_second": metrics["count"] / max(inference_time, 1e-12),
        "model_storage_bytes": model.model_storage_bytes,
        "peak_process_ram": int(peak),
        "g4_reference": g4["g4"],
        "g5": {k: metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "hard_gates": {
            "benefit_over_g4": (
                metrics["accuracy"] > g4["g4"]["accuracy"]
                or model.model_storage_bytes < g4["model_storage_bytes"]
            ),
            "cpu_measured": True,
        },
    }

    root = Path(__file__).resolve().parent
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g5_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g5_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("G5 complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
