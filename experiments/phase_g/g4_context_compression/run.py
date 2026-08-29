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
from common.models import SparseNGramModel  # noqa: E402
from common.phase_components import CompressedContextModel  # noqa: E402
from common.tokenizer import build_stupid_tokenizer_from_lines  # noqa: E402


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def main() -> None:
    seed = 1337
    corpus = CONTEXT_AMBIGUITY_CORPUS + LONG_CONTEXT_CORPUS
    train_lines, eval_lines = build_train_eval_split(corpus, seed=seed, eval_ratio=0.35)
    tokenizer = build_stupid_tokenizer_from_lines(corpus)
    train_sequences = _encode_lines(tokenizer, train_lines)
    eval_sequences = _encode_lines(tokenizer, eval_lines)

    tracemalloc.start()
    t0 = time.perf_counter()
    compressed = CompressedContextModel(vocab_size=len(tokenizer.token_to_id), context_length=4, bucket_count=32)
    dense_reference = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=4, alpha=0.1)
    compressed.fit(train_sequences)
    dense_reference.fit(train_sequences)
    training_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    c_rows, c_targets = compressed.iter_probability_rows(eval_sequences)
    d_rows, d_targets = dense_reference.iter_probability_rows(eval_sequences)
    c_metrics = evaluate_predictions(c_rows, c_targets)
    d_metrics = evaluate_predictions(d_rows, d_targets)
    inference_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    long_sequences = _encode_lines(tokenizer, LONG_CONTEXT_CORPUS)
    long_rows, long_targets = compressed.iter_probability_rows(long_sequences)
    long_metrics = evaluate_predictions(long_rows, long_targets)

    result = {
        "experiment_id": "g4_context_compression",
        "seed": seed,
        "corpus_size": len(corpus),
        "vocab_size": len(tokenizer.token_to_id),
        "context_length": 4,
        "compressed_state_buckets": 32,
        "parameter_count": len(compressed.counts) * len(tokenizer.token_to_id),
        "nonzero_parameters": int(sum((row != 0).sum() for row in compressed.counts.values())),
        "training_time": training_time,
        "inference_time": inference_time,
        "tokens_per_second": c_metrics["count"] / max(inference_time, 1e-12),
        "model_storage_bytes": compressed.model_storage_bytes,
        "explicit_context_storage_bytes": dense_reference.model_storage_bytes,
        "peak_process_ram": int(peak),
        "g4": {k: c_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "order4_reference": {k: d_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "long_context": {k: long_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "hard_gates": {
            "memory_reduced": compressed.model_storage_bytes < dense_reference.model_storage_bytes,
            "retention_noncatastrophic": c_metrics["cross_entropy"] <= d_metrics["cross_entropy"] + 1.0,
            "cpu_practical": (c_metrics["count"] / max(inference_time, 1e-12)) > 0.0,
        },
    }

    root = Path(__file__).resolve().parent
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g4_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g4_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("G4 complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
