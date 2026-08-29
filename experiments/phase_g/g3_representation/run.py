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

from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def _cooccurrence_embeddings(sequences: list[list[int]], vocab_size: int, window: int = 2) -> np.ndarray:
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for sequence in sequences:
        for i, token in enumerate(sequence):
            start = max(0, i - window)
            end = min(len(sequence), i + window + 1)
            for j in range(start, end):
                if i == j:
                    continue
                matrix[token, sequence[j]] += 1.0
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    g2_artifact_dir = PHASE_G_ROOT / "g2_contextual_transition" / "artifacts"
    tokenizer = StupidTokenizer.load(g2_artifact_dir / "tokenizer.json")
    split = json.loads((g2_artifact_dir / "split.json").read_text(encoding="utf-8"))
    train_sequences = _encode_lines(tokenizer, split["train_lines"])
    eval_sequences = _encode_lines(tokenizer, split["eval_lines"])

    tracemalloc.start()
    t0 = time.perf_counter()
    base_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
    base_model.fit(train_sequences)
    embeddings = _cooccurrence_embeddings(train_sequences, len(tokenizer.token_to_id))
    training_time = time.perf_counter() - t0

    def dist(context: list[int]) -> np.ndarray:
        base = base_model.distribution(context)
        if len(context) < 2:
            return base
        v = embeddings[int(context[-2])] + embeddings[int(context[-1])]
        sims = embeddings @ v
        sims = np.maximum(sims, 0.0)
        if sims.sum() <= 0.0:
            return base
        smooth = (sims / sims.sum()) * 0.3
        return (0.7 * base) + smooth

    t1 = time.perf_counter()
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for sequence in eval_sequences:
        for pos in range(len(sequence) - 1):
            context = sequence[max(0, pos - 1) : pos + 1]
            rows.append(dist(context))
            targets.append(int(sequence[pos + 1]))
    metrics = evaluate_predictions(rows, targets)
    inference_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    token_to_id = tokenizer.token_to_id
    cat = embeddings[token_to_id.get("cat", 0)]
    dog = embeddings[token_to_id.get("dog", 0)]
    fish = embeddings[token_to_id.get("fish", 0)]

    g2_metrics = json.loads((PHASE_G_ROOT / "g2_contextual_transition" / "results" / "g2_result.json").read_text(encoding="utf-8"))["g2"]

    result = {
        "experiment_id": "g3_distributed_representation",
        "seed": split["seed"],
        "corpus_size": len(split["train_lines"]) + len(split["eval_lines"]),
        "vocab_size": len(tokenizer.token_to_id),
        "parameter_count": int(base_model.parameter_count + embeddings.size),
        "nonzero_parameters": int(base_model.nonzero_parameters + np.count_nonzero(embeddings)),
        "training_time": training_time,
        "inference_time": inference_time,
        "tokens_per_second": metrics["count"] / max(inference_time, 1e-12),
        "model_storage_bytes": int(base_model.model_storage_bytes + embeddings.nbytes),
        "peak_process_ram": int(peak),
        "g2_reference": g2_metrics,
        "g3": {k: metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "representation_similarity": {
            "cosine_cat_dog": _cosine(cat, dog),
            "cosine_cat_fish": _cosine(cat, fish),
        },
        "hard_gates": {
            "representations_learned": bool(np.count_nonzero(embeddings) > 0),
            "contextual_information": bool(_cosine(cat, dog) >= _cosine(cat, fish)),
            "competitive_with_g2": bool(metrics["cross_entropy"] <= g2_metrics["cross_entropy"] * 1.2),
            "no_pretrained": True,
            "cpu_measured": True,
        },
    }

    root = Path(__file__).resolve().parent
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    np.save(root / "artifacts" / "embeddings.npy", embeddings)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g3_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g3_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("G3 complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
