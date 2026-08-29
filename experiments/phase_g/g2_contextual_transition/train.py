from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import tracemalloc

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.datasets import BASE_CORPUS, CONTEXT_AMBIGUITY_CORPUS, build_train_eval_split  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import build_stupid_tokenizer_from_lines  # noqa: E402
from g2_contextual_transition.contextual_transition_model import ContextualTransitionModel  # noqa: E402


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G2 contextual transition model.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.3)
    args = parser.parse_args()

    corpus = BASE_CORPUS + CONTEXT_AMBIGUITY_CORPUS
    train_lines, eval_lines = build_train_eval_split(corpus, seed=args.seed, eval_ratio=args.eval_ratio)
    tokenizer = build_stupid_tokenizer_from_lines(corpus)
    train_sequences = _encode_lines(tokenizer, train_lines)
    eval_sequences = _encode_lines(tokenizer, eval_lines)

    tracemalloc.start()
    start = time.perf_counter()
    g1 = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=1, alpha=args.alpha)
    g2 = ContextualTransitionModel(vocab_size=len(tokenizer.token_to_id), alpha=args.alpha)
    g1_transitions = g1.fit(train_sequences)
    g2_transitions = g2.fit(train_sequences)
    training_time = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(artifact_dir / "tokenizer.json")
    g1.save(artifact_dir / "g1_baseline.pkl")
    g2.save(artifact_dir / "g2_model.pkl")
    split_payload = {"seed": args.seed, "train_lines": train_lines, "eval_lines": eval_lines}
    (artifact_dir / "split.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    result = {
        "experiment_id": "g2_contextual_transition_train",
        "seed": args.seed,
        "alpha": args.alpha,
        "corpus_size": len(corpus),
        "train_size": len(train_lines),
        "eval_size": len(eval_lines),
        "vocab_size": len(tokenizer.token_to_id),
        "g1_transition_count": g1_transitions,
        "g2_transition_count": g2_transitions,
        "training_time": training_time,
        "peak_process_ram": int(peak),
        "g1_parameter_count": g1.parameter_count,
        "g2_parameter_count": g2.parameter_count,
        "g1_nonzero_parameters": g1.nonzero_parameters,
        "g2_nonzero_parameters": g2.nonzero_parameters,
        "g1_model_storage_bytes": g1.model_storage_bytes,
        "g2_model_storage_bytes": g2.model_storage_bytes,
    }
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "g2_train_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("G2 training complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
