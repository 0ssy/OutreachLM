from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc

from tokenizer import StupidTokenizer, build_stupid_tokenizer_from_file
from transition_model import TransitionModel


DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.txt"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load_lines(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Corpus is empty: {path}")
    return lines


def _encode_lines(tokenizer: StupidTokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G1 transition model from scratch (CPU + NumPy).")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment-id", type=str, default="g1_transition")
    args = parser.parse_args()

    lines = _load_lines(args.corpus_path)
    tokenizer = build_stupid_tokenizer_from_file(args.corpus_path)
    token_sequences = _encode_lines(tokenizer, lines)

    tracemalloc.start()
    start = time.perf_counter()
    model = TransitionModel(vocab_size=len(tokenizer.token_to_id), alpha=args.alpha)
    transition_count = model.fit(token_sequences)
    training_time = time.perf_counter() - start
    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.artifact_dir / "tokenizer.json"
    model_path = args.artifact_dir / "transition_model.npz"
    tokenizer.save(tokenizer_path)
    model.save(model_path)

    result = {
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "corpus_size": len(lines),
        "vocab_size": len(tokenizer.token_to_id),
        "parameter_count": len(tokenizer.token_to_id) * len(tokenizer.token_to_id),
        "training_time": training_time,
        "inference_time": 0.0,
        "tokens_per_second": transition_count / max(training_time, 1e-12),
        "ram_peak": int(ram_peak),
        "accuracy": None,
        "cross_entropy": None,
        "perplexity": None,
        "alpha": args.alpha,
        "transition_count": transition_count,
        "artifacts": {
            "tokenizer": str(tokenizer_path.resolve()),
            "model": str(model_path.resolve()),
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.results_dir / f"{args.experiment_id}_train_seed{args.seed}.json"
    run_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("G1 training complete")
    print(f"Corpus lines: {len(lines)}")
    print(f"Vocab size: {len(tokenizer.token_to_id)}")
    print(f"Transitions: {transition_count}")
    print(f"Training time (s): {training_time:.6f}")
    print(f"Tokens/sec: {result['tokens_per_second']:.2f}")
    print(f"RAM peak (bytes): {ram_peak}")
    print(f"Tokenizer artifact: {tokenizer_path}")
    print(f"Model artifact: {model_path}")
    print(f"Run record: {run_path}")


if __name__ == "__main__":
    main()
