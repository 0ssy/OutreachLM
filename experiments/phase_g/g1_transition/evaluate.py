from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import tracemalloc

from tokenizer import StupidTokenizer
from transition_model import TransitionModel, evaluate_random_baseline


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
    parser = argparse.ArgumentParser(description="Evaluate G1 transition model and random baseline.")
    parser.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment-id", type=str, default="g1_transition")
    args = parser.parse_args()

    tokenizer = StupidTokenizer.load(args.artifact_dir / "tokenizer.json")
    model = TransitionModel.load(args.artifact_dir / "transition_model.npz")
    lines = _load_lines(args.corpus_path)
    token_sequences = _encode_lines(tokenizer, lines)

    tracemalloc.start()
    start = time.perf_counter()
    random_metrics = evaluate_random_baseline(
        token_sequences,
        vocab_size=len(tokenizer.token_to_id),
        seed=args.seed,
    )
    g1_metrics = model.evaluate(token_sequences)
    inference_time = time.perf_counter() - start
    _, ram_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    transition_count = int(g1_metrics["transition_count"])
    tokens_per_second = transition_count / max(inference_time, 1e-12)
    parameter_count = len(tokenizer.token_to_id) * len(tokenizer.token_to_id)

    record = {
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "corpus_size": len(lines),
        "vocab_size": len(tokenizer.token_to_id),
        "parameter_count": parameter_count,
        "training_time": None,
        "inference_time": inference_time,
        "tokens_per_second": tokens_per_second,
        "ram_peak": int(ram_peak),
        "random": {
            "accuracy": random_metrics["accuracy"],
            "cross_entropy": random_metrics["cross_entropy"],
            "perplexity": random_metrics["perplexity"],
        },
        "g1_transition": {
            "accuracy": g1_metrics["accuracy"],
            "cross_entropy": g1_metrics["cross_entropy"],
            "perplexity": g1_metrics["perplexity"],
        },
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.results_dir / f"{args.experiment_id}_eval_seed{args.seed}.json"
    run_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("G1 evaluation complete")
    print(f"Corpus lines: {len(lines)}")
    print(f"Vocab size: {len(tokenizer.token_to_id)}")
    print(f"Transitions: {transition_count}")
    print("")
    print("Random baseline")
    print(f"  Accuracy:      {random_metrics['accuracy']:.6f}")
    print(f"  Cross entropy: {random_metrics['cross_entropy']:.6f}")
    print(f"  Perplexity:    {random_metrics['perplexity']:.6f}")
    print("")
    print("G1 transition model")
    print(f"  Accuracy:      {g1_metrics['accuracy']:.6f}")
    print(f"  Cross entropy: {g1_metrics['cross_entropy']:.6f}")
    print(f"  Perplexity:    {g1_metrics['perplexity']:.6f}")
    print("")
    print(f"Inference time (s): {inference_time:.6f}")
    print(f"Tokens/sec:         {tokens_per_second:.2f}")
    print(f"RAM peak (bytes):   {ram_peak}")
    print(f"Run record:         {run_path}")


if __name__ == "__main__":
    main()
