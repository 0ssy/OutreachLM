from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.datasets import BASE_CORPUS, CONTEXT_AMBIGUITY_CORPUS, LONG_CONTEXT_CORPUS  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import build_stupid_tokenizer_from_lines  # noqa: E402


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def _worker(payload: tuple[list[list[int]], str]) -> tuple[int, float]:
    contexts, model_path = payload
    model = SparseNGramModel.load(model_path)
    start = time.perf_counter()
    count = 0
    for context in contexts:
        _ = model.distribution(context)
        count += 1
    return count, time.perf_counter() - start


def main() -> None:
    seed = 1337
    corpus = BASE_CORPUS + CONTEXT_AMBIGUITY_CORPUS + LONG_CONTEXT_CORPUS
    tokenizer = build_stupid_tokenizer_from_lines(corpus)
    sequences = _encode_lines(tokenizer, corpus)
    model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
    model.fit(sequences)

    contexts: list[list[int]] = []
    for sequence in sequences:
        for pos in range(1, len(sequence) - 1):
            contexts.append(sequence[pos - 1 : pos + 1])
    contexts = contexts * 2000

    root = Path(__file__).resolve().parent
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    model_path = root / "artifacts" / "g8_model.pkl"
    model.save(model_path)

    single_start = time.perf_counter()
    single_count, single_inner = _worker((contexts, str(model_path)))
    single_elapsed = time.perf_counter() - single_start

    mid = len(contexts) // 2
    a = contexts[:mid]
    b = contexts[mid:]
    multi_start = time.perf_counter()
    with mp.get_context("spawn").Pool(processes=2) as pool:
        out = pool.map(_worker, [(a, str(model_path)), (b, str(model_path))])
    multi_elapsed = time.perf_counter() - multi_start
    multi_count = out[0][0] + out[1][0]

    single_tps = single_count / max(single_elapsed, 1e-12)
    multi_tps = multi_count / max(multi_elapsed, 1e-12)

    result = {
        "experiment_id": "g8_multi_cpu",
        "seed": seed,
        "inference_items": len(contexts),
        "single_cpu": {
            "elapsed_seconds": single_elapsed,
            "inner_elapsed_seconds": single_inner,
            "throughput": single_tps,
        },
        "two_cpu": {
            "elapsed_seconds": multi_elapsed,
            "throughput": multi_tps,
        },
        "latency_ratio_two_over_one": multi_elapsed / max(single_elapsed, 1e-12),
        "scaling_efficiency": multi_tps / max(single_tps * 2.0, 1e-12),
        "hard_gates": {
            "useful_parallelism": multi_tps > single_tps,
            "cpu_measured": True,
        },
    }

    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g8_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g8_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("G8 complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
