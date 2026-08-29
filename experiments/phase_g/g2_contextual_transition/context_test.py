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

from common.metrics import distribution_max_abs_diff  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402
from g2_contextual_transition.contextual_transition_model import ContextualTransitionModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 context sensitivity hard-gate test.")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    tokenizer = StupidTokenizer.load(artifact_dir / "tokenizer.json")
    g1 = SparseNGramModel.load(artifact_dir / "g1_baseline.pkl")
    g2 = ContextualTransitionModel.load(artifact_dir / "g2_model.pkl")

    context_a = tokenizer.encode("the cat will", add_bos=False, add_eos=False)
    context_b = tokenizer.encode("the dog will", add_bos=False, add_eos=False)

    tracemalloc.start()
    start = time.perf_counter()
    g1_dist_a = g1.distribution([context_a[-1]])
    g1_dist_b = g1.distribution([context_b[-1]])
    g2_dist_a = g2.distribution(context_a[-2:])
    g2_dist_b = g2.distribution(context_b[-2:])
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    g1_diff = distribution_max_abs_diff(g1_dist_a, g1_dist_b)
    g2_diff = distribution_max_abs_diff(g2_dist_a, g2_dist_b)

    payload = {
        "experiment_id": "g2_context_sensitivity_test",
        "seed": args.seed,
        "contexts": {"a": "the cat will", "b": "the dog will"},
        "final_token_a": tokenizer.id_to_token[context_a[-1]],
        "final_token_b": tokenizer.id_to_token[context_b[-1]],
        "g1_distribution_difference": g1_diff,
        "g2_distribution_difference": g2_diff,
        "hard_gate_context_sensitive": g2_diff > g1_diff,
        "model_storage_bytes": {
            "g1": g1.model_storage_bytes,
            "g2": g2.model_storage_bytes,
        },
        "process_peak_ram_bytes": int(peak),
        "inference_time": elapsed,
    }

    results_dir = Path(__file__).resolve().parent / "results"
    (results_dir / "g2_context_test.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("G2 context test complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
