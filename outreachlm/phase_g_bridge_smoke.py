from __future__ import annotations

import json
from pathlib import Path

from outreachlm.phase_g_bridge import PhaseGHybridRuntime
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT


def main() -> None:
    max_train_lines = 2000
    max_eval_lines = 300
    runtime, train_lines, eval_lines = PhaseGHybridRuntime.from_corpus_path(
        CORPUS_PATH,
        validation_split=VALIDATION_SPLIT,
        max_train_lines=max_train_lines,
        max_eval_lines=max_eval_lines,
    )
    metrics = runtime.evaluate_lines(eval_lines)
    sample_distribution = runtime.distribution_from_text("the cat")
    sample_top_token_id = int(sample_distribution.argmax())
    sample_top_probability = float(sample_distribution[sample_top_token_id])
    sample_top_token = runtime.tokenizer.id_to_token[sample_top_token_id]

    artifact_path = Path("models") / "phase_g_hybrid_runtime.pkl"
    runtime.save(artifact_path)

    result = {
        "experiment_id": "phase_g_bridge_smoke",
        "corpus_path": str(Path(CORPUS_PATH).resolve()),
        "train_line_count": len(train_lines),
        "eval_line_count": len(eval_lines),
        "line_caps": {"max_train_lines": max_train_lines, "max_eval_lines": max_eval_lines},
        "metrics": metrics,
        "sample_prediction": {
            "prompt": "the cat",
            "top_token_id": sample_top_token_id,
            "top_token": sample_top_token,
            "top_probability": sample_top_probability,
        },
        "artifact_path": str(artifact_path.resolve()),
    }

    output_dir = Path("experiments") / "phase_g" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase_g_bridge_smoke.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
