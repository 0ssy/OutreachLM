from __future__ import annotations

import argparse
import json
from pathlib import Path

from outreachlm.phase_h_runtime import BoundedStateRuntime, PhaseHRuntimeConfig
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and save the integrated Phase H bounded runtime.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--validation-split", type=float, default=VALIDATION_SPLIT)
    parser.add_argument("--max-train-lines", type=int, default=2000)
    parser.add_argument("--max-eval-lines", type=int, default=300)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("models") / "outreachlm_phase_h_runtime.pkl",
    )
    parser.add_argument("--quantization-mode", type=str, default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--repetition-decay", type=float, default=0.85)
    parser.add_argument("--repetition-floor", type=float, default=0.25)
    parser.add_argument("--unk-alert-threshold", type=float, default=0.10)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments") / "phase_h" / "runtime_integration_train.json",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    runtime_config = PhaseHRuntimeConfig(
        quantization_mode=args.quantization_mode,
        repetition_decay=args.repetition_decay,
        repetition_floor=args.repetition_floor,
        unk_alert_threshold=args.unk_alert_threshold,
    )

    runtime, train_lines, eval_lines = BoundedStateRuntime.from_corpus_path(
        corpus_path=args.corpus,
        validation_split=args.validation_split,
        max_train_lines=args.max_train_lines,
        max_eval_lines=args.max_eval_lines,
        config=runtime_config,
    )
    eval_metrics = runtime.evaluate_lines(eval_lines, apply_safety=False)
    runtime.save(args.artifact)

    report = {
        "experiment_id": "phase_h_runtime_integration_train",
        "artifact_path": str(args.artifact.resolve()),
        "train_line_count": len(train_lines),
        "eval_line_count": len(eval_lines),
        "runtime_config": runtime_config.to_dict(),
        "eval_metrics": eval_metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
