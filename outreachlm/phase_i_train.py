from __future__ import annotations

import argparse
import json
from pathlib import Path

from outreachlm.phase_h_runtime import PhaseHRuntimeConfig
from outreachlm.phase_i_runtime import PhaseIRuntimeConfig, SemanticRuntime
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and save the integrated Phase I semantic runtime.")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--validation-split", type=float, default=VALIDATION_SPLIT)
    parser.add_argument("--max-train-lines", type=int, default=1200)
    parser.add_argument("--max-eval-lines", type=int, default=160)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("models") / "outreachlm_phase_i_runtime.pkl",
    )
    parser.add_argument("--quantization-mode", type=str, default="fp16", choices=("fp16", "fp32"))
    parser.add_argument("--repetition-decay", type=float, default=0.85)
    parser.add_argument("--repetition-floor", type=float, default=0.25)
    parser.add_argument("--unk-alert-threshold", type=float, default=0.10)
    parser.add_argument("--max-syntax-depth", type=int, default=4)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments") / "phase_i" / "runtime_integration_train.json",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    phase_h_config = PhaseHRuntimeConfig(
        quantization_mode=args.quantization_mode,
        repetition_decay=args.repetition_decay,
        repetition_floor=args.repetition_floor,
        unk_alert_threshold=args.unk_alert_threshold,
    )
    phase_i_config = PhaseIRuntimeConfig(max_syntax_depth=args.max_syntax_depth)
    runtime, train_lines, eval_lines = SemanticRuntime.from_corpus_path(
        corpus_path=args.corpus,
        validation_split=args.validation_split,
        max_train_lines=args.max_train_lines,
        max_eval_lines=args.max_eval_lines,
        phase_h_config=phase_h_config,
        config=phase_i_config,
    )
    eval_metrics = runtime.evaluate_semantic_lines(eval_lines)
    runtime.save(args.artifact)

    report = {
        "experiment_id": "phase_i_runtime_integration_train",
        "artifact_path": str(args.artifact.resolve()),
        "train_line_count": len(train_lines),
        "eval_line_count": len(eval_lines),
        "phase_h_runtime_config": phase_h_config.to_dict(),
        "phase_i_runtime_config": phase_i_config.to_dict(),
        "eval_metrics": eval_metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
