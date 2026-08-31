from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from outreachlm.phase_j_runtime import PhaseJRuntimeConfig, ReasoningRuntime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and save the integrated Phase J reasoning runtime.")
    parser.add_argument("--artifact", type=Path, default=Path("models") / "outreachlm_phase_j_runtime.pkl")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("experiments") / "phase_j" / "runtime_integration_train.json",
    )
    parser.add_argument("--max-hop-depth", type=int, default=32)
    parser.add_argument("--min-deductive-accuracy", type=float, default=0.98)
    parser.add_argument("--min-8-hop-accuracy", type=float, default=0.90)
    parser.add_argument("--max-dependency-violations", type=int, default=0)
    parser.add_argument("--min-self-verification-intercept-rate", type=float, default=0.95)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = PhaseJRuntimeConfig(
        max_hop_depth=args.max_hop_depth,
        min_deductive_accuracy=args.min_deductive_accuracy,
        min_8_hop_accuracy=args.min_8_hop_accuracy,
        max_dependency_violations=args.max_dependency_violations,
        min_self_verification_intercept_rate=args.min_self_verification_intercept_rate,
    )
    runtime = ReasoningRuntime(config=config)
    submission, detail = runtime.run_suite()
    runtime.save(args.artifact)

    report = {
        "experiment_id": "phase_j_runtime_integration_train",
        "artifact_path": str(args.artifact.resolve()),
        "phase_j_runtime_config": config.to_dict(),
        "submission": submission,
        "detail": detail,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
