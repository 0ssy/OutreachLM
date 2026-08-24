import json
from pathlib import Path

from outreachlm.telemetry import TelemetryConfig, TrainingTelemetry


def _read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_training_telemetry_writes_structured_outputs(tmp_path: Path) -> None:
    telemetry = TrainingTelemetry(
        TelemetryConfig(output_dir=tmp_path / "telemetry", run_name="b2.9")
    )
    telemetry.record_step(
        {
            "step": 1,
            "micro_step": 1,
            "optimizer_step": 1,
            "loss": 1.5,
            "evaluated": True,
            "checkpointed": False,
            "throughput": {"tokens_processed": 32},
            "memory": {"parameter_bytes": 1024},
        }
    )
    telemetry.record_step(
        {
            "step": 2,
            "micro_step": 2,
            "optimizer_step": 2,
            "loss": 1.0,
            "evaluated": False,
            "checkpointed": True,
            "throughput": {"tokens_processed": 64},
            "memory": {"parameter_bytes": 1024},
        }
    )
    summary = telemetry.finalize({"step": 2})

    metrics = _read_jsonl(tmp_path / "telemetry" / "metrics.jsonl")
    memory = _read_jsonl(tmp_path / "telemetry" / "memory.jsonl")
    events = _read_jsonl(tmp_path / "telemetry" / "events.jsonl")
    summary_file = json.loads((tmp_path / "telemetry" / "summary.json").read_text(encoding="utf-8"))

    assert len(metrics) == 2
    assert len(memory) == 2
    assert summary["steps_recorded"] == 2
    assert summary_file["loss"]["best"] == 1.0
    event_types = [event["event_type"] for event in events]
    assert "run_started" in event_types
    assert "evaluation_completed" in event_types
    assert "checkpoint_saved" in event_types
    assert "run_finished" in event_types
