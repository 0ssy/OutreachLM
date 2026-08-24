from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
from time import perf_counter


@dataclass(frozen=True)
class TelemetryConfig:
    output_dir: Path
    run_name: str = "training-run"

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name must not be empty.")


class TrainingTelemetry:
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.config.output_dir / "metrics.jsonl"
        self._memory_path = self.config.output_dir / "memory.jsonl"
        self._events_path = self.config.output_dir / "events.jsonl"
        self._summary_path = self.config.output_dir / "summary.json"
        self._step_count = 0
        self._loss_sum = 0.0
        self._first_loss: float | None = None
        self._last_loss: float | None = None
        self._best_loss: float | None = None
        self._start = perf_counter()

        self.record_event("run_started", {"run_name": self.config.run_name})

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self._events_path,
            {
                "event_type": event_type,
                "payload": payload,
            },
        )

    def record_step(self, metrics: dict[str, Any]) -> None:
        self._step_count += 1
        loss = float(metrics.get("loss", float("nan")))
        if loss == loss:
            if self._first_loss is None:
                self._first_loss = loss
            self._last_loss = loss
            self._loss_sum += loss
            if self._best_loss is None or loss < self._best_loss:
                self._best_loss = loss

        core_metrics = {
            "step": metrics.get("step"),
            "micro_step": metrics.get("micro_step"),
            "optimizer_step": metrics.get("optimizer_step"),
            "loss": metrics.get("loss"),
            "evaluated": metrics.get("evaluated", False),
            "checkpointed": metrics.get("checkpointed", False),
            "throughput": metrics.get("throughput", {}),
        }
        self._append_jsonl(self._metrics_path, core_metrics)

        memory_payload = metrics.get("memory", {})
        self._append_jsonl(
            self._memory_path,
            {
                "step": metrics.get("step"),
                "micro_step": metrics.get("micro_step"),
                "optimizer_step": metrics.get("optimizer_step"),
                "memory": memory_payload,
            },
        )

        if metrics.get("evaluated"):
            self.record_event("evaluation_completed", {"step": metrics.get("step")})
        if metrics.get("checkpointed"):
            self.record_event("checkpoint_saved", {"step": metrics.get("step")})
        if metrics.get("finalized_partial_accumulation"):
            self.record_event(
                "partial_accumulation_flushed",
                {"step": metrics.get("step")},
            )

    def finalize(self, final_state: dict[str, Any]) -> dict[str, Any]:
        elapsed = max(perf_counter() - self._start, 1e-12)
        mean_loss = self._loss_sum / self._step_count if self._step_count > 0 else None
        summary = {
            "run_name": self.config.run_name,
            "steps_recorded": self._step_count,
            "elapsed_seconds": elapsed,
            "loss": {
                "first": self._first_loss,
                "last": self._last_loss,
                "best": self._best_loss,
                "mean": mean_loss,
            },
            "final_state": final_state,
        }
        with open(self._summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        self.record_event("run_finished", {"steps_recorded": self._step_count})
        return summary
