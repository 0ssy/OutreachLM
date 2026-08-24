from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from outreachlm.runtime import Runtime, SingleDeviceRuntime
from outreachlm.telemetry import TrainingTelemetry


class TrainerHooks:
    def on_step_start(self, step: int, batch: tuple[torch.Tensor, ...]) -> None:
        pass

    def on_loss_computed(self, step: int, loss: torch.Tensor) -> None:
        pass

    def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def on_evaluation(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def on_checkpoint(self, step: int, payload: dict[str, Any]) -> None:
        pass


@dataclass
class TrainerState:
    step: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    total_tokens_processed: int = 0
    total_samples_processed: int = 0
    run_start_time: float = 0.0


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        runtime: Runtime,
        scheduler: Any = None,
        hooks: TrainerHooks | None = None,
        eval_interval: int | None = None,
        checkpoint_interval: int | None = None,
        evaluation_fn=None,
        checkpoint_fn=None,
        gradient_accumulation_steps: int = 1,
        telemetry: TrainingTelemetry | None = None,
    ):
        if eval_interval is not None and eval_interval <= 0:
            raise ValueError("eval_interval must be > 0 when provided.")
        if checkpoint_interval is not None and checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0 when provided.")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")

        self.runtime = runtime
        self.model = runtime.prepare_model(model)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.hooks = hooks or TrainerHooks()
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.evaluation_fn = evaluation_fn
        self.checkpoint_fn = checkpoint_fn
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.telemetry = telemetry
        self.state = TrainerState(run_start_time=perf_counter())

    def state_dict(self) -> dict[str, Any]:
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        return {
            "trainer_state": {
                "step": self.state.step,
                "micro_step": self.state.micro_step,
                "optimizer_step": self.state.optimizer_step,
                "total_tokens_processed": self.state.total_tokens_processed,
                "total_samples_processed": self.state.total_samples_processed,
                "run_start_time": self.state.run_start_time,
            },
            "scheduler_state": scheduler_state,
            "scaler_state": self.runtime.get_scaler_state(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        trainer_state = payload.get("trainer_state", {})
        self.state.step = trainer_state.get("step", self.state.step)
        self.state.micro_step = trainer_state.get("micro_step", self.state.micro_step)
        self.state.optimizer_step = trainer_state.get("optimizer_step", self.state.optimizer_step)
        self.state.total_tokens_processed = trainer_state.get(
            "total_tokens_processed",
            self.state.total_tokens_processed,
        )
        self.state.total_samples_processed = trainer_state.get(
            "total_samples_processed",
            self.state.total_samples_processed,
        )
        self.state.run_start_time = trainer_state.get(
            "run_start_time",
            self.state.run_start_time,
        )

        scheduler_state = payload.get("scheduler_state")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        self.runtime.load_scaler_state(payload.get("scaler_state"))

    def _batch_stats(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> dict[str, int]:
        sample_count = int(input_ids.shape[0]) if input_ids.ndim > 0 else 1
        token_count = int(target_ids.numel())
        return {
            "sample_count": sample_count,
            "token_count": token_count,
        }

    def _throughput_metrics(
        self,
        *,
        step_time_seconds: float,
        data_time_seconds: float,
        forward_time_seconds: float,
        backward_time_seconds: float,
        optimizer_time_seconds: float,
    ) -> dict[str, float | int]:
        elapsed = max(perf_counter() - self.state.run_start_time, 1e-12)
        optimizer_steps = self.state.optimizer_step
        micro_steps = self.state.micro_step

        return {
            "tokens_processed": self.state.total_tokens_processed,
            "samples_processed": self.state.total_samples_processed,
            "tokens_per_second": self.state.total_tokens_processed / elapsed,
            "samples_per_second": self.state.total_samples_processed / elapsed,
            "microsteps_per_second": micro_steps / elapsed,
            "optimizer_steps_per_second": optimizer_steps / elapsed,
            "steps_per_second": optimizer_steps / elapsed,
            "step_time_seconds": step_time_seconds,
            "time_per_step_seconds": step_time_seconds,
            "time_per_data_seconds": data_time_seconds,
            "time_per_forward_seconds": forward_time_seconds,
            "time_per_backward_seconds": backward_time_seconds,
            "time_per_optimizer_seconds": optimizer_time_seconds,
            "time_per_optimizer_step_seconds": (
                elapsed / optimizer_steps if optimizer_steps > 0 else 0.0
            ),
            "elapsed_seconds": elapsed,
        }

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        step_start_time = perf_counter()
        self.model.train()
        prepared_batch = self.runtime.prepare_batch(batch)
        after_data_time = perf_counter()
        input_ids, target_ids = prepared_batch

        next_micro_step = self.state.micro_step + 1
        accumulation_index = ((next_micro_step - 1) % self.gradient_accumulation_steps) + 1
        should_step_optimizer = accumulation_index == self.gradient_accumulation_steps
        self.hooks.on_step_start(next_micro_step, prepared_batch)

        if accumulation_index == 1:
            self.runtime.zero_grad(self.optimizer)

        forward_start_time = perf_counter()
        with self.runtime.autocast_context():
            model_output = self.model(input_ids)
            logits = model_output[0] if isinstance(model_output, tuple) else model_output
            raw_loss = self.loss_fn(logits, target_ids)
        forward_end_time = perf_counter()
        self.hooks.on_loss_computed(next_micro_step, raw_loss)

        scaled_loss = raw_loss / self.gradient_accumulation_steps
        backward_start_time = perf_counter()
        self.runtime.backward(scaled_loss)
        backward_end_time = perf_counter()

        optimizer_start_time = perf_counter()
        if should_step_optimizer:
            self.runtime.optimizer_step(self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            self.runtime.synchronize()
            self.state.optimizer_step += 1
        optimizer_end_time = perf_counter()

        self.state.micro_step = next_micro_step
        self.state.step = self.state.optimizer_step
        batch_stats = self._batch_stats(input_ids, target_ids)
        self.state.total_tokens_processed += batch_stats["token_count"]
        self.state.total_samples_processed += batch_stats["sample_count"]
        throughput = self._throughput_metrics(
            step_time_seconds=perf_counter() - step_start_time,
            data_time_seconds=after_data_time - step_start_time,
            forward_time_seconds=forward_end_time - forward_start_time,
            backward_time_seconds=backward_end_time - backward_start_time,
            optimizer_time_seconds=optimizer_end_time - optimizer_start_time,
        )
        metrics: dict[str, Any] = {
            "step": self.state.step,
            "micro_step": self.state.micro_step,
            "optimizer_step": self.state.optimizer_step,
            "accumulation_index": accumulation_index,
            "loss": float(raw_loss.item()),
            "evaluated": False,
            "checkpointed": False,
            "memory": self.runtime.collect_memory_stats(self.model, self.optimizer),
            "throughput": throughput,
        }

        if should_step_optimizer and self.evaluation_fn is not None and self.eval_interval is not None:
            if self.state.optimizer_step % self.eval_interval == 0:
                eval_metrics = self.evaluation_fn(self.model, self.state.step)
                metrics["evaluated"] = True
                metrics["evaluation"] = eval_metrics
                self.hooks.on_evaluation(self.state.step, eval_metrics)

        if should_step_optimizer and self.checkpoint_fn is not None and self.checkpoint_interval is not None:
            if self.state.optimizer_step % self.checkpoint_interval == 0:
                checkpoint_payload = self.checkpoint_fn(self.model, self.state.step)
                metrics["checkpointed"] = True
                metrics["checkpoint"] = checkpoint_payload
                self.hooks.on_checkpoint(self.state.step, checkpoint_payload)

        if self.telemetry is not None:
            self.telemetry.record_step(metrics)
        self.hooks.on_step_end(self.state.step, metrics)
        return metrics

    def finalize_accumulation(self) -> dict[str, Any] | None:
        if self.state.micro_step == 0:
            return None
        pending = self.state.micro_step % self.gradient_accumulation_steps
        if pending == 0:
            return None

        self.runtime.optimizer_step(self.optimizer)
        if self.scheduler is not None:
            self.scheduler.step()
        self.runtime.synchronize()
        self.state.optimizer_step += 1
        self.state.step = self.state.optimizer_step
        throughput = self._throughput_metrics(
            step_time_seconds=0.0,
            data_time_seconds=0.0,
            forward_time_seconds=0.0,
            backward_time_seconds=0.0,
            optimizer_time_seconds=0.0,
        )

        metrics: dict[str, Any] = {
            "step": self.state.step,
            "micro_step": self.state.micro_step,
            "optimizer_step": self.state.optimizer_step,
            "accumulation_index": pending,
            "loss": float("nan"),
            "evaluated": False,
            "checkpointed": False,
            "finalized_partial_accumulation": True,
            "memory": self.runtime.collect_memory_stats(self.model, self.optimizer),
            "throughput": throughput,
        }

        if self.evaluation_fn is not None and self.eval_interval is not None:
            if self.state.optimizer_step % self.eval_interval == 0:
                eval_metrics = self.evaluation_fn(self.model, self.state.step)
                metrics["evaluated"] = True
                metrics["evaluation"] = eval_metrics
                self.hooks.on_evaluation(self.state.step, eval_metrics)

        if self.checkpoint_fn is not None and self.checkpoint_interval is not None:
            if self.state.optimizer_step % self.checkpoint_interval == 0:
                checkpoint_payload = self.checkpoint_fn(self.model, self.state.step)
                metrics["checkpointed"] = True
                metrics["checkpoint"] = checkpoint_payload
                self.hooks.on_checkpoint(self.state.step, checkpoint_payload)

        if self.telemetry is not None:
            self.telemetry.record_step(metrics)
        self.hooks.on_step_end(self.state.step, metrics)
        return metrics

    def run_steps(
        self,
        batch_source: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        steps: int,
        flush_partial_accumulation: bool = False,
    ) -> list[dict[str, Any]]:
        if steps <= 0:
            raise ValueError("steps must be > 0.")

        iterator = iter(batch_source)
        outputs = []
        for _ in range(steps):
            batch = next(iterator)
            outputs.append(self.train_step(batch))
        if flush_partial_accumulation:
            final_metrics = self.finalize_accumulation()
            if final_metrics is not None:
                outputs.append(final_metrics)
        if self.telemetry is not None:
            self.telemetry.finalize(
                {
                    "step": self.state.step,
                    "micro_step": self.state.micro_step,
                    "optimizer_step": self.state.optimizer_step,
                    "total_tokens_processed": self.state.total_tokens_processed,
                    "total_samples_processed": self.state.total_samples_processed,
                }
            )
        return outputs
