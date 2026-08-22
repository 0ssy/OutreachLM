from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


class SingleDeviceRuntime:
    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self.device)

    def prepare_batch(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(item.to(self.device) for item in batch)

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()


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


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        runtime: SingleDeviceRuntime,
        scheduler: Any = None,
        hooks: TrainerHooks | None = None,
        eval_interval: int | None = None,
        checkpoint_interval: int | None = None,
        evaluation_fn=None,
        checkpoint_fn=None,
    ):
        if eval_interval is not None and eval_interval <= 0:
            raise ValueError("eval_interval must be > 0 when provided.")
        if checkpoint_interval is not None and checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be > 0 when provided.")

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
        self.state = TrainerState()

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        self.model.train()
        prepared_batch = self.runtime.prepare_batch(batch)
        input_ids, target_ids = prepared_batch

        next_step = self.state.step + 1
        self.hooks.on_step_start(next_step, prepared_batch)

        self.runtime.zero_grad(self.optimizer)

        model_output = self.model(input_ids)
        logits = model_output[0] if isinstance(model_output, tuple) else model_output
        loss = self.loss_fn(logits, target_ids)
        self.hooks.on_loss_computed(next_step, loss)

        self.runtime.backward(loss)
        self.runtime.optimizer_step(self.optimizer)

        if self.scheduler is not None:
            self.scheduler.step()

        self.state.step = next_step
        metrics: dict[str, Any] = {
            "step": self.state.step,
            "loss": float(loss.item()),
            "evaluated": False,
            "checkpointed": False,
        }

        if self.evaluation_fn is not None and self.eval_interval is not None:
            if self.state.step % self.eval_interval == 0:
                eval_metrics = self.evaluation_fn(self.model, self.state.step)
                metrics["evaluated"] = True
                metrics["evaluation"] = eval_metrics
                self.hooks.on_evaluation(self.state.step, eval_metrics)

        if self.checkpoint_fn is not None and self.checkpoint_interval is not None:
            if self.state.step % self.checkpoint_interval == 0:
                checkpoint_payload = self.checkpoint_fn(self.model, self.state.step)
                metrics["checkpointed"] = True
                metrics["checkpoint"] = checkpoint_payload
                self.hooks.on_checkpoint(self.state.step, checkpoint_payload)

        self.hooks.on_step_end(self.state.step, metrics)
        return metrics

    def run_steps(
        self,
        batch_source: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        steps: int,
    ) -> list[dict[str, Any]]:
        if steps <= 0:
            raise ValueError("steps must be > 0.")

        iterator = iter(batch_source)
        outputs = []
        for _ in range(steps):
            batch = next(iterator)
            outputs.append(self.train_step(batch))
        return outputs
