import torch
import torch.nn as nn

from outreachlm.trainer_core import SingleDeviceRuntime, Trainer, TrainerHooks


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RecordingHooks(TrainerHooks):
    def __init__(self) -> None:
        self.step_start_calls = 0
        self.loss_calls = 0
        self.step_end_calls = 0
        self.eval_calls = 0
        self.checkpoint_calls = 0

    def on_step_start(self, step: int, batch: tuple[torch.Tensor, ...]) -> None:
        self.step_start_calls += 1

    def on_loss_computed(self, step: int, loss: torch.Tensor) -> None:
        self.loss_calls += 1

    def on_step_end(self, step: int, metrics: dict[str, object]) -> None:
        self.step_end_calls += 1

    def on_evaluation(self, step: int, metrics: dict[str, object]) -> None:
        self.eval_calls += 1

    def on_checkpoint(self, step: int, payload: dict[str, object]) -> None:
        self.checkpoint_calls += 1


def infinite_batches(batch: tuple[torch.Tensor, torch.Tensor]):
    while True:
        yield batch


def test_trainer_constructs():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    assert trainer.state.step == 0


def test_train_step_updates_parameters():
    torch.manual_seed(7)
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))

    before = model.linear.weight.detach().clone()
    trainer.train_step(batch)
    after = model.linear.weight.detach().clone()
    assert not torch.equal(before, after)


def test_scheduler_executes():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))

    lr_before = optimizer.param_groups[0]["lr"]
    trainer.train_step(batch)
    lr_after = optimizer.param_groups[0]["lr"]
    assert lr_after < lr_before


def test_loss_changes_across_steps():
    torch.manual_seed(123)
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))

    first = trainer.train_step(batch)["loss"]
    second = trainer.train_step(batch)["loss"]
    assert first != second


def test_checkpoint_hook_and_callback_are_called():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = RecordingHooks()
    seen_steps = []

    def checkpoint_fn(_, step: int):
        seen_steps.append(step)
        return {"step": step}

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        hooks=hooks,
        checkpoint_interval=1,
        checkpoint_fn=checkpoint_fn,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    trainer.run_steps(infinite_batches(batch), steps=2)

    assert seen_steps == [1, 2]
    assert hooks.checkpoint_calls == 2


def test_evaluation_hook_and_callback_are_called():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = RecordingHooks()
    seen_steps = []

    def evaluation_fn(_, step: int):
        seen_steps.append(step)
        return {"eval_step": step}

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        hooks=hooks,
        eval_interval=1,
        evaluation_fn=evaluation_fn,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    trainer.run_steps(infinite_batches(batch), steps=2)

    assert seen_steps == [1, 2]
    assert hooks.eval_calls == 2


def test_single_device_runtime_moves_batch_to_runtime_device():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    prepared = trainer.runtime.prepare_batch(batch)
    assert prepared[0].device.type == "cpu"
    assert prepared[1].device.type == "cpu"


def test_hooks_fire_for_each_step():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = RecordingHooks()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        hooks=hooks,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    trainer.run_steps(infinite_batches(batch), steps=3)

    assert hooks.step_start_calls == 3
    assert hooks.loss_calls == 3
    assert hooks.step_end_calls == 3


def test_throughput_metrics_include_counters_and_timings():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    metrics = trainer.train_step(batch)
    throughput = metrics["throughput"]

    assert throughput["tokens_processed"] == 2
    assert throughput["samples_processed"] == 2
    assert throughput["tokens_per_second"] > 0.0
    assert throughput["samples_per_second"] > 0.0
    assert throughput["step_time_seconds"] >= 0.0
    assert throughput["time_per_forward_seconds"] >= 0.0
    assert throughput["time_per_backward_seconds"] >= 0.0
    assert throughput["time_per_optimizer_seconds"] >= 0.0


def test_throughput_is_accumulation_aware():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        gradient_accumulation_steps=2,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))

    first = trainer.train_step(batch)["throughput"]
    second = trainer.train_step(batch)["throughput"]

    assert first["tokens_processed"] == 2
    assert second["tokens_processed"] == 4
    assert first["optimizer_steps_per_second"] == 0.0
    assert second["optimizer_steps_per_second"] > 0.0


def test_gradient_accumulation_delays_optimizer_step():
    torch.manual_seed(11)
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        gradient_accumulation_steps=2,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))

    before = model.linear.weight.detach().clone()
    first_metrics = trainer.train_step(batch)
    after_first = model.linear.weight.detach().clone()
    second_metrics = trainer.train_step(batch)
    after_second = model.linear.weight.detach().clone()

    assert torch.equal(before, after_first)
    assert not torch.equal(after_first, after_second)
    assert first_metrics["optimizer_step"] == 0
    assert second_metrics["optimizer_step"] == 1
    assert trainer.state.micro_step == 2
    assert trainer.state.optimizer_step == 1


def test_accumulated_gradients_match_larger_batch_within_tolerance():
    torch.manual_seed(17)
    reference = TinyModel()
    accumulated = TinyModel()
    accumulated.load_state_dict(reference.state_dict())

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    runtime = SingleDeviceRuntime("cpu")

    large_batch_trainer = Trainer(
        model=reference,
        optimizer=reference_optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    accumulation_trainer = Trainer(
        model=accumulated,
        optimizer=accumulated_optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        gradient_accumulation_steps=2,
    )

    large_inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    large_targets = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    micro_1 = (large_inputs[:2], large_targets[:2])
    micro_2 = (large_inputs[2:], large_targets[2:])

    large_batch_trainer.train_step((large_inputs, large_targets))
    accumulation_trainer.train_step(micro_1)
    accumulation_trainer.train_step(micro_2)

    assert torch.allclose(
        reference.linear.weight.detach(),
        accumulated.linear.weight.detach(),
        atol=1e-6,
        rtol=1e-6,
    )


def test_run_steps_can_flush_partial_accumulation():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        gradient_accumulation_steps=3,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))

    outputs = trainer.run_steps(
        infinite_batches(batch),
        steps=2,
        flush_partial_accumulation=True,
    )

    assert len(outputs) == 3
    assert outputs[-1]["finalized_partial_accumulation"] is True
    assert trainer.state.optimizer_step == 1


def test_eval_and_checkpoint_intervals_use_optimizer_steps_with_accumulation():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = RecordingHooks()
    eval_steps: list[int] = []
    checkpoint_steps: list[int] = []

    def evaluation_fn(_, step: int):
        eval_steps.append(step)
        return {"eval_step": step}

    def checkpoint_fn(_, step: int):
        checkpoint_steps.append(step)
        return {"checkpoint_step": step}

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        hooks=hooks,
        eval_interval=1,
        checkpoint_interval=1,
        evaluation_fn=evaluation_fn,
        checkpoint_fn=checkpoint_fn,
        gradient_accumulation_steps=2,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    trainer.run_steps(infinite_batches(batch), steps=2)

    assert eval_steps == [1]
    assert checkpoint_steps == [1]
    assert hooks.eval_calls == 1
    assert hooks.checkpoint_calls == 1


def test_trainer_runs_with_cpu_bf16_runtime():
    torch.manual_seed(5)
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu", precision="bf16")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    metrics = trainer.train_step(batch)
    assert metrics["optimizer_step"] == 1
    assert runtime.info.precision == "bf16"
    assert metrics["memory"]["device_type"] == "cpu"
    assert metrics["memory"]["parameter_bytes"] > 0


def test_trainer_state_dict_round_trip_restores_progress():
    model = TinyModel()
    runtime = SingleDeviceRuntime("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=nn.MSELoss(),
        runtime=runtime,
        gradient_accumulation_steps=2,
    )
    batch = (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]]))
    trainer.run_steps(infinite_batches(batch), steps=2)
    saved = trainer.state_dict()

    restored_model = TinyModel()
    restored_model.load_state_dict(model.state_dict())
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer,
        step_size=1,
        gamma=0.9,
    )
    restored_trainer = Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        loss_fn=nn.MSELoss(),
        runtime=SingleDeviceRuntime("cpu"),
        gradient_accumulation_steps=2,
    )
    restored_trainer.load_state_dict(saved)

    assert restored_trainer.state.micro_step == trainer.state.micro_step
    assert restored_trainer.state.optimizer_step == trainer.state.optimizer_step
    assert (
        restored_trainer.state.total_tokens_processed
        == trainer.state.total_tokens_processed
    )
    assert (
        restored_trainer.state.total_samples_processed
        == trainer.state.total_samples_processed
    )
