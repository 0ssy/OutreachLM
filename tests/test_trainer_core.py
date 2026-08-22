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
