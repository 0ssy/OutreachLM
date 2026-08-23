import torch
import torch.nn as nn

from outreachlm.runtime import SingleDeviceRuntime


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_single_device_runtime_info_defaults():
    runtime = SingleDeviceRuntime("cpu")
    assert runtime.info.device.type == "cpu"
    assert runtime.info.world_size == 1
    assert runtime.info.rank == 0
    assert runtime.info.is_distributed is False


def test_prepare_model_places_model_on_runtime_device():
    runtime = SingleDeviceRuntime("cpu")
    model = TinyModel()
    prepared = runtime.prepare_model(model)
    assert next(prepared.parameters()).device.type == "cpu"


def test_prepare_batch_places_tensors_on_runtime_device():
    runtime = SingleDeviceRuntime("cpu")
    batch = (torch.randn(4, 2), torch.randn(4, 1))
    prepared = runtime.prepare_batch(batch)
    assert prepared[0].device.type == "cpu"
    assert prepared[1].device.type == "cpu"


def test_backward_and_optimizer_step_update_parameters():
    torch.manual_seed(1)
    runtime = SingleDeviceRuntime("cpu")
    model = runtime.prepare_model(TinyModel())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(8, 2)
    y = torch.randn(8, 1)
    runtime.zero_grad(optimizer)
    pred = model(x)
    loss = nn.functional.mse_loss(pred, y)
    before = model.linear.weight.detach().clone()
    runtime.backward(loss)
    runtime.optimizer_step(optimizer)
    after = model.linear.weight.detach().clone()

    assert not torch.equal(before, after)


def test_synchronize_is_noop():
    runtime = SingleDeviceRuntime("cpu")
    runtime.synchronize()
