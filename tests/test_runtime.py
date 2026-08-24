import pytest
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
    assert runtime.info.precision == "fp32"


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


def test_cpu_bf16_runtime_precision_and_no_scaler():
    runtime = SingleDeviceRuntime("cpu", precision="bf16")
    assert runtime.info.precision == "bf16"
    assert runtime.get_scaler_state() is None
    with runtime.autocast_context():
        value = torch.ones(1) + torch.ones(1)
    assert torch.allclose(value, torch.tensor([2.0]))


def test_fp16_rejected_on_cpu():
    with pytest.raises(ValueError, match="fp16 precision is only supported on CUDA devices"):
        SingleDeviceRuntime("cpu", precision="fp16")


def test_loading_scaler_without_scaler_raises():
    runtime = SingleDeviceRuntime("cpu")
    with pytest.raises(ValueError, match="runtime has no active gradient scaler"):
        runtime.load_scaler_state({"scale": 1.0})


def test_collect_memory_stats_reports_expected_cpu_fields():
    runtime = SingleDeviceRuntime("cpu")
    model = runtime.prepare_model(TinyModel())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    runtime.zero_grad(optimizer)
    loss = nn.functional.mse_loss(model(x), y)
    runtime.backward(loss)

    stats = runtime.collect_memory_stats(model, optimizer)
    assert stats["device_type"] == "cpu"
    assert stats["parameter_count"] > 0
    assert stats["trainable_parameter_count"] > 0
    assert stats["parameter_bytes"] > 0
    assert stats["gradient_bytes"] > 0
    assert stats["optimizer_state_bytes"] >= 0
    assert stats["estimated_training_state_bytes"] >= stats["parameter_bytes"]
    assert stats["cuda_allocated_bytes"] is None
