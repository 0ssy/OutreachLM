from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    world_size: int = 1
    rank: int = 0
    is_distributed: bool = False


class Runtime(Protocol):
    @property
    def info(self) -> RuntimeInfo:
        ...

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        ...

    def prepare_batch(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        ...

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        ...

    def backward(self, loss: torch.Tensor) -> None:
        ...

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        ...

    def synchronize(self) -> None:
        ...


class SingleDeviceRuntime:
    def __init__(self, device: torch.device | str):
        resolved = torch.device(device)
        self._info = RuntimeInfo(
            device=resolved,
            world_size=1,
            rank=0,
            is_distributed=False,
        )

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self.info.device)

    def prepare_batch(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(item.to(self.info.device) for item in batch)

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def synchronize(self) -> None:
        # No-op on single-device runtime.
        return None
