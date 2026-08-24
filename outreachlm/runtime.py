from dataclasses import dataclass
from typing import Literal, Protocol
from contextlib import nullcontext
from typing import Any

import torch

PrecisionMode = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    world_size: int = 1
    rank: int = 0
    is_distributed: bool = False
    precision: PrecisionMode = "fp32"


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

    def autocast_context(self):
        ...

    def get_scaler_state(self) -> dict | None:
        ...

    def load_scaler_state(self, scaler_state: dict | None) -> None:
        ...

    def collect_memory_stats(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        ...

    def synchronize(self) -> None:
        ...


class SingleDeviceRuntime:
    def __init__(self, device: torch.device | str, *, precision: PrecisionMode = "fp32"):
        resolved = torch.device(device)
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported precision mode: {precision}")
        if precision == "fp16" and resolved.type != "cuda":
            raise ValueError("fp16 precision is only supported on CUDA devices.")
        if precision == "bf16" and resolved.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 precision is not supported on this CUDA device.")

        self._precision = precision
        self._scaler: torch.cuda.amp.GradScaler | None = None
        if resolved.type == "cuda" and precision == "fp16":
            self._scaler = torch.cuda.amp.GradScaler(enabled=True)

        self._info = RuntimeInfo(
            device=resolved,
            world_size=1,
            rank=0,
            is_distributed=False,
            precision=precision,
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
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            return
        loss.backward()

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
            return
        optimizer.step()

    def autocast_context(self):
        if self.info.precision == "fp32":
            return nullcontext()

        if self.info.device.type == "cuda":
            dtype = torch.float16 if self.info.precision == "fp16" else torch.bfloat16
            return torch.autocast(device_type="cuda", dtype=dtype)
        if self.info.device.type == "cpu" and self.info.precision == "bf16":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        raise RuntimeError(
            f"Unsupported autocast combination: device={self.info.device.type}, "
            f"precision={self.info.precision}"
        )

    def get_scaler_state(self) -> dict | None:
        if self._scaler is None:
            return None
        return self._scaler.state_dict()

    def load_scaler_state(self, scaler_state: dict | None) -> None:
        if scaler_state is None:
            return
        if self._scaler is None:
            raise ValueError("Scaler state provided but runtime has no active gradient scaler.")
        self._scaler.load_state_dict(scaler_state)

    def collect_memory_stats(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, Any]:
        parameter_count = 0
        trainable_parameter_count = 0
        parameter_bytes = 0
        gradient_bytes = 0
        for parameter in model.parameters():
            count = parameter.numel()
            parameter_count += count
            if parameter.requires_grad:
                trainable_parameter_count += count
            parameter_bytes += count * parameter.element_size()
            if parameter.grad is not None:
                gradient_bytes += parameter.grad.numel() * parameter.grad.element_size()

        buffer_bytes = 0
        for buffer in model.buffers():
            buffer_bytes += buffer.numel() * buffer.element_size()

        optimizer_state_bytes = 0
        for state in optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for value in state.values():
                if torch.is_tensor(value):
                    optimizer_state_bytes += value.numel() * value.element_size()

        stats: dict[str, Any] = {
            "device_type": self.info.device.type,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "buffer_bytes": buffer_bytes,
            "optimizer_state_bytes": optimizer_state_bytes,
            "model_and_grad_bytes": parameter_bytes + gradient_bytes + buffer_bytes,
            "estimated_training_state_bytes": (
                parameter_bytes + gradient_bytes + buffer_bytes + optimizer_state_bytes
            ),
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }

        if self.info.device.type == "cuda":
            device = self.info.device
            stats["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
            stats["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
            stats["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            stats["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)

        return stats

    def synchronize(self) -> None:
        # No-op on single-device runtime.
        return None
