from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, Protocol
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

PrecisionMode = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class RuntimeInfo:
    device: torch.device
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    is_distributed: bool = False
    is_main_process: bool = True
    precision: PrecisionMode = "fp32"


@dataclass(frozen=True)
class DistributedRuntimeConfig:
    backend: str = "gloo"
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    init_method: str | None = None
    device: str | None = None
    precision: PrecisionMode = "fp32"

    def __post_init__(self) -> None:
        if self.world_size <= 1:
            raise ValueError("world_size must be > 1 for DistributedRuntimeConfig.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size.")
        if self.local_rank < 0:
            raise ValueError("local_rank must be >= 0.")
        if self.master_port <= 0:
            raise ValueError("master_port must be > 0.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16.")


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

    def all_reduce_sum(self, value: float) -> float:
        ...

    def barrier(self) -> None:
        ...

    def synchronize(self) -> None:
        ...

    def destroy(self) -> None:
        ...


def _build_scaler(device: torch.device, precision: PrecisionMode) -> torch.cuda.amp.GradScaler | None:
    if device.type == "cuda" and precision == "fp16":
        return torch.cuda.amp.GradScaler(enabled=True)
    return None


def _validate_precision(device: torch.device, precision: PrecisionMode) -> None:
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision mode: {precision}")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 precision is only supported on CUDA devices.")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 precision is not supported on this CUDA device.")


def _resolve_device(local_rank: int, configured: str | None) -> torch.device:
    if configured is not None:
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _memory_stats(
    *,
    info: RuntimeInfo,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    communication_seconds: float,
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
        "rank": info.rank,
        "local_rank": info.local_rank,
        "world_size": info.world_size,
        "device_type": info.device.type,
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
        "communication_time_seconds": communication_seconds,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }

    if info.device.type == "cuda":
        device = info.device
        stats["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
        stats["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
        stats["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        stats["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return stats


class SingleDeviceRuntime:
    def __init__(self, device: torch.device | str, *, precision: PrecisionMode = "fp32"):
        resolved = torch.device(device)
        _validate_precision(resolved, precision)
        self._communication_seconds = 0.0
        self._scaler = _build_scaler(resolved, precision)
        self._info = RuntimeInfo(
            device=resolved,
            world_size=1,
            rank=0,
            local_rank=0,
            is_distributed=False,
            is_main_process=True,
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
        return _memory_stats(
            info=self.info,
            model=model,
            optimizer=optimizer,
            communication_seconds=self._communication_seconds,
        )

    def all_reduce_sum(self, value: float) -> float:
        return value

    def barrier(self) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def destroy(self) -> None:
        return None


class DistributedRuntime:
    def __init__(self, config: DistributedRuntimeConfig):
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available in this environment.")

        self._config = config
        self._device = _resolve_device(config.local_rank, config.device)
        _validate_precision(self._device, config.precision)
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)

        init_method = config.init_method
        if init_method is None:
            init_method = f"tcp://{config.master_addr}:{config.master_port}"

        if not dist.is_initialized():
            dist.init_process_group(
                backend=config.backend,
                init_method=init_method,
                rank=config.rank,
                world_size=config.world_size,
            )
            self._owns_process_group = True
        else:
            self._owns_process_group = False

        self._communication_seconds = 0.0
        self._scaler = _build_scaler(self._device, config.precision)
        self._wrapped_model: DistributedDataParallel | None = None
        self._info = RuntimeInfo(
            device=self._device,
            world_size=config.world_size,
            rank=config.rank,
            local_rank=config.local_rank,
            is_distributed=True,
            is_main_process=config.rank == 0,
            precision=config.precision,
        )

    @property
    def info(self) -> RuntimeInfo:
        return self._info

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        base = model.to(self.info.device)
        if self.info.device.type == "cuda":
            wrapped = DistributedDataParallel(base, device_ids=[self.info.local_rank])
        else:
            wrapped = DistributedDataParallel(base)
        self._wrapped_model = wrapped
        return wrapped

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
        return _memory_stats(
            info=self.info,
            model=model,
            optimizer=optimizer,
            communication_seconds=self._communication_seconds,
        )

    def all_reduce_sum(self, value: float) -> float:
        tensor = torch.tensor(value, device=self.info.device, dtype=torch.float64)
        start = perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self._communication_seconds += perf_counter() - start
        return float(tensor.item())

    def barrier(self) -> None:
        start = perf_counter()
        dist.barrier()
        self._communication_seconds += perf_counter() - start

    def synchronize(self) -> None:
        self.barrier()

    def destroy(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def distributed_runtime_from_env(
    *,
    backend: str = "gloo",
    precision: PrecisionMode = "fp32",
) -> DistributedRuntime:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    return DistributedRuntime(
        DistributedRuntimeConfig(
            backend=backend,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            master_addr=master_addr,
            master_port=master_port,
            precision=precision,
        )
    )
