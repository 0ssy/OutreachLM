import json
import multiprocessing as mp
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from outreachlm.runtime import (
    DistributedRuntime,
    DistributedRuntimeConfig,
    FSDPRuntime,
    create_parallel_runtime,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_world(world_size: int, worker, *args) -> None:
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for rank in range(world_size):
        process = context.Process(target=worker, args=(rank, world_size, *args))
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=60)
        assert process.exitcode == 0


def _metadata_worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    runtime = DistributedRuntime(
        DistributedRuntimeConfig(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            master_addr="127.0.0.1",
            master_port=port,
            device="cpu",
        )
    )
    reduced = runtime.all_reduce_sum(float(rank + 1))
    runtime.barrier()
    payload = {
        "rank": runtime.info.rank,
        "world_size": runtime.info.world_size,
        "local_rank": runtime.info.local_rank,
        "is_main_process": runtime.info.is_main_process,
        "is_distributed": runtime.info.is_distributed,
        "reduced_sum": reduced,
    }
    with open(Path(output_dir) / f"metadata-rank{rank}.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    runtime.destroy()


def _gradient_sync_worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    runtime = DistributedRuntime(
        DistributedRuntimeConfig(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            master_addr="127.0.0.1",
            master_port=port,
            device="cpu",
        )
    )

    torch.manual_seed(100 + rank)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    model = runtime.prepare_model(model)

    inputs = torch.ones(6, 2) * float(rank + 1)
    targets = torch.ones(6, 1)
    runtime.zero_grad(optimizer)
    predictions = model(inputs)
    loss = nn.functional.mse_loss(predictions, targets)
    runtime.backward(loss)
    runtime.optimizer_step(optimizer)
    runtime.synchronize()

    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.linear.weight.detach().cpu(), Path(output_dir) / f"weight-rank{rank}.pt")
    runtime.destroy()


def _parallel_mode_worker(rank: int, world_size: int, port: int, output_dir: str, mode: str) -> None:
    runtime = create_parallel_runtime(
        mode,
        DistributedRuntimeConfig(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            master_addr="127.0.0.1",
            master_port=port,
            device="cpu",
        ),
    )
    payload = {
        "rank": rank,
        "mode": mode,
        "runtime_type": type(runtime).__name__,
        "using_fsdp": runtime.using_fsdp if isinstance(runtime, FSDPRuntime) else False,
    }
    with open(Path(output_dir) / f"mode-{mode}-rank{rank}.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    runtime.destroy()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_runtime_metadata_and_collectives(tmp_path: Path) -> None:
    port = _find_free_port()
    _launch_world(2, _metadata_worker, port, str(tmp_path))

    rank0 = json.loads((tmp_path / "metadata-rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((tmp_path / "metadata-rank1.json").read_text(encoding="utf-8"))
    assert rank0["world_size"] == 2
    assert rank1["world_size"] == 2
    assert rank0["rank"] == 0
    assert rank1["rank"] == 1
    assert rank0["is_main_process"] is True
    assert rank1["is_main_process"] is False
    assert rank0["reduced_sum"] == 3.0
    assert rank1["reduced_sum"] == 3.0


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_gradient_synchronization_keeps_parameters_aligned(tmp_path: Path) -> None:
    port = _find_free_port()
    _launch_world(2, _gradient_sync_worker, port, str(tmp_path))

    weight_rank0 = torch.load(tmp_path / "weight-rank0.pt", map_location="cpu", weights_only=False)
    weight_rank1 = torch.load(tmp_path / "weight-rank1.pt", map_location="cpu", weights_only=False)
    assert torch.allclose(weight_rank0, weight_rank1, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_parallel_runtime_factory_supports_ddp_and_fsdp_modes(tmp_path: Path) -> None:
    ddp_port = _find_free_port()
    _launch_world(2, _parallel_mode_worker, ddp_port, str(tmp_path), "ddp")
    fsdp_port = _find_free_port()
    _launch_world(2, _parallel_mode_worker, fsdp_port, str(tmp_path), "fsdp")

    ddp_rank0 = json.loads((tmp_path / "mode-ddp-rank0.json").read_text(encoding="utf-8"))
    fsdp_rank0 = json.loads((tmp_path / "mode-fsdp-rank0.json").read_text(encoding="utf-8"))
    assert ddp_rank0["runtime_type"] == "DDPRuntime"
    assert fsdp_rank0["runtime_type"] == "FSDPRuntime"
