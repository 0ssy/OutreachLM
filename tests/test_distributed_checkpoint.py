import json
import multiprocessing as mp
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from outreachlm.checkpoint import load_distributed_checkpoint, save_distributed_checkpoint
from outreachlm.data_loader_config import DataLoaderConfig, build_data_loader
from outreachlm.datasets import LanguageModelDataset
from outreachlm.runtime import DistributedRuntime, DistributedRuntimeConfig
from outreachlm.trainer_core import Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    b, t, v = logits.shape
    return nn.functional.cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t))


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cycle_loader(loader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _launch_world(world_size: int, worker, *args) -> None:
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for rank in range(world_size):
        process = context.Process(target=worker, args=(rank, world_size, *args))
        process.start()
        processes.append(process)
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0


def _worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
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
    token_ids = (torch.arange(0, 260, dtype=torch.long) % 32).clone()
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=8)
    loader = build_data_loader(
        dataset,
        DataLoaderConfig(batch_size=4, shuffle=False),
        distributed_rank=rank,
        distributed_world_size=world_size,
    )

    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=_loss, runtime=runtime)
    trainer.run_steps(_cycle_loader(loader), steps=3)
    state = trainer.state_dict()

    checkpoint_dir = Path(output_dir) / "dist-checkpoint"
    save_distributed_checkpoint(
        checkpoint_dir,
        trainer.model,
        trainer.optimizer,
        step=trainer.state.step,
        train_loss=0.0,
        best_validation_loss=0.0,
        config={"phase": "f6"},
        trainer_state={**state["trainer_state"], "data_position": {"epoch": 0, "position": 12}},
        scheduler_state=state["scheduler_state"],
        scaler_state=state["scaler_state"],
        metadata={"stage": "phase-f"},
        runtime=runtime,
    )

    restored_model = TinyLM()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.99)
    restored_trainer = Trainer(
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        loss_fn=_loss,
        runtime=runtime,
    )
    restored = load_distributed_checkpoint(
        checkpoint_dir,
        restored_trainer.model,
        restored_trainer.optimizer,
        torch.device("cpu"),
        scheduler=restored_scheduler,
        runtime=runtime,
    )
    restored_trainer.load_state_dict(
        {
            "trainer_state": restored["trainer_state"],
            "scheduler_state": restored["scheduler_state"],
            "scaler_state": restored["scaler_state"],
        }
    )
    restored_trainer.run_steps(_cycle_loader(loader), steps=2)
    raw_model = restored_trainer.model.module if hasattr(restored_trainer.model, "module") else restored_trainer.model
    torch.save(raw_model.output.weight.detach().cpu(), Path(output_dir) / f"dist-weight-rank{rank}.pt")
    with open(Path(output_dir) / f"dist-state-rank{rank}.json", "w", encoding="utf-8") as file:
        json.dump({"rank": rank, "step": restored_trainer.state.step, "data_position": restored["trainer_state"]["data_position"]}, file, ensure_ascii=False, indent=2)
    runtime.destroy()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_checkpoint_shards_and_restores_state(tmp_path: Path) -> None:
    port = _find_free_port()
    _launch_world(2, _worker, port, str(tmp_path))
    checkpoint_dir = tmp_path / "dist-checkpoint"
    assert (checkpoint_dir / "metadata.json").exists()
    assert (checkpoint_dir / "model" / "shard-rank0.pt").exists()
    assert (checkpoint_dir / "model" / "shard-rank1.pt").exists()
    assert (checkpoint_dir / "optimizer" / "shard-rank0.pt").exists()
    assert (checkpoint_dir / "training_state" / "shard-rank1.pt").exists()

    weight_rank0 = torch.load(tmp_path / "dist-weight-rank0.pt", map_location="cpu", weights_only=False)
    weight_rank1 = torch.load(tmp_path / "dist-weight-rank1.pt", map_location="cpu", weights_only=False)
    assert torch.allclose(weight_rank0, weight_rank1, atol=1e-5, rtol=1e-5)

    state_rank0 = json.loads((tmp_path / "dist-state-rank0.json").read_text(encoding="utf-8"))
    state_rank1 = json.loads((tmp_path / "dist-state-rank1.json").read_text(encoding="utf-8"))
    assert state_rank0["step"] == state_rank1["step"]
    assert state_rank0["data_position"]["position"] == 12
