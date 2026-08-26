import json
import multiprocessing as mp
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from outreachlm.checkpoint import load_checkpoint, save_checkpoint
from outreachlm.data_loader_config import DataLoaderConfig, build_data_loader
from outreachlm.datasets import LanguageModelDataset
from outreachlm.model_config import DenseTransformerConfig
from outreachlm.runtime import DistributedRuntime, DistributedRuntimeConfig
from outreachlm.scalable_model import ScalableTransformerModel
from outreachlm.telemetry import TelemetryConfig, TrainingTelemetry
from outreachlm.trainer_core import Trainer


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
        process.join(timeout=120)
        assert process.exitcode == 0


def _cycle_loader(loader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _moe_loss(model: ScalableTransformerModel, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    language_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return model.combine_with_moe_loss(language_loss)


def _moe_worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    runtime = DistributedRuntime(
        DistributedRuntimeConfig(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            master_addr="127.0.0.1",
            master_port=port,
            device="cpu",
            find_unused_parameters=True,
        )
    )
    cfg = DenseTransformerConfig(
        vocab_size=64,
        context_length=8,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        moe_enabled=True,
        num_experts=2,
        top_k=2,
        load_balancing_weight=0.02,
        capacity_factor=1.0,
    )
    model = ScalableTransformerModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    telemetry = TrainingTelemetry(
        TelemetryConfig(
            output_dir=Path(output_dir) / "telemetry",
            run_name="phase-e",
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            is_main_process=(rank == 0),
        )
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=lambda logits, targets: _moe_loss(model, logits, targets),
        runtime=runtime,
        telemetry=telemetry,
    )

    token_ids = torch.arange(0, 260, dtype=torch.long) % cfg.vocab_size
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=cfg.context_length)
    loader = build_data_loader(
        dataset,
        DataLoaderConfig(batch_size=4, shuffle=False),
        distributed_rank=rank,
        distributed_world_size=world_size,
    )
    trainer.run_steps(_cycle_loader(loader), steps=3)

    checkpoint_path = Path(output_dir) / "moe-checkpoint.pt"
    state = trainer.state_dict()
    save_checkpoint(
        checkpoint_path,
        trainer.model,
        trainer.optimizer,
        step=trainer.state.step,
        train_loss=0.0,
        best_validation_loss=0.0,
        config={"moe_enabled": True},
        scheduler_state=state["scheduler_state"],
        scaler_state=state["scaler_state"],
        trainer_state=state["trainer_state"],
        metadata={"phase": "E"},
        runtime=runtime,
    )

    resumed_model = ScalableTransformerModel(cfg)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.99)
    resumed_trainer = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        loss_fn=lambda logits, targets: _moe_loss(resumed_model, logits, targets),
        runtime=runtime,
    )
    checkpoint_state = load_checkpoint(
        checkpoint_path,
        resumed_trainer.model,
        resumed_trainer.optimizer,
        torch.device("cpu"),
        scheduler=resumed_scheduler,
        runtime=runtime,
    )
    resumed_trainer.load_state_dict(
        {
            "trainer_state": checkpoint_state["trainer_state"],
            "scheduler_state": checkpoint_state["scheduler_state"],
            "scaler_state": checkpoint_state["scaler_state"],
        }
    )
    resumed_trainer.run_steps(_cycle_loader(loader), steps=2)

    raw_model = resumed_trainer.model.module if hasattr(resumed_trainer.model, "module") else resumed_trainer.model
    torch.save(raw_model.token_embedding.weight.detach().cpu(), Path(output_dir) / f"moe-weight-rank{rank}.pt")
    with open(Path(output_dir) / f"moe-result-rank{rank}.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "rank": rank,
                "step": resumed_trainer.state.step,
                "moe_layers_with_stats": len(raw_model.last_moe_stats),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    runtime.destroy()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_phase_e_end_to_end_dense_and_moe_distributed_resume(tmp_path: Path) -> None:
    port = _find_free_port()
    _launch_world(2, _moe_worker, port, str(tmp_path))

    assert (tmp_path / "moe-checkpoint.pt").exists()
    rank0 = json.loads((tmp_path / "moe-result-rank0.json").read_text(encoding="utf-8"))
    rank1 = json.loads((tmp_path / "moe-result-rank1.json").read_text(encoding="utf-8"))
    assert rank0["step"] == rank1["step"]
    assert rank0["moe_layers_with_stats"] > 0

    weight_rank0 = torch.load(tmp_path / "moe-weight-rank0.pt", map_location="cpu", weights_only=False)
    weight_rank1 = torch.load(tmp_path / "moe-weight-rank1.pt", map_location="cpu", weights_only=False)
    assert torch.allclose(weight_rank0, weight_rank1, atol=1e-5, rtol=1e-5)
    assert (tmp_path / "telemetry" / "metrics-rank0.jsonl").exists()
    assert (tmp_path / "telemetry" / "metrics-rank1.jsonl").exists()
