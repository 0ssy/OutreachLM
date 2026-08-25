import json
import multiprocessing as mp
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from outreachlm.checkpoint import load_checkpoint, save_checkpoint
from outreachlm.data_loader_config import DataLoaderConfig, build_data_loader
from outreachlm.datasets import LanguageModelDataset
from outreachlm.runtime import (
    DistributedRuntime,
    DistributedRuntimeConfig,
    SingleDeviceRuntime,
)
from outreachlm.telemetry import TelemetryConfig, TrainingTelemetry
from outreachlm.trainer_core import Trainer
from outreachlm.train import evaluate_validation


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


def _cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, vocab_size = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(batch_size * sequence_length, vocab_size),
        targets.reshape(batch_size * sequence_length),
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_dataset() -> LanguageModelDataset:
    token_ids = (torch.arange(0, 260, dtype=torch.long) % 32).clone()
    return LanguageModelDataset(token_ids=token_ids, context_length=4)


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


def _distributed_resume_worker(rank: int, world_size: int, port: int, output_dir: str) -> None:
    runtime = DistributedRuntime(
        DistributedRuntimeConfig(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            local_rank=rank,
            master_addr="127.0.0.1",
            master_port=port,
            device="cpu",
            precision="fp32",
        )
    )

    dataset = _make_dataset()
    loader_config = DataLoaderConfig(batch_size=4, shuffle=False, num_workers=0)
    loader = build_data_loader(
        dataset,
        loader_config,
        distributed_rank=rank,
        distributed_world_size=world_size,
    )

    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    telemetry = TrainingTelemetry(
        TelemetryConfig(
            output_dir=Path(output_dir) / "telemetry",
            run_name="phase-c-resume",
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            is_main_process=(rank == 0),
        )
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=_cross_entropy_loss,
        runtime=runtime,
        telemetry=telemetry,
    )
    trainer.run_steps(_cycle_loader(loader), steps=3)

    checkpoint_path = Path(output_dir) / "distributed-checkpoint.pt"
    trainer_payload = trainer.state_dict()
    save_checkpoint(
        checkpoint_path,
        trainer.model,
        trainer.optimizer,
        step=trainer.state.step,
        train_loss=0.0,
        best_validation_loss=0.0,
        config={"phase": "c"},
        scheduler_state=trainer_payload["scheduler_state"],
        scaler_state=trainer_payload["scaler_state"],
        trainer_state=trainer_payload["trainer_state"],
        metadata={"phase": "c-resume"},
        runtime=runtime,
    )

    resumed_model = TinyLM()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer,
        step_size=1,
        gamma=0.99,
    )
    resumed_telemetry = TrainingTelemetry(
        TelemetryConfig(
            output_dir=Path(output_dir) / "telemetry-resumed",
            run_name="phase-c-resume-2",
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            is_main_process=(rank == 0),
        )
    )
    resumed_trainer = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        loss_fn=_cross_entropy_loss,
        runtime=runtime,
        telemetry=resumed_telemetry,
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
    resumed_outputs = resumed_trainer.run_steps(_cycle_loader(loader), steps=4)

    eval_loss, _ = evaluate_validation(
        resumed_trainer.model,
        dataset,
        runtime.info.device,
        batch_size=4,
        eval_loader_config=loader_config,
        runtime=runtime,
    )
    raw_model = resumed_trainer.model.module if hasattr(resumed_trainer.model, "module") else resumed_trainer.model
    torch.save(raw_model.output.weight.detach().cpu(), Path(output_dir) / f"weight-rank{rank}.pt")
    with open(Path(output_dir) / f"result-rank{rank}.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "rank": rank,
                "step": resumed_trainer.state.step,
                "eval_loss": eval_loss,
                "global_tokens_per_second": resumed_outputs[-1]["throughput"]["global_tokens_per_second"],
                "local_tokens_per_second": resumed_outputs[-1]["throughput"]["tokens_per_second"],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    runtime.destroy()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_phase_c10_single_device_regression_still_runs(tmp_path: Path) -> None:
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = SingleDeviceRuntime("cpu")
    telemetry = TrainingTelemetry(
        TelemetryConfig(output_dir=tmp_path / "single-telemetry", run_name="single")
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_cross_entropy_loss,
        runtime=runtime,
        telemetry=telemetry,
    )
    loader = build_data_loader(_make_dataset(), DataLoaderConfig(batch_size=4, shuffle=False, num_workers=0))
    outputs = trainer.run_steps(_cycle_loader(loader), steps=3)
    assert outputs[-1]["throughput"]["tokens_processed"] > 0
    assert (tmp_path / "single-telemetry" / "summary.json").exists()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_phase_c10_distributed_smoke_resume_eval_and_telemetry(tmp_path: Path) -> None:
    port = _find_free_port()
    _launch_world(2, _distributed_resume_worker, port, str(tmp_path))

    checkpoint_path = tmp_path / "distributed-checkpoint.pt"
    assert checkpoint_path.exists()

    weight_rank0 = torch.load(tmp_path / "weight-rank0.pt", map_location="cpu", weights_only=False)
    weight_rank1 = torch.load(tmp_path / "weight-rank1.pt", map_location="cpu", weights_only=False)
    assert torch.allclose(weight_rank0, weight_rank1, atol=1e-5, rtol=1e-5)

    result_rank0 = json.loads((tmp_path / "result-rank0.json").read_text(encoding="utf-8"))
    result_rank1 = json.loads((tmp_path / "result-rank1.json").read_text(encoding="utf-8"))
    assert result_rank0["step"] == result_rank1["step"]
    assert abs(result_rank0["eval_loss"] - result_rank1["eval_loss"]) < 1e-8
    assert result_rank0["global_tokens_per_second"] >= result_rank0["local_tokens_per_second"]

    assert (tmp_path / "telemetry" / "metrics-rank0.jsonl").exists()
    assert (tmp_path / "telemetry" / "metrics-rank1.jsonl").exists()
    assert (tmp_path / "telemetry-resumed" / "summary-rank0.json").exists()
    assert (tmp_path / "telemetry-resumed" / "summary-rank1.json").exists()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_phase_c9_performance_benchmark_reports_scaling_efficiency(tmp_path: Path) -> None:
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = SingleDeviceRuntime("cpu")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_cross_entropy_loss,
        runtime=runtime,
    )
    loader = build_data_loader(_make_dataset(), DataLoaderConfig(batch_size=4, shuffle=False, num_workers=0))
    single_outputs = trainer.run_steps(_cycle_loader(loader), steps=4)
    single_throughput = float(single_outputs[-1]["throughput"]["tokens_per_second"])

    port = _find_free_port()
    _launch_world(2, _distributed_resume_worker, port, str(tmp_path / "perf"))
    distributed_result = json.loads(
        (tmp_path / "perf" / "result-rank0.json").read_text(encoding="utf-8")
    )
    distributed_throughput = float(distributed_result["global_tokens_per_second"])
    efficiency = distributed_throughput / max(single_throughput * 2.0, 1e-12)

    assert single_throughput > 0.0
    assert distributed_throughput > 0.0
    assert efficiency > 0.0
