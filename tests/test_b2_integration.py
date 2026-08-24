from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.checkpoint import load_checkpoint, save_checkpoint
from outreachlm.data_loader_config import DataLoaderConfig, build_data_loader
from outreachlm.datasets import LanguageModelDataset
from outreachlm.runtime import SingleDeviceRuntime
from outreachlm.telemetry import TelemetryConfig, TrainingTelemetry
from outreachlm.trainer_core import Trainer


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, embedding_dim: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.output(hidden)


def _cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, vocab_size = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(batch_size * sequence_length, vocab_size),
        targets.reshape(batch_size * sequence_length),
    )


def _cycle_loader(loader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _make_loader(*, num_workers: int) -> object:
    token_ids = torch.arange(0, 1024, dtype=torch.long) % 32
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=16)
    config = DataLoaderConfig(
        batch_size=4,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )
    return build_data_loader(dataset, config)


def _run_scenario(
    *,
    tmp_path: Path,
    name: str,
    precision: str,
    accumulation: int,
    num_workers: int,
    steps: int,
) -> list[dict]:
    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    runtime = SingleDeviceRuntime("cpu", precision=precision)
    telemetry = TrainingTelemetry(
        TelemetryConfig(output_dir=tmp_path / name / "telemetry", run_name=name)
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_cross_entropy_loss,
        runtime=runtime,
        gradient_accumulation_steps=accumulation,
        telemetry=telemetry,
    )
    outputs = trainer.run_steps(
        _cycle_loader(_make_loader(num_workers=num_workers)),
        steps=steps,
        flush_partial_accumulation=True,
    )
    return outputs


def test_b2_integration_matrix_smoke(tmp_path: Path) -> None:
    baseline = _run_scenario(
        tmp_path=tmp_path,
        name="baseline",
        precision="fp32",
        accumulation=1,
        num_workers=0,
        steps=4,
    )
    accumulation = _run_scenario(
        tmp_path=tmp_path,
        name="accumulation",
        precision="fp32",
        accumulation=2,
        num_workers=0,
        steps=4,
    )
    mixed_precision = _run_scenario(
        tmp_path=tmp_path,
        name="mixed-precision",
        precision="bf16",
        accumulation=1,
        num_workers=0,
        steps=4,
    )

    assert baseline[-1]["throughput"]["tokens_processed"] > 0
    assert accumulation[-1]["optimizer_step"] == 2
    assert mixed_precision[-1]["memory"]["device_type"] == "cpu"

    for scenario_name in ("baseline", "accumulation", "mixed-precision"):
        assert (tmp_path / scenario_name / "telemetry" / "metrics.jsonl").exists()
        assert (tmp_path / scenario_name / "telemetry" / "memory.jsonl").exists()
        assert (tmp_path / scenario_name / "telemetry" / "events.jsonl").exists()
        assert (tmp_path / scenario_name / "telemetry" / "summary.json").exists()


def test_b2_checkpoint_resume_and_loader_worker_modes(tmp_path: Path) -> None:
    outputs_workers_0 = _run_scenario(
        tmp_path=tmp_path,
        name="workers-0",
        precision="fp32",
        accumulation=1,
        num_workers=0,
        steps=2,
    )
    outputs_workers_1 = _run_scenario(
        tmp_path=tmp_path,
        name="workers-1",
        precision="fp32",
        accumulation=1,
        num_workers=1,
        steps=2,
    )
    assert outputs_workers_0[-1]["throughput"]["tokens_per_second"] > 0.0
    assert outputs_workers_1[-1]["throughput"]["tokens_per_second"] > 0.0

    model = TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    runtime = SingleDeviceRuntime("cpu", precision="fp32")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    telemetry = TrainingTelemetry(
        TelemetryConfig(output_dir=tmp_path / "resume" / "telemetry", run_name="resume")
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=_cross_entropy_loss,
        runtime=runtime,
        telemetry=telemetry,
    )
    loader_iter = _cycle_loader(_make_loader(num_workers=0))
    trainer.run_steps(loader_iter, steps=3)

    checkpoint_path = tmp_path / "resume" / "checkpoint.pt"
    trainer_payload = trainer.state_dict()
    save_checkpoint(
        checkpoint_path,
        trainer.model,
        trainer.optimizer,
        step=trainer.state.step,
        train_loss=0.0,
        best_validation_loss=0.0,
        config={"phase": "b2.10"},
        scheduler_state=trainer_payload["scheduler_state"],
        scaler_state=trainer_payload["scaler_state"],
        trainer_state=trainer_payload["trainer_state"],
        metadata={"scenario": "resume"},
    )

    resumed_model = TinyLM()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=5e-4)
    resumed_runtime = SingleDeviceRuntime("cpu", precision="fp32")
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer,
        step_size=1,
        gamma=0.99,
    )
    resumed_telemetry = TrainingTelemetry(
        TelemetryConfig(output_dir=tmp_path / "resume" / "telemetry-resumed", run_name="resume-2")
    )
    resumed_trainer = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        loss_fn=_cross_entropy_loss,
        runtime=resumed_runtime,
        telemetry=resumed_telemetry,
    )
    checkpoint_state = load_checkpoint(
        checkpoint_path,
        resumed_trainer.model,
        resumed_trainer.optimizer,
        torch.device("cpu"),
        scheduler=resumed_scheduler,
        runtime=resumed_runtime,
    )
    resumed_trainer.load_state_dict(
        {
            "trainer_state": checkpoint_state["trainer_state"],
            "scheduler_state": checkpoint_state["scheduler_state"],
            "scaler_state": checkpoint_state["scaler_state"],
        }
    )

    resumed_trainer.run_steps(_cycle_loader(_make_loader(num_workers=0)), steps=5)
    assert resumed_trainer.state.step == 8
    assert resumed_trainer.state.total_tokens_processed > 0
    assert (tmp_path / "resume" / "telemetry-resumed" / "summary.json").exists()
