from pathlib import Path

import torch
import torch.nn as nn

from outreachlm.checkpoint import load_checkpoint, save_checkpoint
from outreachlm.data_pipeline import ResumableShardedBatchSource
from outreachlm.datasets import LanguageModelDataset
from outreachlm.runtime import SingleDeviceRuntime
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


def _make_dataset() -> LanguageModelDataset:
    token_ids = (torch.arange(0, 420, dtype=torch.long) % 32).clone()
    return LanguageModelDataset(token_ids=token_ids, context_length=8)


def _train_steps(trainer: Trainer, source: ResumableShardedBatchSource, steps: int) -> None:
    for _ in range(steps):
        trainer.train_step(source.next_batch())


def test_fault_tolerance_resume_preserves_data_position_and_deterministic_progression(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "fault-tolerance.pt"

    torch.manual_seed(99)
    continuous_model = TinyLM()
    continuous_optimizer = torch.optim.AdamW(continuous_model.parameters(), lr=1e-3)
    continuous_runtime = SingleDeviceRuntime("cpu")
    continuous_trainer = Trainer(
        model=continuous_model,
        optimizer=continuous_optimizer,
        loss_fn=_loss,
        runtime=continuous_runtime,
    )
    continuous_source = ResumableShardedBatchSource(
        _make_dataset(),
        batch_size=4,
        shuffle=True,
        seed=123,
        sequence_packing=2,
    )
    _train_steps(continuous_trainer, continuous_source, steps=8)
    continuous_weight = continuous_model.output.weight.detach().clone()

    torch.manual_seed(99)
    split_model = TinyLM()
    split_optimizer = torch.optim.AdamW(split_model.parameters(), lr=1e-3)
    split_runtime = SingleDeviceRuntime("cpu")
    split_trainer = Trainer(model=split_model, optimizer=split_optimizer, loss_fn=_loss, runtime=split_runtime)
    split_source = ResumableShardedBatchSource(
        _make_dataset(),
        batch_size=4,
        shuffle=True,
        seed=123,
        sequence_packing=2,
    )
    _train_steps(split_trainer, split_source, steps=5)
    split_state = split_trainer.state_dict()
    save_checkpoint(
        checkpoint_path,
        split_trainer.model,
        split_trainer.optimizer,
        step=split_trainer.state.step,
        train_loss=0.0,
        best_validation_loss=0.0,
        config={"phase": "f9"},
        trainer_state={
            **split_state["trainer_state"],
            "data_position": split_source.state_dict(),
        },
        scheduler_state=split_state["scheduler_state"],
        scaler_state=split_state["scaler_state"],
    )

    resumed_model = TinyLM()
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_runtime = SingleDeviceRuntime("cpu")
    resumed_trainer = Trainer(
        model=resumed_model,
        optimizer=resumed_optimizer,
        loss_fn=_loss,
        runtime=resumed_runtime,
    )
    checkpoint_state = load_checkpoint(
        checkpoint_path,
        resumed_trainer.model,
        resumed_trainer.optimizer,
        torch.device("cpu"),
    )
    resumed_trainer.load_state_dict(
        {
            "trainer_state": checkpoint_state["trainer_state"],
            "scheduler_state": checkpoint_state["scheduler_state"],
            "scaler_state": checkpoint_state["scaler_state"],
        }
    )
    resumed_source = ResumableShardedBatchSource(
        _make_dataset(),
        batch_size=4,
        shuffle=True,
        seed=123,
        sequence_packing=2,
    )
    resumed_source.load_state_dict(checkpoint_state["trainer_state"]["data_position"])
    _train_steps(resumed_trainer, resumed_source, steps=3)

    resumed_raw_model = resumed_trainer.model.module if hasattr(resumed_trainer.model, "module") else resumed_trainer.model
    assert torch.allclose(
        resumed_raw_model.output.weight.detach(),
        continuous_weight,
        atol=1e-6,
        rtol=1e-6,
    )
