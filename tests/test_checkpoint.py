from pathlib import Path

import pytest
import torch

from outreachlm.checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint


def _make_model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_training_checkpoint_round_trip(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=12,
        train_loss=1.23,
        best_validation_loss=0.75,
        config={"seed": 42, "steps": 100},
        trainer_state={"global_tokens": 2048},
        metadata={"run_name": "b2.2-roundtrip"},
    )

    raw_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert raw_payload["checkpoint_version"] == CHECKPOINT_VERSION
    assert "model_state" in raw_payload
    assert "optimizer_state" in raw_payload
    assert "trainer_state" in raw_payload
    assert "rng_state" in raw_payload
    assert "config" in raw_payload
    assert "metadata" in raw_payload
    assert raw_payload["metadata"]["format"] == "training_checkpoint"

    restored_model, restored_optimizer = _make_model_and_optimizer()
    state = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
    )
    assert state["step"] == 12
    assert state["train_loss"] == 1.23
    assert state["best_validation_loss"] == 0.75
    assert state["config"]["seed"] == 42
    assert state["trainer_state"]["global_tokens"] == 2048
    assert state["metadata"]["run_name"] == "b2.2-roundtrip"
    assert state["is_legacy"] is False


def test_load_checkpoint_supports_legacy_v2_payload(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    checkpoint_path = tmp_path / "legacy_v2.pt"
    legacy_payload = {
        "checkpoint_version": 2,
        "step": 7,
        "train_loss": 2.5,
        "best_validation_loss": 1.1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {"seed": 77},
    }
    torch.save(legacy_payload, checkpoint_path)

    restored_model, restored_optimizer = _make_model_and_optimizer()
    state = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
    )
    assert state["step"] == 7
    assert state["train_loss"] == 2.5
    assert state["best_validation_loss"] == 1.1
    assert state["config"]["seed"] == 77
    assert state["is_legacy"] is True


def test_load_checkpoint_rejects_missing_required_fields(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "broken_v3.pt"
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_state": {},
            "optimizer_state": {},
            "trainer_state": {"step": 1},
            "rng_state": {},
            "config": {},
        },
        checkpoint_path,
    )

    model, optimizer = _make_model_and_optimizer()
    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(checkpoint_path, model, optimizer, torch.device("cpu"))


def _train_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    steps: int,
) -> float:
    last_loss = 0.0
    for _ in range(steps):
        inputs = torch.randn(6, 4)
        targets = torch.randn(6, 3)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    return last_loss


def test_resume_equivalence_continuous_vs_checkpoint_restart(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.pt"

    torch.manual_seed(1234)
    continuous_model, continuous_optimizer = _make_model_and_optimizer()
    continuous_last_loss = _train_steps(continuous_model, continuous_optimizer, steps=10)
    continuous_weight = continuous_model.weight.detach().clone()
    continuous_bias = continuous_model.bias.detach().clone()

    torch.manual_seed(1234)
    split_model, split_optimizer = _make_model_and_optimizer()
    split_first_loss = _train_steps(split_model, split_optimizer, steps=5)
    save_checkpoint(
        checkpoint_path,
        split_model,
        split_optimizer,
        step=5,
        train_loss=split_first_loss,
        best_validation_loss=split_first_loss,
        config={"seed": 1234, "steps": 10},
        trainer_state={"optimizer_step": 5, "micro_step": 5},
        metadata={"resume_test": True},
    )

    resumed_model, resumed_optimizer = _make_model_and_optimizer()
    restored = load_checkpoint(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        torch.device("cpu"),
    )
    assert restored["step"] == 5
    resumed_last_loss = _train_steps(resumed_model, resumed_optimizer, steps=5)

    assert torch.allclose(
        resumed_model.weight.detach(),
        continuous_weight,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        resumed_model.bias.detach(),
        continuous_bias,
        atol=1e-6,
        rtol=1e-6,
    )
    assert abs(resumed_last_loss - continuous_last_loss) < 1e-6
