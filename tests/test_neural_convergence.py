from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.phase_k_learning.core_network import CPUAutoregressiveEngine
from src.phase_k_learning.train_multi_cpu import (
    build_model,
    execute_training_cycle,
    initialize_hardware_environment,
    load_config,
)

SMALL_VOCAB = 32


def _tiny_config() -> dict:
    """A small config so the convergence check runs fast in CI."""
    return {
        "hardware_scaling": {"target_cpus_count": 1, "threads_per_core": 1, "pin_memory": False},
        "neural_dimensions": {
            "vocabulary_size": SMALL_VOCAB,
            "embedding_dim": 32,
            "hidden_dim": 64,
            "max_sequence_length": 16,
        },
        "optimization": {"initial_learning_rate": 0.01, "gradient_clip_norm": 1.0},
    }


def _repeating_stream(steps: int) -> list[tuple[list[int], list[int]]]:
    """A deterministic, perfectly learnable sequence: next token = (prev + 1) % V.

    If gradient descent is wired up correctly, the network must drive loss on
    this far below the random-initialization baseline of ln(V).
    """
    sequence = [i % SMALL_VOCAB for i in range(17)]
    inputs, targets = sequence[:-1], sequence[1:]
    return [(inputs, targets) for _ in range(steps)]


def test_config_yaml_loads_with_expected_sections() -> None:
    cfg = load_config()
    assert cfg["hardware_scaling"]["target_cpus_count"] >= 1
    assert cfg["neural_dimensions"]["vocabulary_size"] > 0
    assert cfg["optimization"]["gradient_clip_norm"] > 0


def test_hardware_environment_sets_requested_thread_count() -> None:
    cfg = _tiny_config()
    info = initialize_hardware_environment(cfg)
    assert info["allocated_physical_cpus"] == 1
    assert info["active_pytorch_threads"] == 1


def test_model_is_randomly_initialized_not_zeroed() -> None:
    """A zero-initialized network cannot learn (symmetric gradients), so verify
    the embedding table actually carries varied random weights."""
    model = CPUAutoregressiveEngine(vocab_size=SMALL_VOCAB, embedding_dim=32, hidden_dim=64)
    assert model.embedding.weight.abs().sum().item() > 0.0
    assert model.embedding.weight.std().item() > 0.0


def test_forward_pass_shapes_and_probability_mass() -> None:
    model = CPUAutoregressiveEngine(vocab_size=SMALL_VOCAB, embedding_dim=32, hidden_dim=64)
    inputs = torch.arange(8, dtype=torch.long).unsqueeze(0)
    logits, hidden = model(inputs)

    assert logits.shape == (1, 8, SMALL_VOCAB)
    assert hidden.shape == (1, 1, 64)

    probabilities = torch.softmax(logits, dim=-1)
    mass = probabilities.sum(dim=-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)


def test_initialization_loss_matches_log_vocab_baseline() -> None:
    """At random init the model has no information, so cross-entropy must sit
    near ln(V) -- the theoretical uniform-guess baseline."""
    cfg = _tiny_config()
    model = build_model(cfg)
    metrics = execute_training_cycle(model, _repeating_stream(1), cfg)

    expected = math.log(SMALL_VOCAB)
    assert abs(metrics["initial_loss"] - expected) < 0.75


def test_gradient_descent_actually_reduces_loss() -> None:
    """The core validator: real backpropagation must systematically drive the
    loss down from the ln(V) noise floor, not sit flat."""
    cfg = _tiny_config()
    torch.manual_seed(1337)
    model = build_model(cfg)
    metrics = execute_training_cycle(model, _repeating_stream(120), cfg)

    baseline = math.log(SMALL_VOCAB)
    assert metrics["initial_loss"] > metrics["final_loss"], "loss did not decrease at all"
    assert metrics["final_loss"] < baseline * 0.5, (
        f"loss failed to converge below half the random baseline "
        f"(final={metrics['final_loss']:.4f}, baseline={baseline:.4f})"
    )


def test_parameters_actually_change_after_training() -> None:
    """Guards against an optimizer that silently no-ops (e.g. detached graph)."""
    cfg = _tiny_config()
    torch.manual_seed(1337)
    model = build_model(cfg)
    before = model.output_head.weight.detach().clone()

    execute_training_cycle(model, _repeating_stream(10), cfg)

    after = model.output_head.weight.detach()
    assert not torch.allclose(before, after), "weights unchanged; gradient descent is not running"


def test_gradient_clipping_counter_is_reported() -> None:
    cfg = _tiny_config()
    cfg["optimization"]["gradient_clip_norm"] = 1e-6  # force clipping every step
    torch.manual_seed(1337)
    model = build_model(cfg)
    metrics = execute_training_cycle(model, _repeating_stream(5), cfg)

    assert metrics["gradient_clipping_trigger_count"] == 5


def test_throughput_metrics_are_measured() -> None:
    cfg = _tiny_config()
    model = build_model(cfg)
    metrics = execute_training_cycle(model, _repeating_stream(5), cfg)

    assert metrics["tokens_processed"] == 5 * 16
    assert metrics["elapsed_seconds"] > 0.0
    assert metrics["tokens_per_second"] > 0.0
