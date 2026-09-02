from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.phase_k_learning.core_network import CPUAutoregressiveEngine

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

TokenBatch = tuple[Sequence[int], Sequence[int]]


def load_config(path: Path | str | None = None) -> dict:
    target = Path(path) if path is not None else CONFIG_PATH
    with open(target, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def initialize_hardware_environment(config: dict) -> dict:
    """Pin PyTorch's internal execution loops to the configured core allocation.

    Important caveat about the BLAS environment variables: OMP_NUM_THREADS /
    MKL_NUM_THREADS / OPENBLAS_NUM_THREADS are read by those libraries when
    they are first loaded, which happens at `import torch`. Setting them here
    (after import) will therefore NOT retroactively repartition an already
    initialized OpenMP pool. They are still set so that any subprocess spawned
    from this process inherits the intended allocation, and
    `torch.set_num_threads` -- which does take effect at runtime -- is the
    mechanism actually controlling this process's intra-op parallelism. To
    control OpenMP for this process itself, export the variables in the shell
    before launching Python.
    """
    cores = int(config["hardware_scaling"]["target_cpus_count"])

    torch.set_num_threads(cores)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Inter-op pool is already initialized (e.g. a previous run in this
        # same process); this is not fatal, the existing setting stands.
        pass

    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)

    active = torch.get_num_threads()
    print(f"[HARDWARE] PyTorch intra-op threads requested={cores} active={active}")
    return {
        "allocated_physical_cpus": cores,
        "active_pytorch_threads": active,
        "host_logical_processors": os.cpu_count(),
    }


def execute_training_cycle(
    model: CPUAutoregressiveEngine,
    token_stream_batches: Iterable[TokenBatch],
    config: dict,
    *,
    optimizer: optim.Optimizer | None = None,
    max_steps: int | None = None,
) -> dict:
    """Execute real gradient descent optimization over the incoming token stream.

    The loss is position-weighted: the FINAL target position of each sequence
    (the answer token the evaluation scores) is multiplied by
    `optimization.answer_loss_weight`. Under uniform weighting that token
    carried ~1.29% of the gradient signal, so the network provably converged
    to a "learned all the boilerplate, learned nothing about the answer"
    floor. Answer-position and context-position losses are tracked separately
    so convergence on the task of interest is directly observable rather than
    inferred from an aggregate number.

    Returns a metrics dict rather than only the mean loss, so callers can log
    the true first-step (initialization) loss, the final loss, the number of
    times gradient clipping actually engaged, and throughput -- all measured,
    not assumed.
    """
    model.train()
    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=config["optimization"]["initial_learning_rate"])
    loss_criterion = nn.CrossEntropyLoss(reduction="none")
    clip_value = float(config["optimization"]["gradient_clip_norm"])
    answer_weight = float(config["optimization"].get("answer_loss_weight", 1.0))

    losses: list[float] = []
    answer_losses: list[float] = []
    context_losses: list[float] = []
    clip_trigger_count = 0
    tokens_processed = 0
    started = time.perf_counter()

    for step, (input_sequence, target_sequence) in enumerate(token_stream_batches):
        if max_steps is not None and step >= max_steps:
            break

        inputs = torch.as_tensor(input_sequence, dtype=torch.long)
        targets = torch.as_tensor(target_sequence, dtype=torch.long)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            targets = targets.unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(inputs)
        per_token = loss_criterion(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        ).view(targets.shape)

        weights = torch.ones_like(per_token)
        weights[:, -1] = answer_weight
        loss = (per_token * weights).sum() / weights.sum()

        loss.backward()

        # clip_grad_norm_ returns the PRE-clipping total norm, so comparing it
        # against the threshold is what actually tells us whether clipping
        # engaged on this step.
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
        if float(total_norm) > clip_value:
            clip_trigger_count += 1

        optimizer.step()

        losses.append(float(loss.item()))
        answer_losses.append(float(per_token[:, -1].mean().item()))
        if per_token.shape[1] > 1:
            context_losses.append(float(per_token[:, :-1].mean().item()))
        tokens_processed += int(targets.numel())

    elapsed = time.perf_counter() - started
    if not losses:
        raise ValueError("No training batches were provided.")

    def _trailing(values: list[float]) -> float:
        if not values:
            return 0.0
        window = values[-50:]
        return sum(window) / len(window)

    return {
        "steps": len(losses),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "trailing_mean_loss": _trailing(losses),
        "losses": losses,
        "initial_answer_loss": answer_losses[0],
        "final_answer_loss": answer_losses[-1],
        "trailing_mean_answer_loss": _trailing(answer_losses),
        "trailing_mean_context_loss": _trailing(context_losses),
        "gradient_clipping_trigger_count": clip_trigger_count,
        "tokens_processed": tokens_processed,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (tokens_processed / elapsed) if elapsed > 0 else 0.0,
    }


def build_model(config: dict) -> CPUAutoregressiveEngine:
    dims = config["neural_dimensions"]
    return CPUAutoregressiveEngine(
        vocab_size=int(dims["vocabulary_size"]),
        embedding_dim=int(dims["embedding_dim"]),
        hidden_dim=int(dims["hidden_dim"]),
    )


def main() -> None:
    cfg = load_config()
    initialize_hardware_environment(cfg)
    outreach_model = build_model(cfg)
    print(
        "[VERIFICATION] OutreachLM Real Neural Training Core Online. "
        f"parameters={outreach_model.parameter_count():,}. Awaiting stream pipelines..."
    )


if __name__ == "__main__":
    main()
