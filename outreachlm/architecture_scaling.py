from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F

from outreachlm.architecture_profiler import profile_architecture
from outreachlm.model_config import DenseTransformerConfig
from outreachlm.runtime import SingleDeviceRuntime
from outreachlm.scalable_model import ScalableTransformerModel


@dataclass(frozen=True)
class ScalingSpec:
    name: str
    config: DenseTransformerConfig
    steps: int = 4
    batch_size: int = 2


def run_architecture_scaling_experiments(specs: list[ScalingSpec], seed: int = 42) -> list[dict]:
    if not specs:
        raise ValueError("specs must not be empty.")

    results: list[dict] = []
    runtime = SingleDeviceRuntime("cpu")
    for spec in specs:
        torch.manual_seed(seed)
        model = runtime.prepare_model(ScalableTransformerModel(spec.config))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

        losses: list[float] = []
        start = perf_counter()
        for _ in range(spec.steps):
            input_ids = torch.randint(
                low=0,
                high=spec.config.vocab_size,
                size=(spec.batch_size, spec.config.context_length),
                device=runtime.info.device,
            )
            target_ids = torch.randint(
                low=0,
                high=spec.config.vocab_size,
                size=(spec.batch_size, spec.config.context_length),
                device=runtime.info.device,
            )
            runtime.zero_grad(optimizer)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, spec.config.vocab_size),
                target_ids.reshape(-1),
            )
            runtime.backward(loss)
            runtime.optimizer_step(optimizer)
            losses.append(float(loss.item()))
        elapsed = max(perf_counter() - start, 1e-12)

        tokens_processed = spec.steps * spec.batch_size * spec.config.context_length
        profile = profile_architecture(
            spec.config,
            per_device_batch_size=spec.batch_size,
            gradient_accumulation_steps=1,
            world_size=1,
        )
        memory = runtime.collect_memory_stats(model, optimizer)
        results.append(
            {
                "name": spec.name,
                "parameter_count": profile.total_parameters,
                "train_loss_first": losses[0],
                "train_loss_last": losses[-1],
                "validation_loss_proxy": losses[-1],
                "tokens_per_second": tokens_processed / elapsed,
                "memory": memory,
                "flops_per_token": profile.flops_per_token,
                "flops_per_step": profile.flops_per_step,
            }
        )
    return results
