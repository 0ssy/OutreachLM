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
        load_balance_losses: list[float] = []
        utilization_snapshots: list[list[float]] = []
        overflow_ratios: list[float] = []
        dropped_ratios: list[float] = []
        routing_entropy_values: list[float] = []
        expert_balance_values: list[float] = []
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
            language_loss = F.cross_entropy(
                logits.reshape(-1, spec.config.vocab_size),
                target_ids.reshape(-1),
            )
            if hasattr(model, "combine_with_moe_loss"):
                loss = model.combine_with_moe_loss(language_loss)
            else:
                loss = language_loss
            runtime.backward(loss)
            runtime.optimizer_step(optimizer)
            losses.append(float(loss.item()))
            load_balance_losses.append(float(getattr(model, "last_moe_load_balancing_loss", torch.tensor(0.0)).item()))
            stats = getattr(model, "last_moe_stats", [])
            if stats:
                utilization = [sum(layer.expert_utilization[i] for layer in stats) / len(stats) for i in range(len(stats[0].expert_utilization))]
                utilization_snapshots.append(utilization)
                tokens_routed = sum(layer.tokens_routed for layer in stats)
                tokens_overflowed = sum(layer.tokens_overflowed for layer in stats)
                tokens_dropped = sum(layer.tokens_dropped for layer in stats)
                overflow_ratios.append(tokens_overflowed / max(tokens_routed, 1))
                dropped_ratios.append(tokens_dropped / max(tokens_routed, 1))
                routing_entropy_values.append(
                    sum(layer.routing_entropy_mean for layer in stats) / len(stats)
                )
                expert_balance_values.append(
                    sum(layer.expert_balance_score for layer in stats) / len(stats)
                )
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
                "active_parameters_per_token": profile.active_parameters_per_token,
                "train_loss_first": losses[0],
                "train_loss_last": losses[-1],
                "validation_loss_proxy": losses[-1],
                "tokens_per_second": tokens_processed / elapsed,
                "memory": memory,
                "flops_per_token": profile.flops_per_token,
                "flops_per_step": profile.flops_per_step,
                "load_balancing_loss_last": load_balance_losses[-1] if load_balance_losses else 0.0,
                "expert_utilization": utilization_snapshots[-1] if utilization_snapshots else [],
                "overflow_ratio": overflow_ratios[-1] if overflow_ratios else 0.0,
                "dropped_ratio": dropped_ratios[-1] if dropped_ratios else 0.0,
                "routing_entropy_mean": routing_entropy_values[-1] if routing_entropy_values else 0.0,
                "expert_balance_score": expert_balance_values[-1] if expert_balance_values else 0.0,
            }
        )
    return results
