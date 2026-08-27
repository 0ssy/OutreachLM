from typing import Any

import math


def summarize_frontier_step(
    step_metrics: dict[str, Any],
    *,
    validation_loss: float | None = None,
    teacher_accuracy: float | None = None,
    free_rollout_match: float | None = None,
    divergence_position: int | None = None,
    recovery_rate: float | None = None,
) -> dict[str, Any]:
    throughput = step_metrics.get("throughput", {})
    memory = step_metrics.get("memory", {})
    moe = step_metrics.get("moe", {})
    perplexity = None
    if validation_loss is not None:
        perplexity = math.exp(validation_loss)

    return {
        "quality": {
            "validation_loss": validation_loss,
            "perplexity": perplexity,
            "teacher_accuracy": teacher_accuracy,
            "free_rollout_match": free_rollout_match,
            "divergence_position": divergence_position,
            "recovery_rate": recovery_rate,
        },
        "systems": {
            "tokens_per_second": throughput.get("tokens_per_second"),
            "samples_per_second": throughput.get("samples_per_second"),
            "global_tokens_per_second": throughput.get("global_tokens_per_second"),
            "global_samples_per_second": throughput.get("global_samples_per_second"),
            "data_loading_time_seconds": throughput.get("time_per_data_seconds"),
            "checkpoint_time_seconds": throughput.get("time_per_optimizer_seconds"),
            "communication_time_seconds": memory.get("communication_time_seconds"),
            "estimated_training_state_bytes": memory.get("estimated_training_state_bytes"),
        },
        "moe": {
            "load_balancing_loss": moe.get("load_balancing_loss"),
            "expert_utilization": moe.get("expert_utilization", []),
            "routing_entropy_mean": moe.get("routing_entropy_mean"),
            "tokens_dropped": moe.get("tokens_dropped"),
            "tokens_overflowed": moe.get("tokens_overflowed"),
            "overflow_ratio": moe.get("overflow_ratio"),
            "expert_balance_score": moe.get("expert_balance_score"),
        },
    }
