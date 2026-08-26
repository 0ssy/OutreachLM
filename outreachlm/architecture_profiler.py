from dataclasses import dataclass
from typing import Any

from outreachlm.distributed_semantics import BatchSemantics
from outreachlm.model_config import DenseTransformerConfig
from outreachlm.scalable_model import ScalableTransformerModel


@dataclass(frozen=True)
class ArchitectureProfile:
    total_parameters: int
    embedding_parameters: int
    attention_parameters: int
    ffn_parameters: int
    normalization_parameters: int
    output_parameters: int
    router_parameters: int
    expert_parameters: int
    active_parameters_per_token: int
    parameter_memory_bytes: int
    activation_memory_bytes: int
    approximate_training_memory_bytes: int
    flops_per_token: float
    flops_per_step: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_parameters": self.total_parameters,
            "embedding_parameters": self.embedding_parameters,
            "attention_parameters": self.attention_parameters,
            "ffn_parameters": self.ffn_parameters,
            "normalization_parameters": self.normalization_parameters,
            "output_parameters": self.output_parameters,
            "router_parameters": self.router_parameters,
            "expert_parameters": self.expert_parameters,
            "active_parameters_per_token": self.active_parameters_per_token,
            "parameter_memory_bytes": self.parameter_memory_bytes,
            "activation_memory_bytes": self.activation_memory_bytes,
            "approximate_training_memory_bytes": self.approximate_training_memory_bytes,
            "flops_per_token": self.flops_per_token,
            "flops_per_step": self.flops_per_step,
        }


def profile_architecture(
    config: DenseTransformerConfig,
    *,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    world_size: int = 1,
) -> ArchitectureProfile:
    model = ScalableTransformerModel(config)
    embedding_parameters = 0
    attention_parameters = 0
    ffn_parameters = 0
    normalization_parameters = 0
    output_parameters = 0
    router_parameters = 0
    expert_parameters = 0

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith("token_embedding"):
            embedding_parameters += count
        elif ".attn." in name:
            attention_parameters += count
        elif ".ffn." in name:
            ffn_parameters += count
        elif "norm" in name:
            normalization_parameters += count
        elif name.startswith("output_head"):
            output_parameters += count
        elif ".router." in name:
            router_parameters += count
        elif ".experts." in name or ".fallback_dense." in name:
            expert_parameters += count
        else:
            # tied embeddings route output parameters through token_embedding weights.
            embedding_parameters += 0

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if config.moe_enabled:
        active_parameters_per_token = (
            embedding_parameters
            + normalization_parameters
            + attention_parameters
            + output_parameters
            + router_parameters
            + max(1, config.top_k) * (expert_parameters // max(1, config.num_experts))
        )
    else:
        active_parameters_per_token = total_parameters
    parameter_memory_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    activation_memory_bytes = (
        per_device_batch_size
        * config.context_length
        * config.embedding_dim
        * 4
        * (2 + (2 * config.num_layers))
    )
    approximate_training_memory_bytes = parameter_memory_bytes * 3 + activation_memory_bytes

    # Approximate dense transformer FLOPs/token (coarse estimator).
    d = config.embedding_dim
    h = config.num_heads
    l = config.num_layers
    c = config.context_length
    f = config.ffn_dim
    attention_linear = l * (8.0 * d * d)
    attention_context = l * (4.0 * c * d)
    ffn = l * (6.0 * d * f)
    output = 2.0 * d * config.vocab_size
    flops_per_token = attention_linear + attention_context + ffn + output

    semantics = BatchSemantics(
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
    )
    flops_per_step = flops_per_token * config.context_length * semantics.effective_batch_size

    return ArchitectureProfile(
        total_parameters=total_parameters,
        embedding_parameters=embedding_parameters,
        attention_parameters=attention_parameters,
        ffn_parameters=ffn_parameters,
        normalization_parameters=normalization_parameters,
        output_parameters=output_parameters,
        router_parameters=router_parameters,
        expert_parameters=expert_parameters,
        active_parameters_per_token=active_parameters_per_token,
        parameter_memory_bytes=parameter_memory_bytes,
        activation_memory_bytes=activation_memory_bytes,
        approximate_training_memory_bytes=approximate_training_memory_bytes,
        flops_per_token=flops_per_token,
        flops_per_step=flops_per_step,
    )
