from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from outreachlm.model_config import DenseTransformerConfig


@dataclass(frozen=True)
class RouterOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class MoEForwardStats:
    tokens_total: int
    tokens_routed: int
    tokens_accepted: int
    tokens_overflowed: int
    tokens_dropped: int
    expert_routed: list[int]
    expert_accepted: list[int]
    expert_overflowed: list[int]
    expert_utilization: list[float]
    routing_entropy_mean: float
    expert_balance_score: float


class ExpertFFN(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        hidden = config.expert_ffn_dim if config.expert_ffn_dim is not None else config.ffn_dim
        self.variant = config.ffn_variant
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_dim = hidden

        if self.variant == "swiglu":
            self.w1 = nn.Linear(config.embedding_dim, hidden, bias=config.use_bias)
            self.w2 = nn.Linear(config.embedding_dim, hidden, bias=config.use_bias)
            self.w3 = nn.Linear(hidden, config.embedding_dim, bias=config.use_bias)
        elif self.variant == "gated":
            self.w_gate = nn.Linear(config.embedding_dim, hidden, bias=config.use_bias)
            self.w_value = nn.Linear(config.embedding_dim, hidden, bias=config.use_bias)
            self.w_out = nn.Linear(hidden, config.embedding_dim, bias=config.use_bias)
        elif self.variant == "standard":
            self.w_in = nn.Linear(config.embedding_dim, hidden, bias=config.use_bias)
            self.w_out = nn.Linear(hidden, config.embedding_dim, bias=config.use_bias)
        else:
            raise ValueError(f"Unsupported expert ffn_variant: {self.variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "swiglu":
            gate = F.silu(self.w1(x))
            value = self.w2(x)
            return self.w3(self.dropout(gate * value))
        if self.variant == "gated":
            gate = torch.sigmoid(self.w_gate(x))
            value = F.gelu(self.w_value(x))
            return self.w_out(self.dropout(gate * value))
        return self.w_out(self.dropout(F.gelu(self.w_in(x))))


class TopKRouter(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.linear = nn.Linear(config.embedding_dim, config.num_experts, bias=config.router_bias)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        logits = self.linear(x)
        probabilities = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probabilities, k=self.top_k, dim=-1)
        return RouterOutput(
            logits=logits,
            probabilities=probabilities,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
        )


class MoELayer(nn.Module):
    def __init__(self, config: DenseTransformerConfig):
        super().__init__()
        self.config = config
        self.router = TopKRouter(config)
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(config.num_experts)])
        self.fallback_dense = ExpertFFN(config) if config.moe_fallback == "dense" else None
        self.last_stats: MoEForwardStats | None = None
        self.last_load_balancing_loss = torch.tensor(0.0)

    def _compute_load_balance_loss(self, probabilities: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        num_experts = self.config.num_experts
        flat_probs = probabilities.reshape(-1, num_experts)
        importance = flat_probs.mean(dim=0)

        flat_indices = topk_indices.reshape(-1)
        dispatch = torch.bincount(flat_indices, minlength=num_experts).to(probabilities.dtype)
        dispatch = dispatch / max(float(flat_indices.numel()), 1.0)
        uniform = torch.full_like(importance, 1.0 / num_experts)
        return ((importance - uniform) ** 2).mean() + ((dispatch - uniform) ** 2).mean()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, MoEForwardStats]:
        batch_size, sequence_length, hidden_dim = x.shape
        tokens = batch_size * sequence_length
        x_flat = x.reshape(tokens, hidden_dim)

        routing = self.router(x_flat)
        topk_indices = routing.topk_indices
        topk_weights = routing.topk_weights

        capacity = max(
            1,
            math.ceil(
                self.config.capacity_factor
                * float(tokens * self.config.top_k)
                / float(self.config.num_experts)
            ),
        )

        output_flat = torch.zeros_like(x_flat)
        expert_routed = [0 for _ in range(self.config.num_experts)]
        expert_accepted = [0 for _ in range(self.config.num_experts)]
        expert_overflowed = [0 for _ in range(self.config.num_experts)]
        tokens_overflowed = 0
        tokens_accepted = 0
        tokens_dropped = 0

        for expert_index, expert in enumerate(self.experts):
            match_positions = (topk_indices == expert_index).nonzero(as_tuple=False)
            expert_routed[expert_index] = int(match_positions.shape[0])
            if match_positions.numel() == 0:
                continue

            accepted = match_positions[:capacity]
            overflow = match_positions[capacity:]
            expert_accepted[expert_index] = int(accepted.shape[0])
            expert_overflowed[expert_index] = int(overflow.shape[0])
            tokens_accepted += int(accepted.shape[0])
            tokens_overflowed += int(overflow.shape[0])

            if accepted.numel() > 0:
                token_indices = accepted[:, 0]
                rank_indices = accepted[:, 1]
                expert_inputs = x_flat[token_indices]
                expert_outputs = expert(expert_inputs)
                weights = topk_weights[token_indices, rank_indices].unsqueeze(-1)
                output_flat[token_indices] = output_flat[token_indices] + (weights * expert_outputs)

            if overflow.numel() > 0:
                if self.fallback_dense is not None:
                    token_indices = overflow[:, 0]
                    rank_indices = overflow[:, 1]
                    fallback_outputs = self.fallback_dense(x_flat[token_indices])
                    weights = topk_weights[token_indices, rank_indices].unsqueeze(-1)
                    output_flat[token_indices] = output_flat[token_indices] + (weights * fallback_outputs)
                else:
                    tokens_dropped += int(overflow.shape[0])

        load_balancing_loss = self._compute_load_balance_loss(routing.probabilities, topk_indices)
        self.last_load_balancing_loss = load_balancing_loss
        utilization = [
            (accepted / max(routed, 1))
            for accepted, routed in zip(expert_accepted, expert_routed)
        ]
        routing_entropy_mean = float(
            -(routing.probabilities * torch.log(routing.probabilities + 1e-12)).sum(dim=-1).mean().item()
        )
        routed_total = max(sum(expert_routed), 1)
        routed_distribution = torch.tensor(
            [routed / routed_total for routed in expert_routed],
            dtype=x.dtype,
            device=x.device,
        )
        uniform_distribution = torch.full_like(routed_distribution, 1.0 / self.config.num_experts)
        expert_balance_score = float(
            1.0 - torch.abs(routed_distribution - uniform_distribution).mean().item()
        )
        stats = MoEForwardStats(
            tokens_total=tokens,
            tokens_routed=tokens * self.config.top_k,
            tokens_accepted=tokens_accepted,
            tokens_overflowed=tokens_overflowed,
            tokens_dropped=tokens_dropped,
            expert_routed=expert_routed,
            expert_accepted=expert_accepted,
            expert_overflowed=expert_overflowed,
            expert_utilization=utilization,
            routing_entropy_mean=routing_entropy_mean,
            expert_balance_score=expert_balance_score,
        )
        self.last_stats = stats
        return output_flat.reshape(batch_size, sequence_length, hidden_dim), load_balancing_loss, stats
