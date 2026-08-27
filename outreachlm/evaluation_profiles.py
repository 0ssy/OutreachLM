from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationProfile:
    prompt_length: int = 40
    eval_length: int = 80
    position_start: int = 40
    position_end: int = 52
    sample_count: int = 4096
    sample_seed: int = 42
    sample_batch_size: int = 256
    fallback_topk: int = 5
    heldout_slices: int = 4
    output_topk: int = 5
    hidden_transition_start: int = 38
    hidden_transition_end: int = 45
    output_sensitivity_start: int = 39
    output_sensitivity_end: int = 43

    def __post_init__(self) -> None:
        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be > 0.")
        if self.eval_length <= 0:
            raise ValueError("eval_length must be > 0.")
        if self.eval_length <= self.prompt_length:
            raise ValueError("eval_length must be greater than prompt_length.")
        if self.position_start < 1:
            raise ValueError("position_start must be >= 1.")
        if self.position_end < self.position_start:
            raise ValueError("position_end must be >= position_start.")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be > 0.")
        if self.sample_seed < 0:
            raise ValueError("sample_seed must be >= 0.")
        if self.sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be > 0.")
        if self.fallback_topk <= 0:
            raise ValueError("fallback_topk must be > 0.")
        if self.heldout_slices <= 0:
            raise ValueError("heldout_slices must be > 0.")
        if self.output_topk <= 0:
            raise ValueError("output_topk must be > 0.")
        if self.hidden_transition_start < 1:
            raise ValueError("hidden_transition_start must be >= 1.")
        if self.hidden_transition_end < self.hidden_transition_start:
            raise ValueError(
                "hidden_transition_end must be >= hidden_transition_start."
            )
        if self.output_sensitivity_start < 1:
            raise ValueError("output_sensitivity_start must be >= 1.")
        if self.output_sensitivity_end < self.output_sensitivity_start:
            raise ValueError(
                "output_sensitivity_end must be >= output_sensitivity_start."
            )
        if self.hidden_transition_end >= self.eval_length:
            raise ValueError("hidden_transition_end must be < eval_length.")
        if self.output_sensitivity_end >= self.eval_length:
            raise ValueError("output_sensitivity_end must be < eval_length.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_length": self.prompt_length,
            "eval_length": self.eval_length,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "sample_count": self.sample_count,
            "sample_seed": self.sample_seed,
            "sample_batch_size": self.sample_batch_size,
            "fallback_topk": self.fallback_topk,
            "heldout_slices": self.heldout_slices,
            "output_topk": self.output_topk,
            "hidden_transition_start": self.hidden_transition_start,
            "hidden_transition_end": self.hidden_transition_end,
            "output_sensitivity_start": self.output_sensitivity_start,
            "output_sensitivity_end": self.output_sensitivity_end,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationProfile":
        return cls(
            prompt_length=payload.get("prompt_length", 40),
            eval_length=payload.get("eval_length", 80),
            position_start=payload.get("position_start", 40),
            position_end=payload.get("position_end", 52),
            sample_count=payload.get("sample_count", 4096),
            sample_seed=payload.get("sample_seed", 42),
            sample_batch_size=payload.get("sample_batch_size", 256),
            fallback_topk=payload.get("fallback_topk", 5),
            heldout_slices=payload.get("heldout_slices", 4),
            output_topk=payload.get("output_topk", 5),
            hidden_transition_start=payload.get("hidden_transition_start", 38),
            hidden_transition_end=payload.get("hidden_transition_end", 45),
            output_sensitivity_start=payload.get("output_sensitivity_start", 39),
            output_sensitivity_end=payload.get("output_sensitivity_end", 43),
        )


@dataclass(frozen=True)
class FrontierEvaluationProfile:
    validation_batches: int = 8
    compute_perplexity: bool = True
    compute_teacher_accuracy: bool = True
    compute_free_rollout: bool = True
    compute_divergence: bool = True
    compute_recovery: bool = True
    long_context_eval_length: int = 1024
    compute_tokens_per_second: bool = True
    compute_samples_per_second: bool = True
    compute_memory: bool = True
    compute_communication_overhead: bool = True
    compute_checkpoint_time: bool = True
    compute_data_loading_time: bool = True
    compute_moe_metrics: bool = True

    def __post_init__(self) -> None:
        if self.validation_batches <= 0:
            raise ValueError("validation_batches must be > 0.")
        if self.long_context_eval_length <= 0:
            raise ValueError("long_context_eval_length must be > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_batches": self.validation_batches,
            "compute_perplexity": self.compute_perplexity,
            "compute_teacher_accuracy": self.compute_teacher_accuracy,
            "compute_free_rollout": self.compute_free_rollout,
            "compute_divergence": self.compute_divergence,
            "compute_recovery": self.compute_recovery,
            "long_context_eval_length": self.long_context_eval_length,
            "compute_tokens_per_second": self.compute_tokens_per_second,
            "compute_samples_per_second": self.compute_samples_per_second,
            "compute_memory": self.compute_memory,
            "compute_communication_overhead": self.compute_communication_overhead,
            "compute_checkpoint_time": self.compute_checkpoint_time,
            "compute_data_loading_time": self.compute_data_loading_time,
            "compute_moe_metrics": self.compute_moe_metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrontierEvaluationProfile":
        return cls(
            validation_batches=payload.get("validation_batches", 8),
            compute_perplexity=payload.get("compute_perplexity", True),
            compute_teacher_accuracy=payload.get("compute_teacher_accuracy", True),
            compute_free_rollout=payload.get("compute_free_rollout", True),
            compute_divergence=payload.get("compute_divergence", True),
            compute_recovery=payload.get("compute_recovery", True),
            long_context_eval_length=payload.get("long_context_eval_length", 1024),
            compute_tokens_per_second=payload.get("compute_tokens_per_second", True),
            compute_samples_per_second=payload.get("compute_samples_per_second", True),
            compute_memory=payload.get("compute_memory", True),
            compute_communication_overhead=payload.get("compute_communication_overhead", True),
            compute_checkpoint_time=payload.get("compute_checkpoint_time", True),
            compute_data_loading_time=payload.get("compute_data_loading_time", True),
            compute_moe_metrics=payload.get("compute_moe_metrics", True),
        )
