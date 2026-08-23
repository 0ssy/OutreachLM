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
