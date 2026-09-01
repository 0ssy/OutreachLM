from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from outreachlm.phase_h_runtime import BoundedStateRuntime
from src.phase_k_reasoning.pointer import PointerResolution, resolve_pointer


@dataclass(frozen=True)
class PointerAugmentedConfig:
    copy_weight: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {"copy_weight": self.copy_weight}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PointerAugmentedConfig":
        return cls(copy_weight=float(payload.get("copy_weight", 0.95)))


@dataclass(frozen=True)
class PointerPrediction:
    predicted_token: str
    probabilities: np.ndarray
    pointer: PointerResolution
    used_pointer: bool


class PointerAugmentedRuntime:
    """Wraps a frozen BoundedStateRuntime with a general in-context relation pointer.

    The frozen n-gram core (`BoundedStateRuntime` / `PhaseGHybridRuntime`) is
    never modified. This class only *blends* its output distribution with a
    pointer-resolved answer, and handles three distinct outcomes honestly
    rather than always committing to one guess:

    - resolved: a single, unambiguous, uncontradicted answer was traced from
      facts stated in the prompt -> confidently boost that token.
    - ambiguous: the queried relation branches (multiple valid successors)
      -> split confidence across the candidates rather than picking one.
    - contradiction: the stated facts contradict each other for this
      relation (an explicit negation of an asserted fact) -> decline to
      confidently answer at all, and fall back to the base distribution.
    """

    def __init__(self, runtime: BoundedStateRuntime, *, config: PointerAugmentedConfig | None = None) -> None:
        self.runtime = runtime
        self.config = config or PointerAugmentedConfig()

    def predict_next(self, prompt_text: str) -> PointerPrediction:
        prefix_tokens = self.runtime.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        base_probabilities = self.runtime._distribution(
            prefix_tokens, recent_tokens=prefix_tokens[-64:], apply_safety=False
        )
        pointer = resolve_pointer(prompt_text)

        def _fallback() -> PointerPrediction:
            predicted_id = int(np.argmax(base_probabilities))
            predicted_token = self.runtime.tokenizer.decode([predicted_id], skip_special_tokens=False)
            return PointerPrediction(
                predicted_token=predicted_token,
                probabilities=base_probabilities,
                pointer=pointer,
                used_pointer=False,
            )

        # An explicit contradiction (asserted fact + its negation) means the
        # stated context is inconsistent for this relation; do not confidently
        # boost any single answer.
        if pointer.contradiction_detected:
            return _fallback()

        if pointer.ambiguous and pointer.candidate_targets:
            candidate_ids = [
                self.runtime.tokenizer.token_to_id[token]
                for token in pointer.candidate_targets
                if token in self.runtime.tokenizer.token_to_id
            ]
            if not candidate_ids:
                return _fallback()
            copy_weight = self.config.copy_weight
            share = copy_weight / len(candidate_ids)
            blended = base_probabilities * (1.0 - copy_weight)
            for token_id in candidate_ids:
                blended[token_id] += share
            blended = blended / blended.sum()
            predicted_id = int(np.argmax(blended))
            predicted_token = self.runtime.tokenizer.decode([predicted_id], skip_special_tokens=False)
            return PointerPrediction(
                predicted_token=predicted_token,
                probabilities=blended,
                pointer=pointer,
                used_pointer=True,
            )

        if pointer.resolved and pointer.resolved_target is not None:
            target_id = self.runtime.tokenizer.token_to_id.get(pointer.resolved_target)
            if target_id is None:
                return _fallback()
            copy_weight = self.config.copy_weight
            blended = base_probabilities * (1.0 - copy_weight)
            blended[target_id] += copy_weight
            blended = blended / blended.sum()
            predicted_id = int(np.argmax(blended))
            predicted_token = self.runtime.tokenizer.decode([predicted_id], skip_special_tokens=False)
            return PointerPrediction(
                predicted_token=predicted_token,
                probabilities=blended,
                pointer=pointer,
                used_pointer=True,
            )

        return _fallback()
