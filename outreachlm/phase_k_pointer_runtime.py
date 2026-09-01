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


@dataclass(frozen=True)
class PointerCompletion:
    prompt: str
    completion: str | None
    pointer: PointerResolution
    used_pointer: bool
    truncated: bool = False


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

    Resolved answers may themselves be multi-word (e.g. "the old bridge",
    "New York City"). The underlying tokenizer only assigns IDs to single
    whitespace-delimited words, so a single `predict_next` call can only ever
    boost one token: it boosts the first word of the resolved answer. Use
    `complete_query` to deterministically obtain the *entire* multi-word
    answer (word by word) rather than just its first token.
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
            # A candidate may itself be multi-word; only its first word can be
            # boosted for this single next-token step.
            candidate_ids = [
                token_id
                for token in pointer.candidate_targets
                if (token_id := self.runtime.tokenizer.token_to_id.get(token.split()[0])) is not None
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
            first_word = pointer.resolved_target.split()[0]
            target_id = self.runtime.tokenizer.token_to_id.get(first_word)
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

    def complete_query(self, prompt_text: str) -> PointerCompletion:
        """Deterministically resolve the *entire* answer span for a query.

        Unlike `predict_next`, which can only directly boost a single token,
        this returns the full resolved answer (all of its words, in order) as
        long as every word is representable in the tokenizer's vocabulary. If
        a later word isn't in vocabulary, the completion stops there rather
        than silently fabricating a token; `truncated=True` makes that
        explicit so the caller doesn't have to independently recompute the
        expected word count to notice a partial answer.
        """
        pointer = resolve_pointer(prompt_text)
        if not pointer.resolved or pointer.resolved_target is None:
            return PointerCompletion(prompt=prompt_text, completion=None, pointer=pointer, used_pointer=False)

        words = pointer.resolved_target.split()
        completed_words: list[str] = []
        for word in words:
            if word not in self.runtime.tokenizer.token_to_id:
                break
            completed_words.append(word)

        truncated = len(completed_words) < len(words)

        if not completed_words:
            return PointerCompletion(
                prompt=prompt_text, completion=None, pointer=pointer, used_pointer=False, truncated=truncated
            )

        return PointerCompletion(
            prompt=prompt_text,
            completion=" ".join(completed_words),
            pointer=pointer,
            used_pointer=True,
            truncated=truncated,
        )

