from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RollbackDecision:
    accepted: bool
    reason: str
    candidate_tokens: tuple[str, ...]


class CandidateWorkspace:
    """Transactional work buffer for reasoning steps that must be rejected if invalid."""

    def __init__(self) -> None:
        self._staged: list[str] = []
        self._facts: set[str] = set()
        self._contradictions: set[str] = set()

    def stage(self, candidate: str | list[str], *, fact_set: set[str] | None = None) -> None:
        tokens = [candidate] if isinstance(candidate, str) else list(candidate)
        self._staged.extend(tokens)
        if fact_set is not None:
            self._facts.update(fact_set)

    def add_fact(self, fact: str) -> None:
        self._facts.add(fact)

    def add_contradiction(self, contradiction: str) -> None:
        self._contradictions.add(contradiction)

    def verify(self, *, required_tokens: list[str] | None = None) -> RollbackDecision:
        if required_tokens is not None:
            missing = [token for token in required_tokens if token not in self._staged]
            if missing:
                self.rollback()
                return RollbackDecision(False, f"missing_required_tokens:{','.join(missing)}", tuple(self._staged))
        for token in self._staged:
            lowered = token.lower()
            for contradiction in self._contradictions:
                if contradiction.lower() in lowered:
                    self.rollback()
                    return RollbackDecision(False, f"contradiction:{contradiction}", tuple(self._staged))
        return RollbackDecision(True, "accepted", tuple(self._staged))

    def rollback(self) -> None:
        self._staged.clear()

    @property
    def staged_tokens(self) -> tuple[str, ...]:
        return tuple(self._staged)
