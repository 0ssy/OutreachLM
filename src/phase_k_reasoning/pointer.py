from __future__ import annotations

from dataclasses import dataclass, field

from src.phase_k_reasoning.relation_engine import resolve_query


@dataclass(frozen=True)
class PointerResolution:
    anchor_entity: str | None
    resolved_target: str | None
    fact_count: int
    ambiguous: bool = False
    contradiction_detected: bool = False
    candidate_targets: frozenset[str] = field(default_factory=frozenset)
    source: str = "context"
    coreference_applied: bool = False

    @property
    def resolved(self) -> bool:
        return self.resolved_target is not None and not self.ambiguous and not self.contradiction_detected


def resolve_pointer(prompt_text: str) -> PointerResolution:
    """Resolve an in-context relational query using the general relation engine.

    Handles arbitrary relation phrasing (not just "links"), multi-word
    entities (quoted spans or capitalized proper-noun runs), pronoun/
    coreference resolution (nearest-preceding-mention heuristic), branching/
    multi-valued relations (reported as `ambiguous` with `candidate_targets`
    rather than silently guessing one), explicit negation-based
    contradictions (reported as `contradiction_detected`), and a fallback to
    an external world-knowledge fact store when the prompt alone can't
    resolve the query (reported via `source`).
    """
    result = resolve_query(prompt_text)
    if result is None:
        return PointerResolution(anchor_entity=None, resolved_target=None, fact_count=0)
    return PointerResolution(
        anchor_entity=result.start,
        resolved_target=result.resolved_target,
        fact_count=result.fact_count,
        ambiguous=result.ambiguous,
        contradiction_detected=result.contradiction_detected,
        candidate_targets=result.candidate_targets,
        source=result.source,
        coreference_applied=result.coreference_applied,
    )
