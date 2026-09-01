from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.phase_k_reasoning.relation_engine import resolve_query

# --- Legacy narrow helpers -------------------------------------------------
# These predate the general relation engine below and only ever recognized a
# single hardcoded relation word ("links") and a single hardcoded query cue
# ("chain complete X links"). They are kept only for direct unit testing of
# that literal pattern and are no longer what powers `resolve_pointer` /
# `PointerAugmentedRuntime`. New code should use `relation_engine.resolve_query`.

_LINK_FACT_PATTERN = re.compile(r"(\S+) links (\S+) \.")
_QUERY_ANCHOR_PATTERN = re.compile(r"chain complete (\S+) links$")


def extract_link_facts(context_text: str) -> dict[str, str]:
    """Parse 'X links Y .' clauses out of the given context text (legacy, narrow)."""
    facts: dict[str, str] = {}
    for source, target in _LINK_FACT_PATTERN.findall(context_text):
        facts[source] = target
    return facts


def detect_query_anchor(prompt_text: str) -> str | None:
    """Return the entity being queried for the literal 'chain complete X links' cue (legacy, narrow)."""
    match = _QUERY_ANCHOR_PATTERN.search(prompt_text.strip())
    if match is None:
        return None
    return match.group(1)


def resolve_transitive_target(link_facts: dict[str, str], start: str, *, max_hops: int = 4096) -> str | None:
    """Follow a single-relation chain dict from `start` to its final endpoint (legacy, narrow).

    Returns None if `start` has no outgoing fact (nothing to point to), and
    guards against cycles so a malformed or adversarial context cannot loop
    forever.
    """
    if start not in link_facts:
        return None
    current = start
    visited = {current}
    hops = 0
    while current in link_facts and hops < max_hops:
        next_entity = link_facts[current]
        if next_entity in visited:
            break
        visited.add(next_entity)
        current = next_entity
        hops += 1
    return current


# --- General entry point ----------------------------------------------------


@dataclass(frozen=True)
class PointerResolution:
    anchor_entity: str | None
    resolved_target: str | None
    fact_count: int
    ambiguous: bool = False
    contradiction_detected: bool = False
    candidate_targets: frozenset[str] = field(default_factory=frozenset)

    @property
    def resolved(self) -> bool:
        return self.resolved_target is not None and not self.ambiguous and not self.contradiction_detected


def resolve_pointer(prompt_text: str) -> PointerResolution:
    """Resolve an in-context relational query using the general relation engine.

    Handles arbitrary relation phrasing (not just "links"), branching/
    multi-valued relations (reported as `ambiguous` with `candidate_targets`
    rather than silently guessing one), and explicit negation-based
    contradictions (reported as `contradiction_detected`).
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
    )
