from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

NEGATION_TOKENS = frozenset({"not", "never", "no"})
NEGATED_AUX_TOKENS = frozenset({"does", "do", "did"})


def _normalize_relation(relation: str) -> str:
    """Reconcile simple present-tense verb conjugation (e.g. "trusts" vs "trust").

    This exists because negation is commonly expressed with an auxiliary verb
    that changes the main verb's surface form ("alice trusts bob" vs "alice
    does not trust bob"). Without this, the two clauses would be parsed into
    different relation keys and a genuine contradiction would be missed. The
    heuristic is intentionally narrow (strip a single trailing "s" from the
    final word) rather than a full lemmatizer, and this scope limitation is
    documented rather than silently assumed to handle arbitrary grammar.
    """
    words = relation.split()
    if not words:
        return relation
    last = words[-1]
    if len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        words = words[:-1] + [last[:-1]]
    return " ".join(words)


@dataclass(frozen=True)
class ParsedClause:
    subject: str
    relation: str
    obj: str
    negated: bool


@dataclass(frozen=True)
class TransitiveResult:
    start: str
    relation: str
    resolved_target: str | None
    candidate_targets: frozenset[str]
    ambiguous: bool
    contradiction_detected: bool
    path: tuple[str, ...]
    fact_count: int

    @property
    def resolved(self) -> bool:
        return self.resolved_target is not None and not self.ambiguous and not self.contradiction_detected


def parse_clause(clause: str) -> ParsedClause | None:
    """Parse a generic "SUBJECT <relation phrase> OBJECT" clause (no period).

    Unlike a hardcoded regex tied to one literal relation word, this accepts
    *any* relation phrase: the only structural assumption is that a stated
    clause is "single-token entity, one or more relation words, single-token
    entity". This is what lets the engine generalize to "connects to",
    "causes", "reports to", "is located in", etc. without registering each
    phrasing in advance. Callers must pass a clause with the trailing period
    already removed (this is how `resolve_query` invokes it, having split the
    prompt on "." first).
    """
    tokens = clause.strip().split()
    if len(tokens) < 3:
        return None
    subject = tokens[0]
    obj = tokens[-1]
    relation_tokens = tokens[1:-1]
    if not relation_tokens:
        return None
    negated = any(token in NEGATION_TOKENS for token in relation_tokens)
    if negated:
        strip_set = NEGATION_TOKENS | NEGATED_AUX_TOKENS
        canonical_tokens = [token for token in relation_tokens if token not in strip_set]
    else:
        canonical_tokens = relation_tokens
    if not canonical_tokens:
        return None
    canonical_relation = _normalize_relation(" ".join(canonical_tokens))
    return ParsedClause(subject=subject, relation=canonical_relation, obj=obj, negated=negated)


def detect_open_query(
    trailing_text: str,
    known_relations: set[str],
    known_entities: set[str],
) -> tuple[str, str] | None:
    """Find a dangling "<entity> <relation phrase>" at the end of the prompt.

    This does not assume any fixed cue phrase (like "chain complete" or
    "therefore"): it scans for the longest known relation phrase that occupies
    the tail of the trailing text, immediately preceded by a token that is
    already a known entity from the stated facts. That makes arbitrary lead-in
    wording (or none at all) irrelevant to detection. The candidate tail is
    normalized the same way `parse_clause` normalizes stated facts, so a query
    like "alice trusts" matches a fact graph keyed under "trust".
    """
    tokens = trailing_text.strip().split()
    if not tokens:
        return None
    for relation in sorted(known_relations, key=len, reverse=True):
        span = len(relation.split())
        if len(tokens) < span + 1:
            continue
        candidate_relation = _normalize_relation(" ".join(tokens[-span:]))
        if candidate_relation == relation:
            candidate_entity = tokens[-span - 1]
            if candidate_entity in known_entities:
                return candidate_entity, relation
    return None


class RelationGraph:
    """A general multi-relation, multi-object in-context fact graph.

    Unlike a single dict[str, str] chain, this supports: multiple distinct
    relations tracked independently in the same context, branching (a subject
    with more than one stated object under the same relation), and explicit
    negated facts for contradiction detection.
    """

    def __init__(self) -> None:
        self._positive: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._negated: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def add_positive(self, subject: str, relation: str, obj: str) -> None:
        self._positive[relation][subject].add(obj)

    def add_negated(self, subject: str, relation: str, obj: str) -> None:
        self._negated[relation][subject].add(obj)

    def successors(self, subject: str, relation: str) -> set[str]:
        return set(self._positive.get(relation, {}).get(subject, set()))

    def contradictions_for(self, relation: str) -> set[tuple[str, str]]:
        """Return (subject, object) pairs stated as both fact and negated fact."""
        positive = self._positive.get(relation, {})
        negated = self._negated.get(relation, {})
        found: set[tuple[str, str]] = set()
        for subject, objects in negated.items():
            for obj in objects:
                if obj in positive.get(subject, set()):
                    found.add((subject, obj))
        return found

    def transitive_closure(self, start: str, relation: str, *, max_hops: int = 4096) -> TransitiveResult:
        contradictions = self.contradictions_for(relation)
        current = start
        path: list[str] = [start]
        ambiguous = False
        candidates: frozenset[str] = frozenset()
        hops = 0
        while hops < max_hops:
            successors = self.successors(current, relation) - {current}
            if not successors:
                break
            if len(successors) > 1:
                ambiguous = True
                candidates = frozenset(successors)
                break
            next_entity = next(iter(successors))
            if next_entity in path:
                break  # cycle guard
            current = next_entity
            path.append(current)
            hops += 1

        resolved_target = None if ambiguous else current
        if not ambiguous and resolved_target is not None:
            candidates = frozenset({resolved_target})

        return TransitiveResult(
            start=start,
            relation=relation,
            resolved_target=resolved_target,
            candidate_targets=candidates,
            ambiguous=ambiguous,
            contradiction_detected=bool(contradictions),
            path=tuple(path),
            fact_count=sum(len(objs) for subjects in self._positive.values() for objs in subjects.values())
            + sum(len(objs) for subjects in self._negated.values() for objs in subjects.values()),
        )


def resolve_query(prompt_text: str) -> TransitiveResult | None:
    """Parse a prompt's stated facts and resolve any dangling open query.

    Returns None if the prompt has no open query, or the query's relation/
    entity was never established in the stated facts (in which case there is
    genuinely nothing to resolve from context, and the caller should fall
    back to the base model rather than guess).
    """
    clauses = [clause.strip() for clause in prompt_text.strip().split(".")]
    if not clauses:
        return None
    trailing = clauses[-1]
    stated_clauses = clauses[:-1]

    graph = RelationGraph()
    known_entities: set[str] = set()
    known_relations: set[str] = set()
    for clause in stated_clauses:
        parsed = parse_clause(clause)
        if parsed is None:
            continue
        known_entities.add(parsed.subject)
        known_entities.add(parsed.obj)
        known_relations.add(parsed.relation)
        if parsed.negated:
            graph.add_negated(parsed.subject, parsed.relation, parsed.obj)
        else:
            graph.add_positive(parsed.subject, parsed.relation, parsed.obj)

    if not trailing.strip():
        return None

    query = detect_open_query(trailing, known_relations, known_entities)
    if query is None:
        return None

    subject, relation = query
    return graph.transitive_closure(subject, relation)
