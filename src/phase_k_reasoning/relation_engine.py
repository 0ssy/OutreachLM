from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

NEGATION_TOKENS = frozenset({"not", "never", "no"})
NEGATED_AUX_TOKENS = frozenset({"does", "do", "did"})

# Common English negation contractions: each one already fuses an auxiliary
# verb with "not" into a single token, so it is treated as a self-contained
# negation marker (triggers `negated=True` and is stripped entirely) rather
# than requiring "does"/"not" to appear as two separate words.
NEGATION_CONTRACTIONS = frozenset(
    {
        "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
        "won't", "can't", "cannot", "couldn't", "wouldn't", "shouldn't",
        "hasn't", "haven't", "hadn't",
    }
)

# Nearest-preceding-mention pronoun set. This is a recency heuristic, not
# full discourse coreference: every pronoun below resolves to whichever
# entity chunk was most recently seen in subject/object position. Gendered
# pronouns (he/him/his/she/her/hers) additionally prefer a gender-compatible
# candidate via `NAME_GENDER_HINTS` (see `_nearest_antecedent`) before
# falling back to plain recency; gender-neutral and plural pronouns always
# use plain recency and do not construct multi-entity group referents.
PRONOUN_TOKENS = frozenset(
    {
        "he", "she", "it", "him", "her", "his", "hers", "its",
        "himself", "herself", "itself",
        "they", "them", "their", "themselves",
    }
)
MASCULINE_PRONOUNS = frozenset({"he", "him", "his", "himself"})
FEMININE_PRONOUNS = frozenset({"she", "her", "hers", "herself"})

# A small, explicitly bounded set of common given names used only to break
# ties in nearest-mention pronoun resolution when a gendered pronoun would
# otherwise resolve to a grammatically incompatible antecedent. This is NOT
# general gender inference: any name absent from this dictionary is simply
# treated as gender-unknown and resolution falls back to plain recency.
NAME_GENDER_HINTS: dict[str, str] = {
    "alice": "fem", "bob": "masc", "charlie": "masc", "dana": "fem",
    "eve": "fem", "frank": "masc", "grace": "fem", "henry": "masc",
    "irene": "fem", "jack": "masc", "karen": "fem", "leo": "masc",
    "mary": "fem", "nathan": "masc", "olivia": "fem", "peter": "masc",
    "queenie": "fem", "robert": "masc", "susan": "fem", "thomas": "masc",
}

# A small set of common lowercase determiners/articles that should not be
# treated as their own standalone "entity" chunk when capitalized only
# because they happen to start a sentence (e.g. "The engine causes damage").
# If one of these is followed by another capitalized word, it is still kept
# as part of that multi-word run (e.g. "The New York Times").
COMMON_NON_ENTITY_WORDS = frozenset({"the", "a", "an", "this", "that", "these", "those"})

# Lowercase connector words that may appear *inside* a multi-word capitalized
# proper-noun run without breaking it, as long as another capitalized word
# follows (e.g. "Bank of America", "Museum of the North"). This lets the
# capitalization heuristic handle real multi-word proper nouns that include
# ordinary lowercase words, not just consecutive capitalized tokens.
CONNECTOR_WORDS = frozenset({"of", "the", "and", "de", "van", "der", "la", "for"})

# A small, explicit lookup table mapping registered relation-phrase synonyms
# onto a single shared canonical key, so a fact stated with one phrasing can
# be queried with a different registered phrasing of the same relation. This
# is a bounded lookup, not semantic similarity matching: an unregistered
# paraphrase is still treated as a distinct relation.
RELATION_SYNONYMS: dict[str, str] = {
    "connects to": "link",
    "is connected to": "link",
    "leads to": "cause",
    "results in": "cause",
    "brings about": "cause",
    "is friends with": "know",
    "is acquainted with": "know",
    "befriend": "know",
}

# Common irregular verb forms that the generic trailing-"s" stripping
# heuristic in `_normalize_relation` cannot reconcile (e.g. "has" vs "have"
# is a 3-character irregular form, and "is"/"are"/"was"/"were" don't end in
# a regular "-s" pattern at all).
IRREGULAR_VERB_FORMS: dict[str, str] = {
    "is": "be", "are": "be", "was": "be", "were": "be",
    "has": "have", "had": "have",
}

DEFAULT_WORLD_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "world_knowledge.json"


def _normalize_relation(relation: str) -> str:
    """Reconcile simple verb conjugation (e.g. "trusts" vs "trust", "has" vs "have").

    This exists because negation is commonly expressed with an auxiliary verb
    that changes the main verb's surface form ("alice trusts bob" vs "alice
    does not trust bob"). Without this, the two clauses would be parsed into
    different relation keys and a genuine contradiction would be missed.
    Common irregular forms (be/have family) are reconciled via
    `IRREGULAR_VERB_FORMS`; the "-ies" -> "-y" pattern (supplies -> supply)
    and a general trailing-"s" strip cover regular verbs. This remains a
    narrow, documented heuristic, not a full lemmatizer: it will not
    reconcile arbitrary irregular verbs outside these patterns.
    """
    words = relation.split()
    if not words:
        return relation
    last = words[-1]
    if last in IRREGULAR_VERB_FORMS:
        words = words[:-1] + [IRREGULAR_VERB_FORMS[last]]
    elif len(last) > 4 and last.endswith("ies"):
        words = words[:-1] + [last[:-3] + "y"]
    elif len(last) > 3 and last.endswith("s") and not last.endswith("ss"):
        words = words[:-1] + [last[:-1]]
    return " ".join(words)


def _canonicalize_relation(relation: str) -> str:
    """Map a registered relation-phrase synonym onto its shared canonical key."""
    return RELATION_SYNONYMS.get(relation, relation)


def _relation_key(raw_relation: str) -> str:
    """Single source of truth for turning a raw relation phrase into the
    key used for storage and lookup: tense normalization, then synonym
    canonicalization."""
    return _canonicalize_relation(_normalize_relation(raw_relation))


def _entity_gender(entity: str) -> str | None:
    words = entity.split()
    if not words:
        return None
    return NAME_GENDER_HINTS.get(words[0].lower())


def _required_gender(pronoun: str) -> str | None:
    lowered = pronoun.lower()
    if lowered in MASCULINE_PRONOUNS:
        return "masc"
    if lowered in FEMININE_PRONOUNS:
        return "fem"
    return None


def _nearest_antecedent(pronoun: str, recency: list[str]) -> str | None:
    """Resolve a pronoun to its nearest preceding mention, preferring a
    gender-compatible candidate for gendered pronouns when one is known via
    `NAME_GENDER_HINTS`. Falls back to plain nearest mention when the
    pronoun is gender-neutral/plural, or when no compatible candidate is
    found among known names."""
    if not recency:
        return None
    required = _required_gender(pronoun)
    if required is not None:
        for candidate in reversed(recency):
            if _entity_gender(candidate) == required:
                return candidate
    return recency[-1]


def _chunk_plain_text(text: str) -> list[str]:
    """Split unquoted text into chunks, grouping consecutive capitalized words.

    Lowercase tokens (typical relation words: "links", "causes", "is",
    "located", "in", ...) each become their own chunk, unchanged from the
    original single-token behaviour. Consecutive capitalized-initial tokens
    ("New York City") are merged into a single multi-word entity chunk, and a
    lowercase connector word (see `CONNECTOR_WORDS`) inside such a run does
    not break it as long as another capitalized word follows ("Bank of
    America"). A capitalized word whose lowercase form is a common
    determiner/article (see `COMMON_NON_ENTITY_WORDS`) is not treated as its
    own standalone entity chunk when it appears alone (typically because it
    only happens to start a sentence, not because it names something), so
    "The engine causes damage" parses `engine` as the subject rather than
    `The`. This remains a lightweight heuristic, not a trained NER model: it
    can still misfire on capitalized words outside these specific patterns.
    """
    tokens = text.strip().split()
    chunks: list[str] = []
    buffer: list[str] = []

    def _flush() -> None:
        if not buffer:
            return
        if len(buffer) == 1 and buffer[0].lower() in COMMON_NON_ENTITY_WORDS:
            buffer.clear()
            return
        chunks.append(" ".join(buffer))
        buffer.clear()

    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token[:1].isupper():
            buffer.append(token)
            i += 1
            continue
        if buffer and token.lower() in CONNECTOR_WORDS and i + 1 < n and tokens[i + 1][:1].isupper():
            buffer.append(token)
            i += 1
            continue
        _flush()
        chunks.append(token)
        i += 1
    _flush()
    return chunks


def _chunk_clause(clause: str) -> list[str]:
    """Chunk a clause into entity/relation-word units.

    Quoted spans ("the red door", 'the old bridge') are always treated as one
    entity chunk regardless of capitalization, letting multi-word lowercase
    entities be expressed explicitly. Unquoted text falls back to the
    capitalized-run heuristic in `_chunk_plain_text`.
    """
    chunks: list[str] = []
    pos = 0
    text = clause
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in ("\"", "'"):
            end = text.find(char, i + 1)
            if end == -1:
                break
            chunks.extend(_chunk_plain_text(text[pos:i]))
            quoted = text[i + 1 : end].strip()
            if quoted:
                chunks.append(quoted)
            i = end + 1
            pos = i
            continue
        i += 1
    chunks.extend(_chunk_plain_text(text[pos:]))
    return chunks


def _resolve_stated_clauses(clause_chunks: list[list[str]]) -> tuple[list[list[str]], list[str], bool]:
    """Replace pronoun chunks in subject/object position with the nearest
    preceding entity mention (recency heuristic, see `PRONOUN_TOKENS`), for
    clauses that follow the strict "SUBJECT relation OBJECT" fact grammar
    (i.e. every clause except the final, possibly filler-prefixed, query
    clause -- see `_resolve_trailing_pronoun` for that case).

    Returns the rewritten clause-chunk lists, the final recency list (most
    recent mention last, used to resolve a pronoun in the trailing query
    clause too), and whether any replacement was made.
    """
    recency: list[str] = []
    resolved: list[list[str]] = []
    applied = False

    def _resolve_slot(chunk: str) -> str:
        nonlocal applied
        if chunk.lower() in PRONOUN_TOKENS and recency:
            antecedent = _nearest_antecedent(chunk, recency)
            if antecedent is not None:
                applied = True
                return antecedent
        return chunk

    for chunks in clause_chunks:
        if not chunks:
            resolved.append(chunks)
            continue
        new_chunks = list(chunks)
        new_chunks[0] = _resolve_slot(new_chunks[0])
        if len(new_chunks) >= 2:
            new_chunks[-1] = _resolve_slot(new_chunks[-1])
        resolved.append(new_chunks)
        if new_chunks[0].lower() not in PRONOUN_TOKENS:
            recency.append(new_chunks[0])
        if len(new_chunks) >= 2 and new_chunks[-1].lower() not in PRONOUN_TOKENS:
            recency.append(new_chunks[-1])
    return resolved, recency, applied


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
    source: str = "context"
    coreference_applied: bool = False

    @property
    def resolved(self) -> bool:
        return self.resolved_target is not None and not self.ambiguous and not self.contradiction_detected


def _parse_chunks(chunks: list[str]) -> ParsedClause | None:
    if len(chunks) < 3:
        return None
    subject = chunks[0]
    obj = chunks[-1]
    relation_words = [word for chunk in chunks[1:-1] for word in chunk.split()]
    if not relation_words:
        return None
    lowered_words = [word.lower() for word in relation_words]
    negated = any(word in NEGATION_TOKENS or word in NEGATION_CONTRACTIONS for word in lowered_words)
    if negated:
        strip_set = NEGATION_TOKENS | NEGATED_AUX_TOKENS | NEGATION_CONTRACTIONS
        canonical_words = [word for word in relation_words if word.lower() not in strip_set]
    else:
        canonical_words = relation_words
    if not canonical_words:
        return None
    canonical_relation = _relation_key(" ".join(canonical_words))
    return ParsedClause(subject=subject, relation=canonical_relation, obj=obj, negated=negated)


def parse_clause(clause: str) -> ParsedClause | None:
    """Parse a generic "SUBJECT <relation phrase> OBJECT" clause (no period).

    Unlike a hardcoded regex tied to one literal relation word, this accepts
    *any* relation phrase, and unlike a plain whitespace split, the subject
    and object may themselves be multi-word entities (quoted spans, or
    consecutive capitalized words like "New York City"). Callers must pass a
    clause with the trailing period already removed (this is how
    `resolve_query` invokes it, having split the prompt on "." first).
    """
    return _parse_chunks(_chunk_clause(clause))


def _match_open_query(
    chunks: list[str],
    known_relations: set[str],
    known_entities: set[str],
    *,
    allow_pronoun: bool = False,
    max_relation_words: int = 5,
) -> tuple[str, str] | None:
    """Find the entity + canonical relation for a dangling query.

    Spans are tried by *word count of the surface text*, not by the word
    count of the stored canonical relation key, because a registered synonym
    (see `RELATION_SYNONYMS`) can canonicalize to a different word count than
    it was written with (e.g. the two-word surface phrase "connects to"
    canonicalizes to the one-word key "link"). Each candidate span is
    canonicalized before comparing against `known_relations`.
    """
    if not chunks:
        return None
    max_span = min(max_relation_words, len(chunks) - 1)
    for span in range(max_span, 0, -1):
        tail_chunks = chunks[-span:]
        if any(len(chunk.split()) != 1 for chunk in tail_chunks):
            # A relation-word position holds an unexpected multi-word chunk;
            # this candidate slicing cannot align with a relation phrase.
            continue
        candidate_relation = _relation_key(" ".join(tail_chunks))
        if candidate_relation in known_relations:
            candidate_entity = chunks[-span - 1]
            if candidate_entity in known_entities:
                return candidate_entity, candidate_relation
            if allow_pronoun and candidate_entity.lower() in PRONOUN_TOKENS:
                # The query position holds a pronoun rather than a literal
                # known entity (e.g. "... He reports to" after lead-in
                # filler). The caller is responsible for resolving it via
                # recency before doing a graph lookup.
                return candidate_entity, candidate_relation
    return None


def detect_open_query(
    trailing_text: str,
    known_relations: set[str],
    known_entities: set[str],
) -> tuple[str, str] | None:
    """Find a dangling "<entity> <relation phrase>" at the end of the prompt.

    This does not assume any fixed cue phrase (like "chain complete" or
    "therefore"): it scans for the longest known relation phrase that occupies
    the tail of the trailing text, immediately preceded by a chunk that is
    already a known entity from the stated facts (which may itself be a
    multi-word entity). That makes arbitrary lead-in wording (or none at all)
    irrelevant to detection.
    """
    return _match_open_query(_chunk_clause(trailing_text), known_relations, known_entities)


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

    def known_entities(self) -> set[str]:
        entities: set[str] = set()
        for store in (self._positive, self._negated):
            for subjects in store.values():
                entities.update(subjects.keys())
                for objects in subjects.values():
                    entities.update(objects)
        return entities

    def known_relations(self) -> set[str]:
        return set(self._positive.keys()) | set(self._negated.keys())

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
        # Path-scoped, not relation-wide: only a contradiction that actually
        # lies on the edge(s) walked while resolving *this* subject's query
        # blocks confident resolution. A contradiction stated for a
        # completely unrelated subject under the same relation type no
        # longer declines to answer for an uncontested subject.
        negated_pairs = self.contradictions_for(relation)
        current = start
        path: list[str] = [start]
        ambiguous = False
        candidates: frozenset[str] = frozenset()
        hops = 0
        moved = False
        contradiction_on_path = False
        while hops < max_hops:
            successors = self.successors(current, relation) - {current}
            if not successors:
                break
            if len(successors) > 1:
                ambiguous = True
                candidates = frozenset(successors)
                if any((current, candidate) in negated_pairs for candidate in successors):
                    contradiction_on_path = True
                break
            next_entity = next(iter(successors))
            if (current, next_entity) in negated_pairs:
                contradiction_on_path = True
            if next_entity in path:
                break  # cycle guard
            current = next_entity
            path.append(current)
            moved = True
            hops += 1

        # A subject with zero outgoing edges under this relation has nothing
        # to resolve to; it must not be reported as "resolved to itself"
        # (that previously masked genuinely unresolved queries and silently
        # blocked the world-knowledge fallback from ever being consulted).
        if ambiguous:
            resolved_target = None
        elif moved:
            resolved_target = current
            candidates = frozenset({resolved_target})
        else:
            resolved_target = None

        return TransitiveResult(
            start=start,
            relation=relation,
            resolved_target=resolved_target,
            candidate_targets=candidates,
            ambiguous=ambiguous,
            contradiction_detected=contradiction_on_path,
            path=tuple(path),
            fact_count=sum(len(objs) for subjects in self._positive.values() for objs in subjects.values())
            + sum(len(objs) for subjects in self._negated.values() for objs in subjects.values()),
        )


def load_world_knowledge_graph(path: Path | None = None) -> RelationGraph:
    """Load a persistent, external, symbolic fact store as a `RelationGraph`.

    This is real, explicit, inspectable world knowledge (a JSON file of
    subject/relation/object triples you can read and extend) consulted as a
    fallback when the current prompt doesn't establish a fact itself. It is
    NOT a claim of comprehensive world knowledge: it only knows what has been
    added to the file. The shipped default file is a modest, hand-curated
    starter set meant to demonstrate the mechanism/seam, not to be exhaustive.
    """
    target = path or DEFAULT_WORLD_KNOWLEDGE_PATH
    graph = RelationGraph()
    if not target.exists():
        return graph
    payload = json.loads(target.read_text(encoding="utf-8"))
    for entry in payload.get("facts", []):
        subject = str(entry["subject"])
        relation = _relation_key(str(entry["relation"]))
        obj = str(entry["object"])
        if bool(entry.get("negated", False)):
            graph.add_negated(subject, relation, obj)
        else:
            graph.add_positive(subject, relation, obj)
    return graph


_WORLD_KNOWLEDGE_CACHE: RelationGraph | None = None


def get_default_world_knowledge_graph() -> RelationGraph:
    global _WORLD_KNOWLEDGE_CACHE
    if _WORLD_KNOWLEDGE_CACHE is None:
        _WORLD_KNOWLEDGE_CACHE = load_world_knowledge_graph()
    return _WORLD_KNOWLEDGE_CACHE


def resolve_query(prompt_text: str, *, world_knowledge: RelationGraph | None = None) -> TransitiveResult | None:
    """Parse a prompt's stated facts and resolve any dangling open query.

    Pipeline: chunk each clause (supporting multi-word quoted/capitalized
    entities) -> resolve pronouns to their nearest preceding mention ->
    extract subject/relation/object facts -> detect the open query at the end
    of the prompt -> resolve it against in-context facts first, falling back
    to the external world-knowledge graph only if the context alone cannot
    resolve it.

    Returns None if the prompt has no open query, or the query's relation/
    entity was never established anywhere (context or world knowledge), in
    which case there is genuinely nothing to resolve and the caller should
    fall back to the base model rather than guess.
    """
    raw_clauses = [clause.strip() for clause in prompt_text.strip().split(".")]
    if not raw_clauses:
        return None

    chunked_clauses = [_chunk_clause(clause) for clause in raw_clauses]
    trailing_chunks = chunked_clauses[-1]
    stated_chunks_raw = chunked_clauses[:-1]

    # Coreference resolution for stated facts assumes the strict "SUBJECT
    # relation OBJECT" grammar (chunk 0 / chunk -1 are the entity slots). The
    # trailing query clause may have arbitrary lead-in filler before its
    # entity, so it is resolved separately below via `_match_open_query`'s
    # `allow_pronoun` mode plus the recency list built here.
    resolved_stated_chunks, recency, coreference_applied = _resolve_stated_clauses(stated_chunks_raw)

    graph = RelationGraph()
    for chunks in resolved_stated_chunks:
        parsed = _parse_chunks(chunks)
        if parsed is None:
            continue
        if parsed.negated:
            graph.add_negated(parsed.subject, parsed.relation, parsed.obj)
        else:
            graph.add_positive(parsed.subject, parsed.relation, parsed.obj)

    if not trailing_chunks:
        return None

    wkb = world_knowledge if world_knowledge is not None else get_default_world_knowledge_graph()

    def _resolve_query_subject(candidate: str) -> tuple[str, bool]:
        """Resolve a query-position candidate that may itself be a pronoun."""
        if candidate.lower() in PRONOUN_TOKENS:
            antecedent = _nearest_antecedent(candidate, recency)
            if antecedent is None:
                return candidate, False
            return antecedent, True
        return candidate, False

    # Prefer resolving strictly from in-context facts first.
    context_query = _match_open_query(
        trailing_chunks, graph.known_relations(), graph.known_entities(), allow_pronoun=True
    )
    if context_query is not None:
        candidate_subject, relation = context_query
        subject, used_coreference = _resolve_query_subject(candidate_subject)
        result = graph.transitive_closure(subject, relation)
        result = replace(result, source="context", coreference_applied=coreference_applied or used_coreference)
        if result.resolved_target is not None or result.ambiguous or result.contradiction_detected:
            return result

    # Broaden the query-detection vocabulary to include world-knowledge
    # relations/entities, then prefer context resolution and fall back to
    # world knowledge only if context has nothing for that (subject, relation).
    combined_relations = graph.known_relations() | wkb.known_relations()
    combined_entities = graph.known_entities() | wkb.known_entities()
    broadened_query = _match_open_query(
        trailing_chunks, combined_relations, combined_entities, allow_pronoun=True
    )
    if broadened_query is None:
        return None

    candidate_subject, relation = broadened_query
    subject, used_coreference = _resolve_query_subject(candidate_subject)
    coreference_applied = coreference_applied or used_coreference

    context_result = graph.transitive_closure(subject, relation)
    if context_result.resolved_target is not None or context_result.ambiguous or context_result.contradiction_detected:
        return replace(context_result, source="context", coreference_applied=coreference_applied)

    world_result = wkb.transitive_closure(subject, relation)
    return replace(world_result, source="world_knowledge", coreference_applied=coreference_applied)
