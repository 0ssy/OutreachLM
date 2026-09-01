from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from outreachlm.phase_g_bridge import PhaseGHybridConfig, PhaseGHybridRuntime, WordTokenizer
from outreachlm.phase_h_runtime import BoundedStateRuntime, PhaseHRuntimeConfig
from outreachlm.phase_k_pointer_runtime import PointerAugmentedConfig, PointerAugmentedRuntime
from src.phase_k_reasoning.relation_engine import (
    RelationGraph,
    _chunk_clause,
    resolve_query,
)


# --- Multi-word entities: capitalized proper-noun runs ----------------------


def test_chunk_clause_groups_consecutive_capitalized_words() -> None:
    assert _chunk_clause("New York City is located in France") == [
        "New York City",
        "is",
        "located",
        "in",
        "France",
    ]


def test_resolve_query_handles_capitalized_multiword_entities() -> None:
    prompt = "New York City is located in France . summary New York City is located in"
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "New York City"
    assert result.resolved_target == "France"


# --- Multi-word entities: quoted spans --------------------------------------


def test_chunk_clause_treats_quoted_span_as_one_entity() -> None:
    assert _chunk_clause('"the red door" links "the old bridge"') == [
        "the red door",
        "links",
        "the old bridge",
    ]


def test_resolve_query_handles_quoted_multiword_entities() -> None:
    prompt = '"the red door" links "the old bridge" . chain complete "the red door" links'
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "the red door"
    assert result.resolved_target == "the old bridge"


# --- Pronoun / coreference resolution (nearest-preceding-mention) -----------


def test_resolve_query_resolves_pronoun_in_stated_clause() -> None:
    # "He" (subject position of the second stated clause) should resolve to
    # "Bob", the nearest preceding mention (the object of the first clause).
    prompt = "Alice manages Bob . He supervises Charlie . summary Bob supervises"
    result = resolve_query(prompt)
    assert result is not None
    assert result.resolved_target == "Charlie"
    assert result.coreference_applied is True


def test_resolve_query_resolves_pronoun_in_the_query_clause_itself() -> None:
    # The pronoun sits in the final, filler-prefixed query clause itself
    # (not just in an earlier stated clause) and must still resolve via the
    # same nearest-preceding-mention recency list.
    prompt = "Bob supervises Charlie . Alice manages Bob . he supervises"
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "Bob"
    assert result.resolved_target == "Charlie"
    assert result.coreference_applied is True


def test_resolve_query_without_pronouns_reports_no_coreference_applied() -> None:
    prompt = "alpha links beta . chain complete alpha links"
    result = resolve_query(prompt)
    assert result is not None
    assert result.coreference_applied is False


# --- External world-knowledge fallback ---------------------------------------


def test_resolve_query_falls_back_to_world_knowledge_when_context_has_nothing() -> None:
    # No in-context fact establishes this at all; only the shipped starter
    # world-knowledge file does.
    result = resolve_query("background check engineer uses")
    assert result is not None
    assert result.resolved_target == "computer"
    assert result.source == "world_knowledge"


def test_resolve_query_prefers_in_context_facts_over_world_knowledge() -> None:
    # The prompt overrides/extends the same relation in-context; context must
    # win over the (unrelated) world-knowledge entry for a different subject.
    custom_wkb = RelationGraph()
    custom_wkb.add_positive("engineer", "use", "slide rule")
    result = resolve_query("engineer uses spreadsheet . summary engineer uses", world_knowledge=custom_wkb)
    assert result is not None
    assert result.source == "context"
    assert result.resolved_target == "spreadsheet"


def test_resolve_query_returns_none_when_neither_context_nor_world_knowledge_helps() -> None:
    empty_wkb = RelationGraph()
    result = resolve_query("completely unestablished nonsense query", world_knowledge=empty_wkb)
    assert result is None


# --- Runtime-level integration for all three ---------------------------------


@pytest.fixture()
def general_purpose_runtime() -> PointerAugmentedRuntime:
    train_lines = [
        "alpha links beta . chain complete alpha links beta .",
    ]
    vocab_only_lines = [
        "New York City is located in France . summary New York City is located in France .",
        # Quotes are spaced apart from the words they enclose so the frozen
        # whitespace tokenizer's vocabulary contains clean word entries (e.g.
        # "the", "red", "door") rather than punctuation-glued tokens like
        # '"the'. The relation engine's quote-aware chunking itself works
        # correctly either way (it scans at the character level), but the
        # underlying tokenizer's vocabulary is a plain whitespace split and
        # is not modified here since it is part of the frozen Phase G core.
        '" the red door " links " the old bridge " . chain complete " the red door " links " the old bridge " .',
        "Bob supervises Charlie . Alice manages Bob . he supervises Charlie .",
        "background check engineer uses computer .",
    ]
    tokenizer = WordTokenizer.from_lines(train_lines + vocab_only_lines)
    phase_g_runtime = PhaseGHybridRuntime(tokenizer=tokenizer, config=PhaseGHybridConfig())
    phase_g_runtime.fit(train_lines)
    base_runtime = BoundedStateRuntime(phase_g_runtime, config=PhaseHRuntimeConfig())
    return PointerAugmentedRuntime(base_runtime, config=PointerAugmentedConfig())


def test_runtime_resolves_multiword_capitalized_entity(
    general_purpose_runtime: PointerAugmentedRuntime,
) -> None:
    prompt = "New York City is located in France . summary New York City is located in"
    prediction = general_purpose_runtime.predict_next(prompt)
    assert prediction.used_pointer is True
    assert prediction.predicted_token == "France"


def test_runtime_completes_full_multiword_answer_via_complete_query(
    general_purpose_runtime: PointerAugmentedRuntime,
) -> None:
    prompt = "Bob supervises Charlie . Alice manages Bob . he supervises"
    completion = general_purpose_runtime.complete_query(prompt)
    assert completion.used_pointer is True
    assert completion.completion == "Charlie"


def test_runtime_resolves_quoted_multiword_entity(
    general_purpose_runtime: PointerAugmentedRuntime,
) -> None:
    prompt = '"the red door" links "the old bridge" . chain complete "the red door" links'
    # A multi-word answer cannot be expressed as a single next-token
    # prediction; `complete_query` deterministically resolves the full span.
    completion = general_purpose_runtime.complete_query(prompt)
    assert completion.used_pointer is True
    assert completion.completion == "the old bridge"

    # `predict_next` still correctly boosts the *first* word of that answer.
    prediction = general_purpose_runtime.predict_next(prompt)
    assert prediction.used_pointer is True
    assert prediction.predicted_token == "the"


def test_runtime_resolves_pronoun_query(general_purpose_runtime: PointerAugmentedRuntime) -> None:
    prompt = "Bob supervises Charlie . Alice manages Bob . he supervises"
    prediction = general_purpose_runtime.predict_next(prompt)
    assert prediction.used_pointer is True
    assert prediction.pointer.coreference_applied is True
    assert prediction.predicted_token == "Charlie"


def test_runtime_resolves_world_knowledge_fallback(
    general_purpose_runtime: PointerAugmentedRuntime,
) -> None:
    prediction = general_purpose_runtime.predict_next("background check engineer uses")
    assert prediction.used_pointer is True
    assert prediction.pointer.source == "world_knowledge"
    assert prediction.predicted_token == "computer"
