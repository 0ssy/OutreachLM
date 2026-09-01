from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_k_reasoning.relation_engine import (
    RelationGraph,
    _chunk_plain_text,
    resolve_query,
)


# --- Fix: sentence-initial capitalized determiner is not treated as an entity


def test_chunk_plain_text_drops_standalone_sentence_initial_determiner() -> None:
    assert _chunk_plain_text("The engine causes damage") == ["engine", "causes", "damage"]


def test_chunk_plain_text_keeps_determiner_inside_a_real_proper_noun_run() -> None:
    # "The" followed by more capitalized words is a legitimate part of a
    # multi-word proper noun (e.g. a title), so it must NOT be dropped here.
    assert _chunk_plain_text("The New York Times reported it") == [
        "The New York Times",
        "reported",
        "it",
    ]


def test_resolve_query_resolves_subject_past_a_sentence_initial_determiner() -> None:
    result = resolve_query("The engine causes damage . summary engine causes")
    assert result is not None
    assert result.start == "engine"
    assert result.resolved_target == "damage"


# --- Fix: lowercase connector words inside a multi-word proper noun --------


def test_chunk_plain_text_keeps_connector_word_inside_capitalized_run() -> None:
    assert _chunk_plain_text("Bank of America reported profit") == [
        "Bank of America",
        "reported",
        "profit",
    ]


def test_chunk_plain_text_drops_trailing_connector_not_followed_by_capital() -> None:
    # The connector must only be swallowed when it actually continues a
    # capitalized run; otherwise it should not silently disappear.
    assert _chunk_plain_text("Paris of course is nice") == ["Paris", "of", "course", "is", "nice"]


def test_resolve_query_handles_connector_word_multiword_entity() -> None:
    prompt = "Bank of America is located in America . summary Bank of America is located in"
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "Bank of America"
    assert result.resolved_target == "America"


# --- Fix: negation contractions (doesn't/isn't/won't/...) ------------------


def test_resolve_query_detects_contraction_based_contradiction() -> None:
    prompt = "alice trusts bob . alice doesn't trust bob . summary alice trusts"
    result = resolve_query(prompt)
    assert result is not None
    assert result.contradiction_detected is True


def test_resolve_query_contraction_matches_positive_relation_key() -> None:
    # A negated clause using a contraction must canonicalize to the exact
    # same relation key as its positive counterpart.
    prompt = "alice knows bob . alice doesn't know charlie . summary alice knows"
    result = resolve_query(prompt)
    assert result is not None
    assert result.resolved_target == "bob"
    assert result.contradiction_detected is False


# --- Fix: irregular verb reconciliation (has/have, is/are/was/were) --------


def test_resolve_query_reconciles_has_have_irregular_form() -> None:
    prompt = "alice has bob . alice does not have bob . summary alice has"
    result = resolve_query(prompt)
    assert result is not None
    assert result.contradiction_detected is True


# --- Fix: registered relation synonyms -------------------------------------


def test_resolve_query_matches_registered_relation_synonym() -> None:
    # Fact stated with "links" (canonicalizes to "link"); queried with the
    # registered synonym phrasing "connects to" (also canonicalizes to "link").
    prompt = "alpha links beta . summary alpha connects to"
    result = resolve_query(prompt)
    assert result is not None
    assert result.resolved_target == "beta"


def test_resolve_query_unregistered_paraphrase_still_treated_as_distinct() -> None:
    # A paraphrase NOT in the synonym table must not be silently matched;
    # this keeps the mechanism honest about its bounded scope.
    prompt = "alpha links beta . summary alpha is somehow associated with"
    result = resolve_query(prompt)
    assert result is None


# --- Fix: gender-aware nearest-mention pronoun resolution -------------------


def test_resolve_query_gendered_pronoun_skips_wrong_gender_nearest_mention() -> None:
    # Plain nearest-mention would pick "Bob" (the object of the prior
    # clause); gender-aware resolution must skip it for a feminine pronoun
    # and continue searching for "Alice".
    prompt = "Alice manages Bob . summary she manages"
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "Alice"
    assert result.resolved_target == "Bob"
    assert result.coreference_applied is True


def test_resolve_query_masculine_pronoun_skips_wrong_gender_nearest_mention() -> None:
    prompt = "Bob manages Alice . summary he manages"
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "Bob"
    assert result.resolved_target == "Alice"


def test_resolve_query_unknown_name_falls_back_to_plain_recency() -> None:
    # "Zorblax" has no gender hint; gendered pronoun resolution must not
    # error out, it should fall back to plain nearest-mention.
    prompt = "Zorblax manages Bob . summary she manages"
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "Bob"


# --- Fix: path-scoped (not relation-wide) contradiction detection ----------


def test_contradiction_for_one_subject_does_not_block_a_different_subject() -> None:
    prompt = "alice trusts bob . alice doesn't trust bob . charlie trusts dana . summary charlie trusts"
    result = resolve_query(prompt)
    assert result is not None
    assert result.start == "charlie"
    assert result.resolved_target == "dana"
    assert result.contradiction_detected is False


def test_contradiction_still_blocks_the_actually_contested_subject() -> None:
    prompt = "alice trusts bob . alice doesn't trust bob . charlie trusts dana . summary alice trusts"
    result = resolve_query(prompt)
    assert result is not None
    assert result.contradiction_detected is True


# --- Fix: expanded world-knowledge coverage --------------------------------


def test_expanded_world_knowledge_covers_multiple_categories() -> None:
    assert resolve_query("background check doctor uses").resolved_target == "stethoscope"
    assert resolve_query("background check earth orbits").resolved_target == "sun"
    assert resolve_query("background check bees produce").resolved_target == "honey"
