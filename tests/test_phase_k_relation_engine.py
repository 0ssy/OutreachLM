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
from src.phase_k_reasoning.relation_engine import RelationGraph, parse_clause, resolve_query


# --- Arbitrary relation phrasing (not just "links") -------------------------


def test_parse_clause_accepts_any_relation_phrase() -> None:
    assert parse_clause("alice reports to bob").relation == "reports to"
    assert parse_clause("event_a causes event_b").relation == "cause"
    assert parse_clause("paris is located in france").relation == "is located in"


def test_resolve_query_generalizes_to_unregistered_relation_phrasing() -> None:
    prompt = "site_a connects to site_b . site_b connects to site_c . network audit site_a connects to"
    result = resolve_query(prompt)
    assert result is not None
    assert result.resolved_target == "site_c"
    assert result.resolved is True


def test_resolve_query_ignores_arbitrary_lead_in_wording() -> None:
    # No fixed cue phrase ("chain complete", "therefore", ...) is required;
    # any lead-in text works as long as it doesn't overlap the relation span.
    prompt = "alpha causes beta . beta causes gamma . in our final assessment we conclude that alpha causes"
    result = resolve_query(prompt)
    assert result is not None
    assert result.resolved_target == "gamma"


# --- Branching / multi-valued relations (ambiguity, not a silent guess) -----


def test_transitive_closure_reports_ambiguous_branch_instead_of_guessing() -> None:
    graph = RelationGraph()
    graph.add_positive("hub", "routes to", "east")
    graph.add_positive("hub", "routes to", "west")
    result = graph.transitive_closure("hub", "routes to")
    assert result.ambiguous is True
    assert result.resolved_target is None
    assert result.candidate_targets == frozenset({"east", "west"})


def test_resolve_query_surfaces_branching_from_a_full_prompt() -> None:
    prompt = "hub routes to east . hub routes to west . dispatch hub routes to"
    result = resolve_query(prompt)
    assert result is not None
    assert result.ambiguous is True
    assert result.candidate_targets == frozenset({"east", "west"})


# --- Contradiction detection (explicit negation of an asserted fact) -------


def test_contradictions_for_detects_asserted_and_negated_same_triple() -> None:
    graph = RelationGraph()
    graph.add_positive("alice", "trusts", "bob")
    graph.add_negated("alice", "trusts", "bob")
    assert ("alice", "bob") in graph.contradictions_for("trusts")


def test_resolve_query_flags_contradiction_and_declines_to_resolve() -> None:
    prompt = "alice trusts bob . alice does not trust bob . summary alice trusts"
    result = resolve_query(prompt)
    assert result is not None
    assert result.contradiction_detected is True


# --- Multiple independent relations tracked in the same context ------------


def test_multiple_relations_do_not_interfere_with_each_other() -> None:
    prompt = (
        "engineA causes eventB . engineA feeds turbineC . "
        "diagnostics engineA causes"
    )
    result = resolve_query(prompt)
    assert result is not None
    assert result.relation == "cause"
    assert result.resolved_target == "eventB"


# --- Runtime-level behaviour for ambiguity and contradiction ---------------


@pytest.fixture()
def general_pointer_runtime() -> PointerAugmentedRuntime:
    train_lines = [
        "alpha causes beta . chain complete alpha causes beta .",
        "gamma reports to delta . chain complete gamma reports to delta .",
    ]
    vocab_only_lines = [
        "hub routes to east . hub routes to west . dispatch hub routes to east .",
        "alice trusts bob . alice does not trust bob . summary alice trusts bob .",
    ]
    tokenizer = WordTokenizer.from_lines(train_lines + vocab_only_lines)
    phase_g_runtime = PhaseGHybridRuntime(tokenizer=tokenizer, config=PhaseGHybridConfig())
    phase_g_runtime.fit(train_lines)
    base_runtime = BoundedStateRuntime(phase_g_runtime, config=PhaseHRuntimeConfig())
    return PointerAugmentedRuntime(base_runtime, config=PointerAugmentedConfig())


def test_runtime_resolves_unregistered_relation_phrasing(
    general_pointer_runtime: PointerAugmentedRuntime,
) -> None:
    prediction = general_pointer_runtime.predict_next("gamma reports to delta . chain complete gamma reports to")
    assert prediction.used_pointer is True
    assert prediction.predicted_token == "delta"


def test_runtime_splits_confidence_across_branch_candidates(
    general_pointer_runtime: PointerAugmentedRuntime,
) -> None:
    prompt = "hub routes to east . hub routes to west . dispatch hub routes to"
    prediction = general_pointer_runtime.predict_next(prompt)
    assert prediction.used_pointer is True
    assert prediction.pointer.ambiguous is True
    east_id = general_pointer_runtime.runtime.tokenizer.token_to_id["east"]
    west_id = general_pointer_runtime.runtime.tokenizer.token_to_id["west"]
    # Mass should be split roughly evenly between the two candidates, not
    # concentrated entirely on one.
    assert prediction.probabilities[east_id] > 0.3
    assert prediction.probabilities[west_id] > 0.3


def test_runtime_declines_to_answer_on_contradiction(
    general_pointer_runtime: PointerAugmentedRuntime,
) -> None:
    prompt = "alice trusts bob . alice does not trust bob . summary alice trusts"
    prediction = general_pointer_runtime.predict_next(prompt)
    assert prediction.used_pointer is False
    assert prediction.pointer.contradiction_detected is True
