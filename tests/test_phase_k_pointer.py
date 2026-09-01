from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "experiments" / "phase_k" / "fixtures" / "multi_hop"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

import pytest

from outreachlm.phase_g_bridge import PhaseGHybridConfig, PhaseGHybridRuntime, WordTokenizer
from outreachlm.phase_h_runtime import BoundedStateRuntime, PhaseHRuntimeConfig
from outreachlm.phase_k_pointer_runtime import PointerAugmentedConfig, PointerAugmentedRuntime
from src.phase_k_reasoning.pointer import (
    detect_query_anchor,
    extract_link_facts,
    resolve_pointer,
    resolve_transitive_target,
)


def test_extract_link_facts_parses_all_stated_links() -> None:
    facts = extract_link_facts("nodeA links nodeB . nodeB links nodeC . unrelated text here")
    assert facts == {"nodeA": "nodeB", "nodeB": "nodeC"}


def test_detect_query_anchor_only_matches_completion_queries() -> None:
    assert detect_query_anchor("nodeA links nodeB . chain complete nodeA links") == "nodeA"
    assert detect_query_anchor("nodeA links nodeB .") is None
    assert detect_query_anchor("") is None


def test_resolve_transitive_target_follows_multi_hop_chain() -> None:
    facts = {"A": "B", "B": "C", "C": "D"}
    assert resolve_transitive_target(facts, "A") == "D"
    assert resolve_transitive_target(facts, "Z") is None


def test_resolve_transitive_target_guards_against_cycles() -> None:
    facts = {"A": "B", "B": "A"}
    # Must terminate rather than looping forever.
    result = resolve_transitive_target(facts, "A")
    assert result in {"A", "B"}


def test_resolve_pointer_end_to_end() -> None:
    prompt = "node034 links node145 . node145 links node200 . chain complete node034 links"
    resolution = resolve_pointer(prompt)
    assert resolution.anchor_entity == "node034"
    assert resolution.resolved_target == "node200"
    assert resolution.resolved is True


def test_resolve_pointer_returns_unresolved_for_non_query_prompts() -> None:
    resolution = resolve_pointer("just some ordinary text with no query pattern")
    assert resolution.resolved is False
    assert resolution.anchor_entity is None


@pytest.fixture()
def small_pointer_runtime() -> PointerAugmentedRuntime:
    train_lines = [
        "alpha links beta . chain complete alpha links beta .",
        "gamma links delta . chain complete gamma links delta .",
    ]
    # The tokenizer's vocabulary must include tokens for entities that only
    # ever appear at evaluation time (never fit into the n-gram counts), the
    # same way the real empirical harness builds vocab from train+eval lines.
    vocab_only_lines = ["epsilon links zeta . chain complete epsilon links zeta ."]
    tokenizer = WordTokenizer.from_lines(train_lines + vocab_only_lines)
    phase_g_runtime = PhaseGHybridRuntime(tokenizer=tokenizer, config=PhaseGHybridConfig())
    phase_g_runtime.fit(train_lines)
    base_runtime = BoundedStateRuntime(phase_g_runtime, config=PhaseHRuntimeConfig())
    return PointerAugmentedRuntime(base_runtime, config=PointerAugmentedConfig())


def test_pointer_augmented_runtime_uses_pointer_for_unseen_chain(
    small_pointer_runtime: PointerAugmentedRuntime,
) -> None:
    # "epsilon links zeta" was never seen during training; only the pointer
    # mechanism (not memorized n-gram counts) can solve this.
    prompt = "epsilon links zeta . chain complete epsilon links"
    prediction = small_pointer_runtime.predict_next(prompt)
    assert prediction.used_pointer is True
    assert prediction.predicted_token == "zeta"


def test_pointer_augmented_runtime_falls_back_when_no_query_pattern(
    small_pointer_runtime: PointerAugmentedRuntime,
) -> None:
    prediction = small_pointer_runtime.predict_next("alpha links beta")
    assert prediction.used_pointer is False


def test_pointer_augmented_runtime_falls_back_when_anchor_has_no_fact(
    small_pointer_runtime: PointerAugmentedRuntime,
) -> None:
    prediction = small_pointer_runtime.predict_next("chain complete unknown_entity links")
    assert prediction.used_pointer is False
