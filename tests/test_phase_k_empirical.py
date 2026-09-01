from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "experiments" / "phase_k" / "fixtures" / "multi_hop"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

from experiments.phase_k.run_phase_k_empirical import _measure_depth


def test_empirical_harness_recovers_high_in_sample_accuracy() -> None:
    """Sanity check: the real runtime must be able to memorize small training sets.

    This guards against the harness silently measuring nothing (e.g. a tokenizer or
    offset bug) rather than a genuine architectural limitation.
    """
    from real_chain_corpus import generate_trials
    from outreachlm.phase_g_bridge import PhaseGHybridConfig, PhaseGHybridRuntime, WordTokenizer
    from outreachlm.phase_h_runtime import BoundedStateRuntime, PhaseHRuntimeConfig
    import numpy as np

    trials = generate_trials(1, count=20, seed=1)
    train_lines = [t.training_line for t in trials]
    tokenizer = WordTokenizer.from_lines(train_lines)
    phase_g_runtime = PhaseGHybridRuntime(tokenizer=tokenizer, config=PhaseGHybridConfig())
    phase_g_runtime.fit(train_lines)
    runtime = BoundedStateRuntime(phase_g_runtime, config=PhaseHRuntimeConfig())

    correct = 0
    for trial in trials:
        prefix_tokens = runtime.tokenizer.encode(trial.query_prefix, add_bos=True, add_eos=False)
        probabilities = runtime._distribution(prefix_tokens, recent_tokens=prefix_tokens[-64:], apply_safety=False)
        predicted_token = runtime.tokenizer.decode([int(np.argmax(probabilities))], skip_special_tokens=False)
        correct += 1 if predicted_token == trial.expected_final_entity else 0

    assert correct / len(trials) >= 0.8


def test_empirical_k1_measures_real_generalization_gap_at_shallow_depths() -> None:
    """The frozen n-gram architecture has no copy/attention mechanism, so it cannot
    generalize the transitive-chain completion task to unseen entity permutations,
    even at the shallowest tested depth. This test locks in that honest finding so a
    future change to the measurement harness (not the architecture) is caught."""
    result = _measure_depth(1)
    assert result["eval_trial_count"] == 60
    assert result["argmax_next_token_accuracy"] < 0.5
