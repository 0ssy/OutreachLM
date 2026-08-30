from __future__ import annotations

import numpy as np

from outreachlm.phase_g_bridge import PhaseGHybridRuntime


def _lines() -> list[str]:
    return [
        "the cat sat on the mat",
        "the cat ate the fish",
        "the dog sat on the mat",
        "the dog ate the food",
        "the boy saw the dog",
        "the girl saw the cat",
        "the girl fed the dog",
        "the boy fed the cat",
    ]


def test_phase_g_bridge_fit_and_evaluate() -> None:
    runtime, _, eval_lines = PhaseGHybridRuntime.from_corpus_lines(_lines(), seed=1337, eval_ratio=0.25)
    metrics = runtime.evaluate_lines(eval_lines)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["cross_entropy"] > 0.0
    assert metrics["perplexity"] > 0.0
    assert metrics["mass_error_max"] <= 1e-6
    assert metrics["operations_per_token_estimate"] > 0.0


def test_phase_g_bridge_save_load_round_trip(tmp_path) -> None:
    runtime, _, _ = PhaseGHybridRuntime.from_corpus_lines(_lines(), seed=1337, eval_ratio=0.25)
    artifact_path = tmp_path / "phase_g.pkl"
    runtime.save(artifact_path)

    loaded = PhaseGHybridRuntime.load(artifact_path)
    context = loaded.tokenizer.encode("the cat", add_bos=True, add_eos=False)
    expected = runtime.distribution(context)
    actual = loaded.distribution(context)

    assert np.allclose(expected, actual)
    assert abs(float(actual.sum()) - 1.0) <= 1e-6


def test_phase_g_bridge_handles_oov_path() -> None:
    runtime, _, _ = PhaseGHybridRuntime.from_corpus_lines(_lines(), seed=1337, eval_ratio=0.25)
    context = runtime.tokenizer.encode("the quokka sat", add_bos=True, add_eos=False)
    distribution = runtime.distribution(context)

    assert distribution[runtime.tokenizer.unk_id] > 0.0
    assert abs(float(distribution.sum()) - 1.0) <= 1e-6
