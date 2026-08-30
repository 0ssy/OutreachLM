from __future__ import annotations

from outreachlm.phase_g_bridge import PhaseGHybridRuntime
from outreachlm.phase_h_runtime import BoundedStateRuntime, PhaseHRuntimeConfig


def _lines() -> list[str]:
    return [
        "the cat sat on the mat",
        "the cat ate the fish",
        "the dog sat on the mat",
        "the dog ate the food",
        "the girl saw the cat",
        "the boy saw the dog",
        "the boy fed the cat",
        "the girl fed the dog",
    ]


def test_phase_h_runtime_evaluate_and_generate() -> None:
    phase_g, train_lines, eval_lines = PhaseGHybridRuntime.from_corpus_lines(_lines(), seed=1337, eval_ratio=0.25)
    runtime = BoundedStateRuntime(
        phase_g_runtime=phase_g,
        config=PhaseHRuntimeConfig(quantization_mode="fp16"),
    )

    metrics = runtime.evaluate_lines(eval_lines, apply_safety=False)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["cross_entropy"] > 0.0
    assert metrics["mass_error_max"] <= 1e-6

    generation = runtime.generate("the cat", max_new_tokens=20, top_k=8, apply_safety=True, seed=7)
    assert isinstance(generation["generated_text"], str)
    assert len(generation["generated_token_ids"]) > 0
    assert generation["max_repetition_run"] >= 1


def test_phase_h_runtime_save_load_and_ingest(tmp_path) -> None:
    phase_g, _, _ = PhaseGHybridRuntime.from_corpus_lines(_lines(), seed=1337, eval_ratio=0.25)
    runtime = BoundedStateRuntime(phase_g_runtime=phase_g)
    artifact_path = tmp_path / "phase_h_runtime.pkl"
    runtime.save(artifact_path)

    loaded = BoundedStateRuntime.load(artifact_path)
    ingest = loaded.ingest_lines(["the cat saw the bird", "the dog saw the park"])
    assert ingest["lines_ingested"] == 2.0
    assert ingest["transitions_ingested"] > 0.0
