from __future__ import annotations

from pathlib import Path

from outreachlm.phase_h_runtime import BoundedStateRuntime
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT

from src.phase_i_semantic.config_loader import load_phase_i_config


def load_runtime() -> BoundedStateRuntime:
    artifact = Path("models") / "outreachlm_phase_h_runtime.pkl"
    if artifact.exists():
        return BoundedStateRuntime.load(artifact)
    runtime, _, _ = BoundedStateRuntime.from_corpus_path(
        corpus_path=CORPUS_PATH,
        validation_split=VALIDATION_SPLIT,
        max_train_lines=1200,
        max_eval_lines=160,
    )
    runtime.save(artifact)
    return runtime


def get_config() -> dict:
    return load_phase_i_config()
