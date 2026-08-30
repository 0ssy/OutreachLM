from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from outreachlm.phase_g_bridge import PhaseGHybridRuntime
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT

from src.phase_h_cache import PhaseHConfig


def _tv_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _build_noisy_sequence(length: int, *, anchor: str, seed: int) -> str:
    rng = random.Random(seed)
    distractors = [
        "the", "and", "of", "with", "context", "shift", "topic", "entity", "reference", "nested",
        "open", "close", "bank", "signal", "residual", "noise", "track", "history",
    ]
    words = [anchor, "bank"]
    while len(words) < length:
        words.append(distractors[rng.randint(0, len(distractors) - 1)])
    return " ".join(words[:length])


def _select_anchor_suffixes(runtime: PhaseGHybridRuntime, eval_lines: list[str]) -> tuple[list[int], list[int], float]:
    contexts: list[list[int]] = []
    for line in eval_lines:
        seq = runtime.tokenizer.encode(line, add_bos=True, add_eos=True)
        for pos in range(min(len(seq) - 1, 20)):
            context = seq[: pos + 1]
            if len(context) >= 4:
                contexts.append(context[-4:])
    if len(contexts) < 2:
        fallback = runtime.tokenizer.encode("the bank", add_bos=True, add_eos=False)
        return fallback[-4:], fallback[-4:], 0.0

    best_pair = (contexts[0], contexts[1], 0.0)
    sample_size = min(80, len(contexts))
    sampled = contexts[:sample_size]
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            left = sampled[i]
            right = sampled[j]
            p_left = runtime.distribution(left)
            p_right = runtime.distribution(right)
            delta = _tv_delta(p_left, p_right)
            if delta > best_pair[2]:
                best_pair = (left, right, delta)
    return best_pair


def run() -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw["h8"]
    runtime, _, eval_lines = PhaseGHybridRuntime.from_corpus_path(
        CORPUS_PATH,
        validation_split=VALIDATION_SPLIT,
        max_train_lines=int(config["train_lines"]),
        max_eval_lines=int(config["eval_lines"]),
    )

    lengths = [int(value) for value in config["context_lengths"]]
    gate = float(config["context_delta_gate"])
    seed = int(config["seed"])
    rows: list[dict[str, float]] = []
    left_suffix, right_suffix, base_delta = _select_anchor_suffixes(runtime, eval_lines)
    suffix_left_text = runtime.tokenizer.decode(left_suffix, skip_special_tokens=True).strip() or "the bank"
    suffix_right_text = runtime.tokenizer.decode(right_suffix, skip_special_tokens=True).strip() or "river bank"

    for length in lengths:
        finance_text = _build_noisy_sequence(length, anchor="contexta", seed=seed + length)
        river_text = _build_noisy_sequence(length, anchor="contextb", seed=seed + length + 1)
        finance_text = f"{finance_text} {suffix_left_text}"
        river_text = f"{river_text} {suffix_right_text}"
        finance_ctx = runtime.tokenizer.encode(finance_text, add_bos=True, add_eos=False)
        river_ctx = runtime.tokenizer.encode(river_text, add_bos=True, add_eos=False)
        p_fin = runtime.distribution(finance_ctx)
        p_riv = runtime.distribution(river_ctx)
        delta = _tv_delta(p_fin, p_riv)
        nested_tracking = 1.0 if delta >= gate else 0.0
        rows.append(
            {
                "context_length": float(length),
                "context_intervention_delta": delta,
                "nested_tracking": nested_tracking,
            }
        )

    graceful = [int(row["context_length"]) for row in rows if row["context_intervention_delta"] >= gate]
    maximum_graceful = max(graceful) if graceful else 0
    smearing_detected = 0
    for row in rows:
        if row["context_intervention_delta"] < gate:
            smearing_detected = int(row["context_length"])
            break

    nested_rate = float(np.mean(np.asarray([row["nested_tracking"] for row in rows], dtype=np.float64)))
    return {
        "experiment_id": "h8_long_context",
        "config": config,
        "anchor_delta_baseline": float(base_delta),
        "anchor_suffix_left": suffix_left_text,
        "anchor_suffix_right": suffix_right_text,
        "rows": rows,
        "maximum_graceful_context_sequence_limit": maximum_graceful,
        "context_smearing_detected_at_length": smearing_detected,
        "nested_structure_tracking_rate": nested_rate,
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "deep_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h8_long_context.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
