from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from outreachlm.phase_g_bridge import PhaseGHybridRuntime
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.safety.cycle_detection import max_repetition_run, repeated_ngram_count
from src.phase_h_cache.safety.repetition import apply_repetition_penalty


def _topk_prune(probabilities: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_k >= len(probabilities):
        return probabilities / probabilities.sum()
    idx = np.argpartition(probabilities, -top_k)[-top_k:]
    out = np.zeros_like(probabilities, dtype=np.float64)
    out[idx] = probabilities[idx]
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum()
    return out


def _trial(
    runtime: PhaseGHybridRuntime,
    *,
    prompt_tokens: list[int],
    max_tokens: int,
    mode: str,
    top_k: int,
    rng_seed: int,
) -> dict[str, Any]:
    rng = random.Random(rng_seed)
    context = list(prompt_tokens)
    generated: list[int] = []
    entropies: list[float] = []
    mass_errors: list[float] = []

    for _ in range(max_tokens):
        probs = runtime.distribution(context)
        if mode in {"repetition", "combined"}:
            probs = apply_repetition_penalty(probs, generated[-64:])
        if mode in {"pruning", "combined"}:
            probs = _topk_prune(probs, top_k=top_k)
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum()
        mass_errors.append(abs(float(probs.sum()) - 1.0))
        entropies.append(float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()))

        if mode == "baseline":
            next_id = int(np.argmax(probs))
        else:
            next_id = int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])
        generated.append(next_id)
        context.append(next_id)

    return {
        "mode": mode,
        "generated_count": len(generated),
        "max_repetition_run": int(max_repetition_run(generated)),
        "repeated_bigram_count": int(repeated_ngram_count(generated, 2)),
        "entropy_mean": float(np.mean(np.asarray(entropies, dtype=np.float64))),
        "mass_loss_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
        "tokens": generated,
    }


def run() -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw["h9"]
    runtime, _, _ = PhaseGHybridRuntime.from_corpus_path(
        CORPUS_PATH,
        validation_split=VALIDATION_SPLIT,
        max_train_lines=int(config["train_lines"]),
        max_eval_lines=int(config["eval_lines"]),
    )
    prompt = runtime.tokenizer.encode("the bank", add_bos=True, add_eos=False)
    max_tokens = int(config["generation_tokens"])
    top_k = int(config["pruning_top_k"])
    seed = int(config["seed"])

    baseline = _trial(runtime, prompt_tokens=prompt, max_tokens=max_tokens, mode="baseline", top_k=top_k, rng_seed=seed)
    repetition = _trial(
        runtime, prompt_tokens=prompt, max_tokens=max_tokens, mode="repetition", top_k=top_k, rng_seed=seed + 1
    )
    pruning = _trial(runtime, prompt_tokens=prompt, max_tokens=max_tokens, mode="pruning", top_k=top_k, rng_seed=seed + 2)
    combined = _trial(runtime, prompt_tokens=prompt, max_tokens=max_tokens, mode="combined", top_k=top_k, rng_seed=seed + 3)

    entropy_shift = 0.0
    if baseline["entropy_mean"] > 0.0:
        entropy_shift = (combined["entropy_mean"] - baseline["entropy_mean"]) / baseline["entropy_mean"]

    return {
        "experiment_id": "h9_safety_audit",
        "config": config,
        "trials": {
            "baseline": baseline,
            "repetition": repetition,
            "pruning": pruning,
            "combined": combined,
        },
        "baseline_repetition_run_length": int(baseline["max_repetition_run"]),
        "governed_repetition_run_length": int(combined["max_repetition_run"]),
        "safety_layer_mass_loss_error": float(combined["mass_loss_error_max"]),
        "sequence_entropy_shift_rate": float(entropy_shift),
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "deep_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h9_safety_audit.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
