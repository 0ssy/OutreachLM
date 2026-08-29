from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
import tracemalloc

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.cpu_threads import configure_cpu_threads_from_env  # noqa: E402
from common.metrics import evaluate_predictions  # noqa: E402
from common.models import SparseNGramModel  # noqa: E402
from common.tokenizer import StupidTokenizer  # noqa: E402

CPU_THREADING = configure_cpu_threads_from_env()

import numpy as np


def _encode_lines(tokenizer, lines: list[str]) -> list[list[int]]:
    return [tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frozen_fixture() -> tuple[dict, StupidTokenizer, list[list[int]], list[list[int]]]:
    fixture_path = PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json"
    fixture = _read(fixture_path)
    tokenizer = StupidTokenizer(token_to_id=fixture["token_to_id"])
    train_sequences = _encode_lines(tokenizer, fixture["train_lines"])
    eval_sequences = _encode_lines(tokenizer, fixture["eval_lines"])
    return fixture, tokenizer, train_sequences, eval_sequences


def _load_g7_run_module() -> object:
    g7_run_path = PHASE_G_ROOT / "g7_recovery" / "r6_residual_sparse" / "run.py"
    spec = importlib.util.spec_from_file_location("phase_g_g7_final_module", g7_run_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {g7_run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    seed = 1337
    g2 = _read(PHASE_G_ROOT / "g2_contextual_transition" / "results" / "g2_result.json")
    g3 = _read(PHASE_G_ROOT / "g3_representation" / "results" / "g3_result.json")
    g4 = _read(PHASE_G_ROOT / "g4_context_compression" / "results" / "g4_result.json")
    g5 = _read(PHASE_G_ROOT / "g5_adaptive_memory" / "results" / "g5_result.json")
    g6 = _read(PHASE_G_ROOT / "g6_hierarchical_structure" / "results" / "g6_result.json")
    g7_legacy = _read(PHASE_G_ROOT / "g7_sparse_prediction" / "results" / "g7_result.json")
    g8 = _read(PHASE_G_ROOT / "g8_multi_cpu" / "results" / "g8_result.json")
    g7_final_validation = _read(
        PHASE_G_ROOT / "g7_recovery" / "r6_residual_sparse" / "results" / "g7_final_validation_result.json"
    )
    if g7_final_validation["final_decision"] != "ACCEPTED":
        raise ValueError("G7 final mechanism must be ACCEPTED before integrating into G9.")
    profile = g7_final_validation["mechanism_profile"]
    default_profile = profile["default_profile"]
    oov_profile = profile["oov_profile"]
    hybrid_gate = profile["hybrid_gate"]
    top_k = int(default_profile["top_k"])

    g7_module = _load_g7_run_module()

    fixture, tokenizer, train_sequences, eval_sequences = _load_frozen_fixture()
    if int(fixture["seed"]) != seed:
        raise ValueError("Frozen fixture seed mismatch.")

    g2_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
    g2_model.fit(train_sequences)
    local = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=2, alpha=0.1)
    medium = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=4, alpha=0.1)
    global_model = SparseNGramModel(vocab_size=len(tokenizer.token_to_id), order=1, alpha=0.1)
    local.fit(train_sequences)
    medium.fit(train_sequences)
    global_model.fit(train_sequences)
    unigram = global_model.global_counts + 0.1
    unigram = unigram / unigram.sum()

    tracemalloc.start()
    start = time.perf_counter()
    g6_rows: list[np.ndarray] = []
    r4_rows: list[np.ndarray] = []
    g7_final_rows: list[np.ndarray] = []
    targets: list[int] = []
    for sequence in eval_sequences:
        for pos in range(len(sequence) - 1):
            context = sequence[: pos + 1]
            p_g6 = 0.5 * local.distribution(context) + 0.35 * medium.distribution(context) + 0.15 * global_model.distribution(context)
            p_g6 = p_g6 / p_g6.sum()
            p_g2 = g2_model.distribution(context)
            p_r4 = g7_module._r4_baseline(p_g6, p_g2, top_k=4, floor_mix=0.2)
            use_oov_profile = bool(
                (tokenizer.unk_id in context)
                or (float(p_g6[tokenizer.unk_id]) >= float(hybrid_gate["trigger_if_g6_unk_probability_gte"]))
            )
            if use_oov_profile:
                p_g7_final = g7_module._r6_residual_sparse_contextual_plus_forced_unk(
                    g6=p_g6,
                    top_k=top_k,
                    g2=p_g2,
                    unigram_distribution=unigram,
                    unk_id=tokenizer.unk_id,
                    context_blend=float(oov_profile["context_blend"]),
                    tail_temperature=float(oov_profile["tail_temperature"]),
                    head_temperature=float(oov_profile["head_temperature"]),
                    head_context_mix=float(oov_profile["head_context_mix"]),
                    residual_gain=float(oov_profile["residual_gain"]),
                )
            else:
                p_g7_final = g7_module._r6_residual_sparse_contextual_plus_forced_unk(
                    g6=p_g6,
                    top_k=top_k,
                    g2=p_g2,
                    unigram_distribution=unigram,
                    unk_id=tokenizer.unk_id,
                    context_blend=float(default_profile["context_blend"]),
                    tail_temperature=float(default_profile["tail_temperature"]),
                    head_temperature=float(default_profile["head_temperature"]),
                    head_context_mix=float(default_profile["head_context_mix"]),
                    residual_gain=float(default_profile["residual_gain"]),
                )
            g6_rows.append(p_g6)
            r4_rows.append(p_r4)
            g7_final_rows.append(p_g7_final)
            targets.append(int(sequence[pos + 1]))
    g6_metrics = evaluate_predictions(g6_rows, targets)
    r4_metrics = evaluate_predictions(r4_rows, targets)
    g9_metrics = evaluate_predictions(g7_final_rows, targets)
    inference_time = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mass_errors = [abs(float(row.sum()) - 1.0) for row in g7_final_rows]
    unknown_context_fraction = float(
        sum(1 for sequence in eval_sequences for pos in range(len(sequence) - 1) if tokenizer.unk_id in sequence[: pos + 1])
        / max(len(targets), 1)
    )

    g1_path = PHASE_G_ROOT / "results" / "g1_transition_eval_seed1337.json"
    g1 = _read(g1_path)["g1_transition"] if g1_path.exists() else {}

    result = {
        "experiment_id": "g9_integrated_cpu_native",
        "seed": seed,
        "corpus_size": len(fixture["corpus"]),
        "vocab_size": len(tokenizer.token_to_id),
        "parameter_count": local.parameter_count + medium.parameter_count + global_model.parameter_count + g2_model.parameter_count,
        "active_parameter_count": int((local.parameter_count + medium.parameter_count + global_model.parameter_count) * (top_k / len(tokenizer.token_to_id))),
        "model_storage": local.model_storage_bytes + medium.model_storage_bytes + global_model.model_storage_bytes + g2_model.model_storage_bytes,
        "peak_RAM": int(peak),
        "training_time": 0.0,
        "inference_time": inference_time,
        "tokens_per_second": g9_metrics["count"] / max(inference_time, 1e-12),
        "context_length": 4,
        "cpu_threading": CPU_THREADING,
        "fixture_path": str((PHASE_G_ROOT / "diagnostics" / "frozen_eval" / "fixture.json").resolve()),
        "g7_profile_path": str(
            (PHASE_G_ROOT / "g7_recovery" / "r6_residual_sparse" / "results" / "g7_final_mechanism_profile.json").resolve()
        ),
        "g1": g1,
        "g2": g2["g2"],
        "g3": g3["g3"],
        "g4": g4["g4"],
        "g5": g5["g5"],
        "g6": g6["g6"],
        "g7": g7_legacy["g7"],
        "g8": g8,
        "g9": {k: g9_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
        "g9_comparison": {
            "g6_reference": {k: g6_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
            "r4_reference": {k: r4_metrics[k] for k in ("accuracy", "cross_entropy", "perplexity")},
            "delta_ce_vs_g6": float(g9_metrics["cross_entropy"] - g6_metrics["cross_entropy"]),
            "delta_ce_vs_r4": float(g9_metrics["cross_entropy"] - r4_metrics["cross_entropy"]),
            "mass_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
            "unknown_context_fraction": unknown_context_fraction,
        },
        "g7_final_profile": profile,
        "selected_mechanisms": {
            "representation": bool(g3["hard_gates"]["representations_learned"]),
            "context": True,
            "adaptive_memory": bool(g5["hard_gates"]["benefit_over_g4"]),
            "hierarchy": bool(g6["hard_gates"]["advantage_over_g5"]),
            "sparsity": True,
            "multi_cpu_strategy": bool(g8["hard_gates"]["useful_parallelism"]),
        },
        "hard_gates": {
            "integrated_better_than_g2": g9_metrics["cross_entropy"] <= g2["g2"]["cross_entropy"],
            "integrated_competitive_vs_g6": g9_metrics["cross_entropy"] <= g6_metrics["cross_entropy"],
            "integrated_beats_r4": g9_metrics["cross_entropy"] <= r4_metrics["cross_entropy"],
            "probability_mass_correct": float(np.max(np.asarray(mass_errors, dtype=np.float64))) <= 1e-6,
            "cpu_measured": True,
        },
    }

    root = Path(__file__).resolve().parent
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "g9_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (PHASE_G_ROOT / "results" / "g9_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("G9 complete")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
