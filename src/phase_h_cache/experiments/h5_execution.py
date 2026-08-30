from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any

import numpy as np
import psutil

from outreachlm.phase_g_bridge import PhaseGHybridRuntime
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.execution.affinity import pin_current_thread, pin_process_to_first_cores
from src.phase_h_cache.execution.parallel import benchmark_inference, set_thread_env


def _build_runtime(train_lines: int, seed: int) -> tuple[PhaseGHybridRuntime, list[list[int]]]:
    runtime, _, eval_lines = PhaseGHybridRuntime.from_corpus_path(
        CORPUS_PATH,
        validation_split=VALIDATION_SPLIT,
        max_train_lines=train_lines,
        max_eval_lines=150,
    )
    rng = random.Random(seed)
    contexts: list[list[int]] = []
    for line in eval_lines:
        seq = runtime.tokenizer.encode(line, add_bos=True, add_eos=True)
        for pos in range(len(seq) - 1):
            contexts.append(seq[: pos + 1])
    rng.shuffle(contexts)
    if not contexts:
        raise ValueError("No contexts available for execution benchmark.")
    return runtime, contexts


def _ingestion_process(stop_flag: mp.Event, pin_core: int | None) -> None:
    if pin_core is not None:
        proc = psutil.Process()
        if hasattr(proc, "cpu_affinity"):
            proc.cpu_affinity([pin_core])
    rng = np.random.default_rng(1337)
    left = rng.random((128, 128), dtype=np.float64)
    right = rng.random((128, 128), dtype=np.float64)
    while not stop_flag.is_set():
        _ = np.matmul(left, right)
        left, right = right, left


def _measure_mode(
    runtime: PhaseGHybridRuntime,
    contexts: list[list[int]],
    *,
    steps: int,
    pinned: bool,
    cores: int,
) -> dict[str, float]:
    index = {"value": 0}
    stop_signal = mp.Event()

    def infer_once() -> np.ndarray:
        current = contexts[index["value"] % len(contexts)]
        index["value"] += 1
        return runtime.distribution(current)

    ingestion_proc: mp.Process | None = None
    if cores >= 2:
        ingestion_pin_core = 1 if pinned and cores >= 2 else None
        ingestion_proc = mp.Process(target=_ingestion_process, args=(stop_signal, ingestion_pin_core), daemon=True)
        ingestion_proc.start()

    if pinned:
        with pin_process_to_first_cores(cores) as did_pin:
            with pin_current_thread(0) as did_thread_pin:
                data = benchmark_inference(infer_once, steps=steps)
                data["did_pin"] = 1.0 if did_pin else 0.0
                data["did_thread_pin"] = 1.0 if did_thread_pin else 0.0
                stop_signal.set()
                if ingestion_proc is not None:
                    ingestion_proc.join(timeout=2.0)
                    if ingestion_proc.is_alive():
                        ingestion_proc.terminate()
                return data

    data = benchmark_inference(infer_once, steps=steps)
    data["did_pin"] = 0.0
    data["did_thread_pin"] = 0.0
    stop_signal.set()
    if ingestion_proc is not None:
        ingestion_proc.join(timeout=2.0)
        if ingestion_proc.is_alive():
            ingestion_proc.terminate()
    return data


def run() -> dict[str, Any]:
    phase_cfg = PhaseHConfig.load_default().raw
    config = phase_cfg["h5"]
    steps = int(config["steps"])
    cores_list = [int(value) for value in config["cores"]]
    seed = int(config["seed"])
    trials = 3

    runtime, contexts = _build_runtime(train_lines=int(config["train_lines"]), seed=seed)
    cpu_percent_before = psutil.cpu_percent(interval=0.25)

    rows: list[dict[str, Any]] = []
    for cores in cores_list:
        set_thread_env(cores)
        unpinned_trials = [_measure_mode(runtime, contexts, steps=steps, pinned=False, cores=cores) for _ in range(trials)]
        pinned_trials = [_measure_mode(runtime, contexts, steps=steps, pinned=True, cores=cores) for _ in range(trials)]

        def aggregate(trial_rows: list[dict[str, float]]) -> dict[str, float]:
            return {
                "tokens_per_second": float(np.median([row["tokens_per_second"] for row in trial_rows])),
                "latency_mean_us": float(np.median([row["latency_mean_us"] for row in trial_rows])),
                "latency_std_us": float(np.median([row["latency_std_us"] for row in trial_rows])),
                "did_pin": float(np.max([row["did_pin"] for row in trial_rows])),
                "did_thread_pin": float(np.max([row["did_thread_pin"] for row in trial_rows])),
            }

        rows.append(
            {
                "cores": cores,
                "unpinned": aggregate(unpinned_trials),
                "pinned": aggregate(pinned_trials),
                "unpinned_trials": unpinned_trials,
                "pinned_trials": pinned_trials,
            }
        )

    cpu_percent_after = psutil.cpu_percent(interval=0.25)

    best = max(rows, key=lambda row: row["pinned"]["tokens_per_second"])
    two_core = next((row for row in rows if row["cores"] == 2), rows[0])
    unpinned_latency_means = np.asarray(
        [row["latency_mean_us"] for row in two_core["unpinned_trials"]],
        dtype=np.float64,
    )
    pinned_latency_means = np.asarray(
        [row["latency_mean_us"] for row in two_core["pinned_trials"]],
        dtype=np.float64,
    )
    unpinned_std = float(np.std(unpinned_latency_means))
    pinned_std = float(np.std(pinned_latency_means))
    variance_reduction = 0.0
    if unpinned_std > 0.0:
        variance_reduction = (unpinned_std - pinned_std) / unpinned_std

    return {
        "experiment_id": "h5_execution_topology_affinity",
        "config": config,
        "trials_per_mode": trials,
        "cpu_percent_before": cpu_percent_before,
        "cpu_percent_after": cpu_percent_after,
        "rows": rows,
        "optimal_physical_cores": int(best["cores"]),
        "unpinned_tokens_per_second": float(two_core["unpinned"]["tokens_per_second"]),
        "true_os_pinned_tokens_per_second": float(two_core["pinned"]["tokens_per_second"]),
        "measured_latency_variance_reduction_rate": float(variance_reduction),
        "latency_variance_basis": "std_of_latency_mean_us_across_repeated_runs",
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h5_execution.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
