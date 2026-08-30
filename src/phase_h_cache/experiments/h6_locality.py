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
from src.phase_h_cache.memory.platform_perf import benchmark_steps, build_padding_state, size_and_memory


def _ingestion_process(stop_flag: mp.Event, pin_core: int | None) -> None:
    if pin_core is not None:
        proc = psutil.Process()
        if hasattr(proc, "cpu_affinity"):
            proc.cpu_affinity([pin_core])
    rng = np.random.default_rng(1337)
    left = rng.random((96, 96), dtype=np.float64)
    right = rng.random((96, 96), dtype=np.float64)
    while not stop_flag.is_set():
        _ = np.matmul(left, right)
        left, right = right, left


def _build_runtime(train_lines: int, eval_lines: int, seed: int) -> tuple[PhaseGHybridRuntime, list[list[int]]]:
    runtime, _, eval_text_lines = PhaseGHybridRuntime.from_corpus_path(
        CORPUS_PATH,
        validation_split=VALIDATION_SPLIT,
        max_train_lines=train_lines,
        max_eval_lines=eval_lines,
    )
    rng = random.Random(seed)
    contexts: list[list[int]] = []
    for line in eval_text_lines:
        seq = runtime.tokenizer.encode(line, add_bos=True, add_eos=True)
        for pos in range(len(seq) - 1):
            contexts.append(seq[: pos + 1])
    rng.shuffle(contexts)
    if not contexts:
        raise ValueError("No contexts available for locality benchmark.")
    return runtime, contexts


def _find_spillover(rows: list[dict[str, float]]) -> float:
    if len(rows) < 3:
        return float(rows[0]["target_state_size_mb"]) if rows else 0.0
    latencies = np.asarray([row["latency_us"] for row in rows], dtype=np.float64)
    states = [float(row["target_state_size_mb"]) for row in rows]
    baseline = float(np.median(latencies[:2]))
    for idx, value in enumerate(latencies):
        if value >= baseline * 1.35:
            return states[idx]
    return states[-1]


def run() -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw["h6"]
    runtime, contexts = _build_runtime(
        train_lines=int(config["train_lines"]),
        eval_lines=int(config["eval_lines"]),
        seed=int(config["seed"]),
    )
    steps = int(config["steps"])
    sizes_mb = [float(value) for value in config["state_sizes_mb"]]

    index = {"value": 0}

    def infer_once() -> np.ndarray:
        context = contexts[index["value"] % len(contexts)]
        index["value"] += 1
        return runtime.distribution(context)

    rows: list[dict[str, Any]] = []
    with pin_process_to_first_cores(2):
        for size_mb in sizes_mb:
            padding = build_padding_state(int(size_mb * 1024 * 1024))
            stop_signal = mp.Event()
            ingest = mp.Process(target=_ingestion_process, args=(stop_signal, 1), daemon=True)
            ingest.start()
            with pin_current_thread(0):
                sample = benchmark_steps(infer_once, steps=steps)
            stop_signal.set()
            ingest.join(timeout=2.0)
            if ingest.is_alive():
                ingest.terminate()
            mem = size_and_memory({"padding": padding})
            rows.append(
                {
                    "target_state_size_mb": size_mb,
                    "logical_state_size_bytes": mem["logical_bytes"],
                    "process_rss_bytes": mem["rss_bytes"],
                    "process_vms_bytes": mem["vms_bytes"],
                    "elapsed_seconds": sample.elapsed_seconds,
                    "latency_us": sample.latency_us,
                    "cycles_per_token_estimate": sample.cycles_per_token_estimate,
                    "cache_counters_available": False,
                    "llc_miss_rate": None,
                }
            )

    spillover_mb = _find_spillover(rows)
    pre = [row["latency_us"] for row in rows if row["target_state_size_mb"] <= spillover_mb]
    post = [row["latency_us"] for row in rows if row["target_state_size_mb"] > spillover_mb]
    pre_mean = float(np.mean(np.asarray(pre, dtype=np.float64))) if pre else 0.0
    post_mean = float(np.mean(np.asarray(post, dtype=np.float64))) if post else pre_mean
    latency_increase = 0.0 if pre_mean <= 0.0 else max(0.0, (post_mean - pre_mean) / pre_mean)
    optimal_row = min(rows, key=lambda row: row["latency_us"])

    return {
        "experiment_id": "h6_locality_mapping",
        "config": config,
        "rows": rows,
        "measured_cache_spillover_threshold_mb": spillover_mb,
        "latency_increase_rate_post_spillover": latency_increase,
        "cycles_per_token_at_optimal_size": float(optimal_row["cycles_per_token_estimate"]),
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "deep_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h6_locality.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
