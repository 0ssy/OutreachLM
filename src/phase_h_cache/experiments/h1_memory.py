from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from outreachlm.train import CORPUS_PATH, load_corpus

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.memory.accounting import process_memory_bytes
from src.phase_h_cache.memory.topology_guard import TopologyGuard


def _stream_tokens(sample_vocab: int) -> list[str]:
    text = load_corpus(CORPUS_PATH)
    words = [token for token in text.split() if token]
    if not words:
        raise ValueError("Corpus stream is empty.")
    return words[:sample_vocab]


def _slope(points: list[tuple[int, int]]) -> float:
    if len(points) < 2:
        return 0.0
    (x0, y0), (x1, y1) = points[0], points[-1]
    return (y1 - y0) / max(1, x1 - x0)


def run() -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw["h1"]
    min_tokens = int(config["min_tokens"])
    block_tokens = int(config["block_tokens"])
    warmup_tokens = int(config["warmup_tokens"])
    sample_vocab = int(config["sample_vocab"])

    stream_base = _stream_tokens(sample_vocab)
    guard = TopologyGuard(max_nodes=50_000_000, strategy="none")
    metrics: list[dict[str, int]] = []

    ingested = 0
    previous = "<BOS>"
    while ingested < min_tokens:
        for token in stream_base:
            guard.observe((previous,), 0, timestamp=ingested, protected=False)
            previous = token
            ingested += 1
            if ingested % block_tokens == 0:
                logical = guard.logical_size_bytes()
                mem = process_memory_bytes()
                metrics.append(
                    {
                        "ingested_tokens": ingested,
                        "logical_model_size_bytes": int(logical),
                        "process_rss_bytes": int(mem["rss_bytes"]),
                        "process_vms_bytes": int(mem["vms_bytes"]),
                    }
                )
            if ingested >= min_tokens:
                break

    warmup_rows = [row for row in metrics if row["ingested_tokens"] <= warmup_tokens]
    post_rows = [row for row in metrics if row["ingested_tokens"] > warmup_tokens]
    warmup_slope = _slope([(row["ingested_tokens"], row["process_rss_bytes"]) for row in warmup_rows])
    post_slope = _slope([(row["ingested_tokens"], row["process_rss_bytes"]) for row in post_rows])
    scaling_profile = "sublinear" if post_slope <= warmup_slope else "linear"

    ratios = [
        row["process_rss_bytes"] / max(1, row["logical_model_size_bytes"])
        for row in metrics
        if row["ingested_tokens"] > warmup_tokens
    ]
    ratio_stability = {
        "min": float(min(ratios) if ratios else 0.0),
        "max": float(max(ratios) if ratios else 0.0),
    }
    output = {
        "experiment_id": "h1_memory_topology",
        "config": config,
        "final_ingested_tokens": ingested,
        "peak_logical_model_size_bytes": max(row["logical_model_size_bytes"] for row in metrics),
        "peak_process_rss_bytes": max(row["process_rss_bytes"] for row in metrics),
        "peak_process_vms_bytes": max(row["process_vms_bytes"] for row in metrics),
        "memory_scaling_profile": scaling_profile,
        "warmup_rss_slope_bytes_per_token": warmup_slope,
        "post_warmup_rss_slope_bytes_per_token": post_slope,
        "rss_to_logical_ratio_post_warmup": ratio_stability,
        "rows": metrics,
    }
    return output


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h1_memory.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
