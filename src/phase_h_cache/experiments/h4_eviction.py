from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.memory.topology_guard import TopologyGuard


class CappedVocab:
    def __init__(self, cap: int) -> None:
        if cap < 4:
            raise ValueError("cap must be >= 4")
        self.cap = cap
        self.token_to_id = {"<UNK>": 0}

    @property
    def unk_id(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, token: str) -> int:
        existing = self.token_to_id.get(token)
        if existing is not None:
            return existing
        if len(self.token_to_id) < self.cap:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            return token_id
        return self.unk_id


def _tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _build_stream(total_tokens: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    stream: list[str] = []
    financial = "the central bank approved loan credit account rate".split()
    river = "the river bank had water current flood stone".split()
    for idx in range(total_tokens // 14):
        stream.extend(financial)
        stream.extend(river)
        stream.append(f"novel_token_{idx}_{rng.randint(0, 100000)}")
    return stream[:total_tokens]


def _run_strategy(strategy: str, *, config: dict[str, Any], seed: int) -> dict[str, Any]:
    vocab = CappedVocab(cap=int(config["vocab_cap"]))
    guard = TopologyGuard(max_nodes=int(config["max_nodes"]), strategy=strategy)
    stream = _build_stream(int(config["total_tokens"]), seed)
    checkpoint = int(config["checkpoint_tokens"])

    context_fin = ("central", "bank")
    context_riv = ("river", "bank")
    context_fin_ids = (vocab.encode(context_fin[0]), vocab.encode(context_fin[1]))
    context_riv_ids = (vocab.encode(context_riv[0]), vocab.encode(context_riv[1]))
    protected_contexts = {context_fin_ids, context_riv_ids}

    evicted_total = 0
    halted = False
    unk_seen = 0
    rows: list[dict[str, float]] = []
    previous_tokens = [vocab.encode("<UNK>"), vocab.encode("<UNK>")]

    for index, token in enumerate(stream, start=1):
        token_id = vocab.encode(token)
        if token_id == vocab.unk_id:
            unk_seen += 1
        context = (previous_tokens[-2], previous_tokens[-1])
        result = guard.observe(
            context,
            token_id,
            timestamp=index,
            protected=(context in protected_contexts),
        )
        evicted_total += result.evicted
        if result.halted:
            halted = True
            break
        previous_tokens.append(token_id)

        if index % checkpoint == 0:
            fin_probs = np.asarray(
                guard.distribution(context_fin_ids, vocab_size=vocab.size, alpha=0.1),
                dtype=np.float64,
            )
            riv_probs = np.asarray(
                guard.distribution(context_riv_ids, vocab_size=vocab.size, alpha=0.1),
                dtype=np.float64,
            )
            delta = _tv_distance(fin_probs, riv_probs)
            unk_mass = float(max(fin_probs[vocab.unk_id], riv_probs[vocab.unk_id]))
            rows.append(
                {
                    "ingested_tokens": float(index),
                    "context_intervention_delta": delta,
                    "unk_mass": unk_mass,
                    "node_count": float(len(guard.nodes)),
                }
            )

    final_delta = float(rows[-1]["context_intervention_delta"]) if rows else 0.0
    peak_unk_mass = float(max((row["unk_mass"] for row in rows), default=0.0))
    context_gate = final_delta >= float(config["context_delta_gate"])
    unk_gate = peak_unk_mass <= float(config["unk_gate"])
    gate_status = "PASS" if (context_gate and unk_gate and not halted) else "FAIL"
    return {
        "strategy": strategy,
        "halted": halted,
        "evicted_total": evicted_total,
        "final_context_intervention_delta": final_delta,
        "peak_unk_absorbed_mass_percentage": peak_unk_mass * 100.0,
        "hard_gates": {
            "context_preservation": context_gate,
            "unk_pollution": unk_gate,
        },
        "gate_status": gate_status,
        "rows": rows,
    }


def run() -> dict[str, Any]:
    phase_cfg = PhaseHConfig.load_default().raw
    config = phase_cfg["h4"]
    seed = int(phase_cfg["h5"]["seed"])
    strategies = ["none", "frequency", "lru", "utility"]
    results = [_run_strategy(strategy, config=config, seed=seed) for strategy in strategies]

    passing = [row for row in results if row["gate_status"] == "PASS"]
    if passing:
        winning = max(passing, key=lambda row: row["final_context_intervention_delta"])
    else:
        winning = max(results, key=lambda row: row["final_context_intervention_delta"])

    return {
        "experiment_id": "h4_topological_eviction_stability",
        "config": config,
        "strategies": results,
        "winning_eviction_strategy": winning["strategy"],
        "final_context_intervention_delta": winning["final_context_intervention_delta"],
        "peak_unk_absorbed_mass_percentage": winning["peak_unk_absorbed_mass_percentage"],
        "gate_status": winning["gate_status"],
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h4_eviction.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
