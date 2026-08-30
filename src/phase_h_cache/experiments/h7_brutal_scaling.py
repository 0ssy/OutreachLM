from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from src.phase_h_cache import PhaseHConfig
from src.phase_h_cache.memory.accounting import process_memory_bytes
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


def _token_stream(seed: int):
    rng = random.Random(seed)
    finance = "the bank approved loan credit account rate".split()
    river = "the river bank had water current flood stone".split()
    while True:
        for token in finance:
            yield token
        for token in river:
            yield token
        yield f"novel_{rng.randint(0, 99999999)}"


def _cross_entropy_sample(
    guard: TopologyGuard,
    vocab: CappedVocab,
    *,
    sample_eval_tokens: int,
    seed: int,
) -> float:
    stream = _token_stream(seed + 17)
    prev = [vocab.unk_id, vocab.unk_id]
    nll = 0.0
    count = 0
    for _ in range(sample_eval_tokens):
        token = next(stream)
        token_id = vocab.encode(token)
        probs = guard.distribution((prev[-2], prev[-1]), vocab_size=vocab.size, alpha=0.1)
        p = float(probs[token_id]) if token_id < len(probs) else 1e-12
        nll += -np.log(max(p, 1e-12))
        count += 1
        prev.append(token_id)
    return float(nll / max(1, count))


def _context_delta(guard: TopologyGuard, vocab: CappedVocab) -> float:
    left = (vocab.encode("central"), vocab.encode("bank"))
    right = (vocab.encode("river"), vocab.encode("bank"))
    p_left = np.asarray(guard.distribution(left, vocab_size=vocab.size, alpha=0.1), dtype=np.float64)
    p_right = np.asarray(guard.distribution(right, vocab_size=vocab.size, alpha=0.1), dtype=np.float64)
    return float(0.5 * np.abs(p_left - p_right).sum())


def run() -> dict[str, Any]:
    config = PhaseHConfig.load_default().raw["h7"]
    milestones = [int(value) for value in config["milestones"]]
    seed = int(config["seed"])
    sample_eval_tokens = int(config["sample_eval_tokens"])

    vocab = CappedVocab(cap=int(config["vocab_cap"]))
    guard = TopologyGuard(max_nodes=int(config["max_nodes"]), strategy="frequency")
    stream = _token_stream(seed)
    prev = [vocab.unk_id, vocab.unk_id]
    unk_hits = 0
    evicted_total = 0
    halted = False
    rows: list[dict[str, Any]] = []

    milestone_idx = 0
    target = milestones[milestone_idx]
    for step in range(1, milestones[-1] + 1):
        token = next(stream)
        token_id = vocab.encode(token)
        if token_id == vocab.unk_id:
            unk_hits += 1
        context = (prev[-2], prev[-1])
        result = guard.observe(context, token_id, timestamp=step)
        evicted_total += result.evicted
        if result.halted:
            halted = True
            break
        prev.append(token_id)

        if step == target:
            mem = process_memory_bytes()
            delta = _context_delta(guard, vocab)
            ce = _cross_entropy_sample(guard, vocab, sample_eval_tokens=sample_eval_tokens, seed=seed)
            rows.append(
                {
                    "milestone_tokens": step,
                    "process_rss_bytes": int(mem["rss_bytes"]),
                    "process_vms_bytes": int(mem["vms_bytes"]),
                    "logical_state_size_bytes": int(guard.logical_size_bytes()),
                    "active_graph_edges": int(sum(len(node["next_counts"]) for node in guard.nodes.values())),
                    "vocabulary_size": int(vocab.size),
                    "cross_entropy": ce,
                    "context_intervention_delta": delta,
                    "unk_mass_percentage": (unk_hits / step) * 100.0,
                    "evicted_total": evicted_total,
                }
            )
            milestone_idx += 1
            if milestone_idx >= len(milestones):
                break
            target = milestones[milestone_idx]

    terminal = rows[-1] if rows else {}
    status = "PASS"
    if halted:
        status = "FAIL"
    if terminal and terminal.get("unk_mass_percentage", 100.0) > float(config["unk_gate_percent"]):
        status = "FAIL"
    if terminal and terminal.get("context_intervention_delta", 0.0) < float(config["context_delta_gate"]):
        status = "FAIL"
    if terminal and int(terminal.get("vocabulary_size", 0)) > int(config["vocab_cap"]):
        status = "FAIL"
    if terminal and int(terminal.get("active_graph_edges", 0)) > int(config["max_nodes"]) * 8:
        status = "FAIL"

    return {
        "experiment_id": "h7_brutal_scaling",
        "config": config,
        "rows": rows,
        "max_brutal_ingested_tokens": int(terminal.get("milestone_tokens", 0)),
        "terminal_process_rss_bytes": int(terminal.get("process_rss_bytes", 0)),
        "terminal_unk_mass_percentage": float(terminal.get("unk_mass_percentage", 0.0)),
        "unbounded_ingestion_stability_status": status,
    }


def main() -> None:
    result = run()
    output_dir = Path("experiments") / "phase_h" / "deep_profile"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "h7_brutal_scaling.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
