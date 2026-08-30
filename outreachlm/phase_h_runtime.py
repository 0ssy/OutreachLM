from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
import random
from typing import Any

import numpy as np

from outreachlm.phase_g_bridge import (
    PhaseGHybridConfig,
    PhaseGHybridRuntime,
    SparseNGramModel,
    WordTokenizer,
)
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT


@dataclass(frozen=True)
class PhaseHRuntimeConfig:
    quantization_mode: str = "fp16"
    repetition_decay: float = 0.85
    repetition_floor: float = 0.25
    unk_alert_threshold: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantization_mode": self.quantization_mode,
            "repetition_decay": self.repetition_decay,
            "repetition_floor": self.repetition_floor,
            "unk_alert_threshold": self.unk_alert_threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseHRuntimeConfig":
        return cls(
            quantization_mode=str(payload.get("quantization_mode", "fp16")),
            repetition_decay=float(payload.get("repetition_decay", 0.85)),
            repetition_floor=float(payload.get("repetition_floor", 0.25)),
            unk_alert_threshold=float(payload.get("unk_alert_threshold", 0.10)),
        )


def _apply_fp16_quantization(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be a rank-1 vector.")
    out = np.asarray(probabilities, dtype=np.float16).astype(np.float64)
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum()
    return out


def _apply_repetition_penalty(
    probabilities: np.ndarray,
    recent_tokens: list[int],
    *,
    decay: float,
    floor: float,
) -> np.ndarray:
    out = np.asarray(probabilities, dtype=np.float64).copy()
    if not recent_tokens:
        return out / out.sum()
    repeats: dict[int, int] = {}
    for token_id in recent_tokens:
        repeats[token_id] = repeats.get(token_id, 0) + 1
    for token_id, count in repeats.items():
        if 0 <= token_id < len(out):
            multiplier = max(floor, decay**count)
            out[token_id] *= multiplier
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum()
    return out


def _top_k_filter(probabilities: np.ndarray, top_k: int | None) -> np.ndarray:
    if top_k is None or top_k <= 0 or top_k >= len(probabilities):
        return probabilities
    indices = np.argpartition(probabilities, -top_k)[-top_k:]
    out = np.zeros_like(probabilities, dtype=np.float64)
    out[indices] = probabilities[indices]
    out = np.clip(out, 1e-12, 1.0)
    out = out / out.sum()
    return out


def _max_repetition_run(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(token_ids)):
        if token_ids[idx] == token_ids[idx - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


class BoundedStateRuntime:
    def __init__(
        self,
        phase_g_runtime: PhaseGHybridRuntime,
        config: PhaseHRuntimeConfig | None = None,
    ) -> None:
        self.phase_g_runtime = phase_g_runtime
        self.config = config or PhaseHRuntimeConfig()

    @property
    def tokenizer(self) -> WordTokenizer:
        return self.phase_g_runtime.tokenizer

    @classmethod
    def from_corpus_path(
        cls,
        *,
        corpus_path: str | Path = CORPUS_PATH,
        validation_split: float = VALIDATION_SPLIT,
        max_train_lines: int | None = 2000,
        max_eval_lines: int | None = 300,
        phase_g_config: PhaseGHybridConfig | None = None,
        config: PhaseHRuntimeConfig | None = None,
    ) -> tuple["BoundedStateRuntime", list[str], list[str]]:
        runtime, train_lines, eval_lines = PhaseGHybridRuntime.from_corpus_path(
            corpus_path,
            validation_split=validation_split,
            config=phase_g_config,
            max_train_lines=max_train_lines,
            max_eval_lines=max_eval_lines,
        )
        return cls(runtime, config=config), train_lines, eval_lines

    def _distribution(
        self,
        context_tokens: list[int],
        *,
        recent_tokens: list[int] | None = None,
        apply_safety: bool = True,
    ) -> np.ndarray:
        probabilities = self.phase_g_runtime.distribution(context_tokens)
        if self.config.quantization_mode == "fp16":
            probabilities = _apply_fp16_quantization(probabilities)
        elif self.config.quantization_mode != "fp32":
            raise ValueError(f"Unsupported quantization_mode: {self.config.quantization_mode}")

        if apply_safety:
            probabilities = _apply_repetition_penalty(
                probabilities,
                recent_tokens or [],
                decay=self.config.repetition_decay,
                floor=self.config.repetition_floor,
            )
        return probabilities

    def evaluate_lines(self, lines: list[str], *, apply_safety: bool = False) -> dict[str, float]:
        rows: list[np.ndarray] = []
        targets: list[int] = []
        for line in lines:
            seq = self.tokenizer.encode(line, add_bos=True, add_eos=True)
            for pos in range(len(seq) - 1):
                context = seq[: pos + 1]
                rows.append(self._distribution(context, recent_tokens=context[-64:], apply_safety=apply_safety))
                targets.append(int(seq[pos + 1]))
        if not rows:
            raise ValueError("No samples available for evaluation.")

        correct = 0
        nll = 0.0
        mass_errors: list[float] = []
        unk_probabilities: list[float] = []
        for probabilities, target in zip(rows, targets):
            prediction = int(np.argmax(probabilities))
            if prediction == target:
                correct += 1
            p = float(probabilities[target])
            nll += -math.log(max(p, 1e-12))
            mass_errors.append(abs(float(probabilities.sum()) - 1.0))
            unk_probabilities.append(float(probabilities[self.tokenizer.unk_id]))

        count = len(rows)
        cross_entropy = nll / count
        return {
            "accuracy": correct / count,
            "cross_entropy": cross_entropy,
            "perplexity": math.exp(cross_entropy),
            "count": float(count),
            "mass_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
            "unk_probability_mean": float(np.mean(np.asarray(unk_probabilities, dtype=np.float64))),
            "unk_alert": float(np.mean(np.asarray(unk_probabilities, dtype=np.float64))) > self.config.unk_alert_threshold,
        }

    def ingest_lines(self, lines: list[str]) -> dict[str, float]:
        sequences = [self.tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines if line.strip()]
        transitions = self.phase_g_runtime.g2.fit(sequences)
        self.phase_g_runtime.local.fit(sequences)
        self.phase_g_runtime.medium.fit(sequences)
        self.phase_g_runtime.global_model.fit(sequences)
        return {
            "lines_ingested": float(len(sequences)),
            "transitions_ingested": float(transitions),
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 8,
        seed: int = 1337,
        apply_safety: bool = True,
    ) -> dict[str, Any]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0.")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        rng = random.Random(seed)
        generated = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        sampled: list[int] = []
        unk_probabilities: list[float] = []

        for _ in range(max_new_tokens):
            probabilities = self._distribution(generated, recent_tokens=sampled[-64:], apply_safety=apply_safety)
            logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
            scaled = np.exp(logits - np.max(logits))
            scaled = scaled / scaled.sum()
            scaled = _top_k_filter(scaled, top_k)

            token_ids = list(range(len(scaled)))
            next_token = int(rng.choices(token_ids, weights=scaled.tolist(), k=1)[0])
            sampled.append(next_token)
            generated.append(next_token)
            unk_probabilities.append(float(scaled[self.tokenizer.unk_id]))

            if next_token == self.tokenizer.eos_id:
                break

        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return {
            "prompt": prompt,
            "generated_text": output_text,
            "generated_token_ids": sampled,
            "max_repetition_run": _max_repetition_run(sampled),
            "unk_probability_mean": float(np.mean(np.asarray(unk_probabilities, dtype=np.float64)))
            if unk_probabilities
            else 0.0,
            "unk_alert": (
                float(np.mean(np.asarray(unk_probabilities, dtype=np.float64)))
                if unk_probabilities
                else 0.0
            )
            > self.config.unk_alert_threshold,
        }

    def _to_payload(self) -> dict[str, Any]:
        return {
            "format": "phase_h_bounded_runtime_v1",
            "phase_h_config": self.config.to_dict(),
            "phase_g": {
                "config": self.phase_g_runtime.config.to_dict(),
                "token_to_id": self.tokenizer.token_to_id,
                "models": {
                    "g2": self.phase_g_runtime.g2.to_payload(),
                    "local": self.phase_g_runtime.local.to_payload(),
                    "medium": self.phase_g_runtime.medium.to_payload(),
                    "global": self.phase_g_runtime.global_model.to_payload(),
                },
            },
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "BoundedStateRuntime":
        if payload.get("format") != "phase_h_bounded_runtime_v1":
            raise ValueError("Unsupported Phase H runtime artifact format.")
        phase_g_payload = payload["phase_g"]
        tokenizer = WordTokenizer(token_to_id=phase_g_payload["token_to_id"])
        phase_g_config = PhaseGHybridConfig.from_dict(phase_g_payload["config"])
        phase_g_runtime = PhaseGHybridRuntime(tokenizer=tokenizer, config=phase_g_config)
        phase_g_runtime.g2 = SparseNGramModel.from_payload(phase_g_payload["models"]["g2"])
        phase_g_runtime.local = SparseNGramModel.from_payload(phase_g_payload["models"]["local"])
        phase_g_runtime.medium = SparseNGramModel.from_payload(phase_g_payload["models"]["medium"])
        phase_g_runtime.global_model = SparseNGramModel.from_payload(phase_g_payload["models"]["global"])
        phase_g_runtime._fitted = True
        phase_h_config = PhaseHRuntimeConfig.from_dict(payload["phase_h_config"])
        return cls(phase_g_runtime=phase_g_runtime, config=phase_h_config)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as file:
            pickle.dump(self._to_payload(), file)

    @classmethod
    def load(cls, path: str | Path) -> "BoundedStateRuntime":
        with open(Path(path), "rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Phase H runtime artifact payload must be a dictionary.")
        return cls._from_payload(payload)
