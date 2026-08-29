from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class TransitionModel:
    vocab_size: int
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0.")
        self.counts = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)

    def fit(self, token_sequences: Iterable[list[int]]) -> int:
        transition_count = 0
        for sequence in token_sequences:
            if len(sequence) < 2:
                continue
            for current_token, next_token in zip(sequence[:-1], sequence[1:]):
                self.counts[current_token, next_token] += 1.0
                transition_count += 1
        return transition_count

    def transition_probabilities(self) -> np.ndarray:
        smoothed = self.counts + self.alpha
        row_sums = smoothed.sum(axis=1, keepdims=True)
        return smoothed / row_sums

    def predict_next_distribution(self, current_token: int) -> np.ndarray:
        probabilities = self.transition_probabilities()
        return probabilities[current_token]

    def predict_next_token_argmax(self, current_token: int) -> int:
        distribution = self.predict_next_distribution(current_token)
        return int(np.argmax(distribution))

    def evaluate(self, token_sequences: Iterable[list[int]]) -> dict[str, float]:
        probabilities = self.transition_probabilities()
        total = 0
        correct = 0
        nll_sum = 0.0
        for sequence in token_sequences:
            if len(sequence) < 2:
                continue
            for current_token, next_token in zip(sequence[:-1], sequence[1:]):
                prediction = int(np.argmax(probabilities[current_token]))
                if prediction == next_token:
                    correct += 1
                nll_sum += -float(np.log(probabilities[current_token, next_token]))
                total += 1
        if total == 0:
            raise ValueError("No transitions available for evaluation.")
        cross_entropy = nll_sum / total
        perplexity = float(np.exp(cross_entropy))
        accuracy = correct / total
        return {
            "transition_count": float(total),
            "accuracy": float(accuracy),
            "cross_entropy": float(cross_entropy),
            "perplexity": float(perplexity),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            counts=self.counts,
            alpha=np.array([self.alpha], dtype=np.float64),
            vocab_size=np.array([self.vocab_size], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TransitionModel":
        payload = np.load(Path(path), allow_pickle=False)
        vocab_size = int(payload["vocab_size"][0])
        alpha = float(payload["alpha"][0])
        model = cls(vocab_size=vocab_size, alpha=alpha)
        model.counts = payload["counts"].astype(np.float64)
        return model


def evaluate_random_baseline(
    token_sequences: Iterable[list[int]],
    *,
    vocab_size: int,
    seed: int,
) -> dict[str, float]:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0.")
    rng = np.random.default_rng(seed)
    total = 0
    correct = 0
    for sequence in token_sequences:
        if len(sequence) < 2:
            continue
        for _, next_token in zip(sequence[:-1], sequence[1:]):
            prediction = int(rng.integers(0, vocab_size))
            if prediction == next_token:
                correct += 1
            total += 1
    if total == 0:
        raise ValueError("No transitions available for baseline evaluation.")
    cross_entropy = float(np.log(vocab_size))
    perplexity = float(np.exp(cross_entropy))
    return {
        "transition_count": float(total),
        "accuracy": float(correct / total),
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
    }
