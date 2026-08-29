from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np


Context = tuple[int, ...]


@dataclass
class SparseNGramModel:
    vocab_size: int
    order: int
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.order <= 0:
            raise ValueError("order must be > 0.")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0.")
        self.counts: dict[Context, np.ndarray] = {}
        self.global_counts = np.zeros(self.vocab_size, dtype=np.float64)

    def _context(self, sequence: list[int], position: int) -> Context:
        start = position - self.order + 1
        if start < 0:
            prefix = [sequence[0]] * (-start)
            body = sequence[0 : position + 1]
            return tuple(prefix + body)
        return tuple(sequence[start : position + 1])

    def fit(self, sequences: Iterable[list[int]]) -> int:
        transitions = 0
        for sequence in sequences:
            if len(sequence) < 2:
                continue
            for pos in range(len(sequence) - 1):
                context = self._context(sequence, pos)
                next_token = int(sequence[pos + 1])
                if context not in self.counts:
                    self.counts[context] = np.zeros(self.vocab_size, dtype=np.float64)
                self.counts[context][next_token] += 1.0
                self.global_counts[next_token] += 1.0
                transitions += 1
        return transitions

    def distribution(self, context_tokens: list[int]) -> np.ndarray:
        if not context_tokens:
            raise ValueError("context_tokens must not be empty.")
        if len(context_tokens) >= self.order:
            key = tuple(context_tokens[-self.order :])
        else:
            key = tuple([context_tokens[0]] * (self.order - len(context_tokens)) + context_tokens)
        counts = self.counts.get(key)
        if counts is None:
            counts = self.global_counts
        smoothed = counts + self.alpha
        return smoothed / smoothed.sum()

    def predict(self, context_tokens: list[int]) -> int:
        return int(np.argmax(self.distribution(context_tokens)))

    def iter_probability_rows(self, sequences: Iterable[list[int]]) -> tuple[list[np.ndarray], list[int]]:
        rows: list[np.ndarray] = []
        targets: list[int] = []
        for sequence in sequences:
            if len(sequence) < 2:
                continue
            for pos in range(len(sequence) - 1):
                context = sequence[max(0, pos - self.order + 1) : pos + 1]
                rows.append(self.distribution(context))
                targets.append(int(sequence[pos + 1]))
        return rows, targets

    @property
    def nonzero_parameters(self) -> int:
        return int(sum(int(np.count_nonzero(v)) for v in self.counts.values()))

    @property
    def parameter_count(self) -> int:
        return int(len(self.counts) * self.vocab_size)

    @property
    def model_storage_bytes(self) -> int:
        total = 0
        for key, value in self.counts.items():
            total += len(key) * 8
            total += value.nbytes
        total += self.global_counts.nbytes
        return int(total)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as file:
            pickle.dump(
                {
                    "vocab_size": self.vocab_size,
                    "order": self.order,
                    "alpha": self.alpha,
                    "counts": self.counts,
                    "global_counts": self.global_counts,
                },
                file,
            )

    @classmethod
    def load(cls, path: str | Path) -> "SparseNGramModel":
        with open(Path(path), "rb") as file:
            payload = pickle.load(file)
        model = cls(
            vocab_size=int(payload["vocab_size"]),
            order=int(payload["order"]),
            alpha=float(payload["alpha"]),
        )
        model.counts = payload["counts"]
        model.global_counts = payload["global_counts"]
        return model
