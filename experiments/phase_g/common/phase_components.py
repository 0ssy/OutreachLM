from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompressedContextModel:
    vocab_size: int
    context_length: int = 4
    bucket_count: int = 64
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0.")
        if self.bucket_count <= 0:
            raise ValueError("bucket_count must be > 0.")
        self.counts: dict[int, np.ndarray] = {}
        self.global_counts = np.zeros(self.vocab_size, dtype=np.float64)

    def _key(self, context: list[int]) -> int:
        value = 0
        for token in context[-self.context_length :]:
            value = (value * 131 + int(token) + 1) % self.bucket_count
        return value

    def fit(self, sequences: list[list[int]]) -> int:
        transitions = 0
        for sequence in sequences:
            for pos in range(len(sequence) - 1):
                context = sequence[: pos + 1]
                key = self._key(context)
                nxt = int(sequence[pos + 1])
                if key not in self.counts:
                    self.counts[key] = np.zeros(self.vocab_size, dtype=np.float64)
                self.counts[key][nxt] += 1.0
                self.global_counts[nxt] += 1.0
                transitions += 1
        return transitions

    def distribution(self, context: list[int]) -> np.ndarray:
        counts = self.counts.get(self._key(context), self.global_counts)
        smoothed = counts + self.alpha
        return smoothed / smoothed.sum()

    def iter_probability_rows(self, sequences: list[list[int]]) -> tuple[list[np.ndarray], list[int]]:
        rows: list[np.ndarray] = []
        targets: list[int] = []
        for sequence in sequences:
            for pos in range(len(sequence) - 1):
                rows.append(self.distribution(sequence[: pos + 1]))
                targets.append(int(sequence[pos + 1]))
        return rows, targets

    @property
    def model_storage_bytes(self) -> int:
        return int(len(self.counts) * (8 + self.vocab_size * 8) + self.global_counts.nbytes)


@dataclass
class AdaptiveMemoryModel:
    vocab_size: int
    memory_size: int = 3
    window_size: int = 6
    alpha: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be > 0.")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0.")
        self.counts: dict[tuple[int, ...], np.ndarray] = {}
        self.global_counts = np.zeros(self.vocab_size, dtype=np.float64)

    def _memory_key(self, context: list[int]) -> tuple[int, ...]:
        tail = context[-self.window_size :]
        weighted = list(enumerate(tail))
        weighted.sort(key=lambda item: item[0], reverse=True)
        return tuple(int(token) for _, token in weighted[: self.memory_size])

    def fit(self, sequences: list[list[int]]) -> int:
        transitions = 0
        for sequence in sequences:
            for pos in range(len(sequence) - 1):
                key = self._memory_key(sequence[: pos + 1])
                nxt = int(sequence[pos + 1])
                if key not in self.counts:
                    self.counts[key] = np.zeros(self.vocab_size, dtype=np.float64)
                self.counts[key][nxt] += 1.0
                self.global_counts[nxt] += 1.0
                transitions += 1
        return transitions

    def distribution(self, context: list[int]) -> np.ndarray:
        counts = self.counts.get(self._memory_key(context), self.global_counts)
        smoothed = counts + self.alpha
        return smoothed / smoothed.sum()

    def iter_probability_rows(self, sequences: list[list[int]]) -> tuple[list[np.ndarray], list[int]]:
        rows: list[np.ndarray] = []
        targets: list[int] = []
        for sequence in sequences:
            for pos in range(len(sequence) - 1):
                rows.append(self.distribution(sequence[: pos + 1]))
                targets.append(int(sequence[pos + 1]))
        return rows, targets

    @property
    def model_storage_bytes(self) -> int:
        return int(len(self.counts) * (self.memory_size * 8 + self.vocab_size * 8) + self.global_counts.nbytes)
