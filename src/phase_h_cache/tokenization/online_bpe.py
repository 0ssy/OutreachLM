from __future__ import annotations

from collections import Counter


class OnlineBPETokenizer:
    def __init__(self, vocab_limit: int = 32000) -> None:
        if vocab_limit <= 256:
            raise ValueError("vocab_limit must be > 256")
        self.vocab_limit = vocab_limit
        self.next_token_id = 256
        self.merges: list[tuple[int, int, int]] = []

    @property
    def vocab_size(self) -> int:
        return self.next_token_id

    @staticmethod
    def _replace_all(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out

    @staticmethod
    def _count_pairs(tokens: list[int]) -> Counter[tuple[int, int]]:
        pairs: Counter[tuple[int, int]] = Counter()
        for idx in range(len(tokens) - 1):
            pairs[(tokens[idx], tokens[idx + 1])] += 1
        return pairs

    def learn_from_stream(self, text: str, *, merge_steps: int) -> None:
        if merge_steps <= 0:
            return
        tokens = list(text.encode("utf-8", errors="ignore"))
        if len(tokens) < 2:
            return

        for _ in range(merge_steps):
            if self.next_token_id >= self.vocab_limit:
                return
            for left, right, new_id in self.merges:
                tokens = self._replace_all(tokens, (left, right), new_id)
            pairs = self._count_pairs(tokens)
            if not pairs:
                return
            best_pair, count = max(pairs.items(), key=lambda item: item[1])
            if count < 2:
                return
            new_id = self.next_token_id
            self.next_token_id += 1
            self.merges.append((best_pair[0], best_pair[1], new_id))
            tokens = self._replace_all(tokens, best_pair, new_id)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8", errors="ignore"))
        for left, right, new_id in self.merges:
            tokens = self._replace_all(tokens, (left, right), new_id)
        return tokens

