from __future__ import annotations


def max_repetition_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def repeated_ngram_count(tokens: list[int], n: int) -> int:
    if n <= 0:
        raise ValueError("n must be > 0")
    if len(tokens) < n:
        return 0
    seen: set[tuple[int, ...]] = set()
    repeats = 0
    for idx in range(len(tokens) - n + 1):
        gram = tuple(tokens[idx : idx + n])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    return repeats

