from __future__ import annotations

import random


BASE_CORPUS = [
    "the cat sat on the mat",
    "the cat ate the fish",
    "the dog sat on the mat",
    "the dog ate the food",
    "the boy saw the dog",
    "the boy fed the cat",
    "the girl saw the cat",
    "the girl fed the dog",
]

CONTEXT_AMBIGUITY_CORPUS = [
    "the cat will eat",
    "the dog will sleep",
    "the cat will purr",
    "the dog will bark",
    "the cat eats fish",
    "the dog eats food",
    "the cat sees birds",
    "the dog sees people",
    "the cat likes milk",
    "the dog likes bones",
]

LONG_CONTEXT_CORPUS = [
    "topic alpha begin detail one then filler words and final answer yes",
    "topic beta begin detail two then filler words and final answer no",
    "topic alpha begin detail three then filler words and final answer yes",
    "topic beta begin detail four then filler words and final answer no",
]


def build_train_eval_split(lines: list[str], seed: int, eval_ratio: float = 0.3) -> tuple[list[str], list[str]]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1.")
    if len(lines) < 2:
        raise ValueError("Need at least two lines for train/eval split.")
    rng = random.Random(seed)
    indices = list(range(len(lines)))
    rng.shuffle(indices)
    eval_size = max(1, int(round(len(lines) * eval_ratio)))
    eval_indices = set(indices[:eval_size])
    train = [line for idx, line in enumerate(lines) if idx not in eval_indices]
    eval_set = [line for idx, line in enumerate(lines) if idx in eval_indices]
    return train, eval_set
