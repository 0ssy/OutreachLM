from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
import random
from typing import Any, Iterable

import numpy as np

from outreachlm.train import load_corpus, split_corpus

SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")


@dataclass(frozen=True)
class PhaseGSparseProfile:
    top_k: int
    context_blend: float
    tail_temperature: float
    head_temperature: float
    head_context_mix: float
    residual_gain: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "context_blend": self.context_blend,
            "tail_temperature": self.tail_temperature,
            "head_temperature": self.head_temperature,
            "head_context_mix": self.head_context_mix,
            "residual_gain": self.residual_gain,
        }


@dataclass(frozen=True)
class PhaseGHybridConfig:
    default_profile: PhaseGSparseProfile = PhaseGSparseProfile(
        top_k=4,
        context_blend=0.9,
        tail_temperature=0.75,
        head_temperature=1.1,
        head_context_mix=0.0,
        residual_gain=0.8,
    )
    oov_profile: PhaseGSparseProfile = PhaseGSparseProfile(
        top_k=4,
        context_blend=0.9,
        tail_temperature=0.9,
        head_temperature=0.8,
        head_context_mix=0.0,
        residual_gain=0.6,
    )
    unk_probability_threshold: float = 0.02
    smoothing_alpha: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_profile": self.default_profile.to_dict(),
            "oov_profile": self.oov_profile.to_dict(),
            "unk_probability_threshold": self.unk_probability_threshold,
            "smoothing_alpha": self.smoothing_alpha,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseGHybridConfig":
        default = payload["default_profile"]
        oov = payload["oov_profile"]
        return cls(
            default_profile=PhaseGSparseProfile(**default),
            oov_profile=PhaseGSparseProfile(**oov),
            unk_probability_threshold=float(payload["unk_probability_threshold"]),
            smoothing_alpha=float(payload["smoothing_alpha"]),
        )


@dataclass(frozen=True)
class WordTokenizer:
    token_to_id: dict[str, int]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def id_to_token(self) -> list[str]:
        output = [""] * len(self.token_to_id)
        for token, token_id in self.token_to_id.items():
            output[token_id] = token
        return output

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        output: list[int] = []
        if add_bos:
            output.append(self.bos_id)
        for token in text.strip().split():
            output.append(self.token_to_id.get(token, self.unk_id))
        if add_eos:
            output.append(self.eos_id)
        return output

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        words: list[str] = []
        vocab = self.id_to_token
        for token_id in token_ids:
            token = vocab[token_id] if 0 <= token_id < len(vocab) else "<UNK>"
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words)

    @classmethod
    def from_lines(cls, lines: list[str]) -> "WordTokenizer":
        token_to_id: dict[str, int] = {}
        for line in lines:
            for token in line.strip().split():
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
        for special in SPECIAL_TOKENS:
            token_to_id.setdefault(special, len(token_to_id))
        return cls(token_to_id=token_to_id)


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
        self.counts: dict[Context, dict[int, float]] = {}
        self.context_totals: dict[Context, float] = {}
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
                    self.counts[context] = {}
                    self.context_totals[context] = 0.0
                self.counts[context][next_token] = self.counts[context].get(next_token, 0.0) + 1.0
                self.context_totals[context] += 1.0
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
        sparse_counts = self.counts.get(key)
        if sparse_counts is None:
            smoothed = self.global_counts + self.alpha
            return smoothed / smoothed.sum()

        total = self.context_totals[key]
        denominator = total + (self.alpha * self.vocab_size)
        out = np.full(self.vocab_size, self.alpha / denominator, dtype=np.float64)
        for token_id, count in sparse_counts.items():
            out[token_id] = (count + self.alpha) / denominator
        return out

    def to_payload(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "order": self.order,
            "alpha": self.alpha,
            "counts": self.counts,
            "context_totals": self.context_totals,
            "global_counts": self.global_counts,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SparseNGramModel":
        model = cls(vocab_size=int(payload["vocab_size"]), order=int(payload["order"]), alpha=float(payload["alpha"]))
        counts_payload = payload["counts"]
        if counts_payload and isinstance(next(iter(counts_payload.values())), np.ndarray):
            converted_counts: dict[Context, dict[int, float]] = {}
            converted_totals: dict[Context, float] = {}
            for context, vector in counts_payload.items():
                indices = np.nonzero(vector)[0]
                converted_counts[context] = {int(i): float(vector[i]) for i in indices}
                converted_totals[context] = float(vector.sum())
            model.counts = converted_counts
            model.context_totals = converted_totals
        else:
            model.counts = counts_payload
            model.context_totals = payload.get(
                "context_totals",
                {context: float(sum(token_counts.values())) for context, token_counts in counts_payload.items()},
            )
        model.global_counts = payload["global_counts"]
        return model


def _evaluate_rows(probability_rows: Iterable[np.ndarray], targets: Iterable[int]) -> dict[str, float]:
    total = 0
    correct = 0
    nll_sum = 0.0
    for probs, target in zip(probability_rows, targets):
        prediction = int(np.argmax(probs))
        if prediction == int(target):
            correct += 1
        p = float(probs[int(target)])
        nll_sum += -math.log(max(p, 1e-12))
        total += 1
    if total == 0:
        raise ValueError("No samples provided for evaluation.")
    cross_entropy = nll_sum / total
    return {
        "accuracy": correct / total,
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy),
        "count": float(total),
    }


def _entropy(probabilities: np.ndarray, epsilon: float = 1e-12) -> float:
    p = np.clip(probabilities, epsilon, 1.0)
    return float(-(p * np.log(p)).sum())


class PhaseGHybridRuntime:
    def __init__(
        self,
        tokenizer: WordTokenizer,
        config: PhaseGHybridConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or PhaseGHybridConfig()
        self.g2 = SparseNGramModel(vocab_size=tokenizer.vocab_size, order=2, alpha=self.config.smoothing_alpha)
        self.local = SparseNGramModel(vocab_size=tokenizer.vocab_size, order=2, alpha=self.config.smoothing_alpha)
        self.medium = SparseNGramModel(vocab_size=tokenizer.vocab_size, order=4, alpha=self.config.smoothing_alpha)
        self.global_model = SparseNGramModel(vocab_size=tokenizer.vocab_size, order=1, alpha=self.config.smoothing_alpha)
        self._fitted = False

    def fit(self, train_lines: list[str]) -> None:
        train_sequences = [self.tokenizer.encode(line, add_bos=True, add_eos=True) for line in train_lines]
        self.g2.fit(train_sequences)
        self.local.fit(train_sequences)
        self.medium.fit(train_sequences)
        self.global_model.fit(train_sequences)
        self._fitted = True

    @classmethod
    def from_corpus_lines(
        cls,
        lines: list[str],
        *,
        config: PhaseGHybridConfig | None = None,
        seed: int = 1337,
        eval_ratio: float = 0.3,
    ) -> tuple["PhaseGHybridRuntime", list[str], list[str]]:
        if not 0.0 < eval_ratio < 1.0:
            raise ValueError("eval_ratio must be between 0 and 1.")
        rng = random.Random(seed)
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        eval_size = max(1, int(round(len(lines) * eval_ratio)))
        eval_indices = set(indices[:eval_size])
        train_lines = [line for idx, line in enumerate(lines) if idx not in eval_indices]
        eval_lines = [line for idx, line in enumerate(lines) if idx in eval_indices]
        tokenizer = WordTokenizer.from_lines(lines)
        runtime = cls(tokenizer=tokenizer, config=config)
        runtime.fit(train_lines)
        return runtime, train_lines, eval_lines

    @classmethod
    def from_corpus_path(
        cls,
        corpus_path: str | Path,
        *,
        validation_split: float,
        config: PhaseGHybridConfig | None = None,
        max_train_lines: int | None = None,
        max_eval_lines: int | None = None,
    ) -> tuple["PhaseGHybridRuntime", list[str], list[str]]:
        text = load_corpus(corpus_path)
        training_text, validation_text = split_corpus(text, validation_split)
        train_lines = [line.strip() for line in training_text.splitlines() if line.strip()]
        eval_lines = [line.strip() for line in validation_text.splitlines() if line.strip()]
        if max_train_lines is not None:
            if max_train_lines <= 0:
                raise ValueError("max_train_lines must be > 0 when provided.")
            train_lines = train_lines[:max_train_lines]
        if max_eval_lines is not None:
            if max_eval_lines <= 0:
                raise ValueError("max_eval_lines must be > 0 when provided.")
            eval_lines = eval_lines[:max_eval_lines]
        if not train_lines or not eval_lines:
            raise ValueError("Training and validation line sets must both be non-empty.")
        tokenizer = WordTokenizer.from_lines(train_lines + eval_lines)
        runtime = cls(tokenizer=tokenizer, config=config)
        runtime.fit(train_lines)
        return runtime, train_lines, eval_lines

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PhaseGHybridRuntime is not fitted.")

    def _g6_distribution(self, context_tokens: list[int]) -> np.ndarray:
        out = (
            0.5 * self.local.distribution(context_tokens)
            + 0.35 * self.medium.distribution(context_tokens)
            + 0.15 * self.global_model.distribution(context_tokens)
        )
        return out / out.sum()

    @staticmethod
    def _forced_unk_sparse(
        *,
        g6: np.ndarray,
        g2: np.ndarray,
        unigram: np.ndarray,
        unk_id: int,
        profile: PhaseGSparseProfile,
    ) -> np.ndarray:
        top_k = profile.top_k
        idx = np.argpartition(g6, -top_k)[-top_k:]
        if unk_id not in idx:
            smallest_local = int(np.argmin(g6[idx]))
            idx[smallest_local] = int(unk_id)
        idx = np.unique(idx)
        if len(idx) > top_k:
            keep = np.argsort(g6[idx])[-top_k:]
            idx = idx[keep]

        in_topk = np.zeros_like(g6, dtype=bool)
        in_topk[idx] = True
        candidate_probs = np.zeros_like(g6)
        candidate_probs[idx] = g6[idx]
        candidate_mass = float(candidate_probs.sum())
        residual_mass = max(0.0, min(1.0, (1.0 - candidate_mass) * profile.residual_gain))

        g6_head = np.clip(g6[idx], 1e-12, 1.0)
        g6_head = np.power(g6_head, profile.head_temperature)
        g2_head = np.clip(g2[idx], 1e-12, 1.0)
        g2_head = g2_head / max(float(g2_head.sum()), 1e-12)
        blended_head = ((1.0 - profile.head_context_mix) * g6_head) + (profile.head_context_mix * g2_head)
        blended_head = blended_head / max(float(blended_head.sum()), 1e-12)

        background = (profile.context_blend * g2) + ((1.0 - profile.context_blend) * unigram)
        background = np.clip(background, 1e-12, 1.0)
        background = np.power(background, profile.tail_temperature)

        out = np.zeros_like(g6)
        out[idx] = (1.0 - residual_mass) * blended_head
        tail_mask = ~in_topk
        tail_q = np.where(tail_mask, background, 0.0)
        tail_q_sum = float(tail_q.sum())
        if tail_q_sum > 0.0 and residual_mass > 0.0:
            out[tail_mask] = residual_mass * (tail_q[tail_mask] / tail_q_sum)
        out = out / out.sum()
        return out

    def distribution(self, context_tokens: list[int]) -> np.ndarray:
        self._ensure_fitted()
        g6 = self._g6_distribution(context_tokens)
        g2 = self.g2.distribution(context_tokens)
        unigram = self.global_model.global_counts + self.config.smoothing_alpha
        unigram = unigram / unigram.sum()

        use_oov = bool(
            (self.tokenizer.unk_id in context_tokens)
            or (float(g6[self.tokenizer.unk_id]) >= self.config.unk_probability_threshold)
        )
        profile = self.config.oov_profile if use_oov else self.config.default_profile
        return self._forced_unk_sparse(
            g6=g6,
            g2=g2,
            unigram=unigram,
            unk_id=self.tokenizer.unk_id,
            profile=profile,
        )

    def distribution_from_text(self, prefix_text: str) -> np.ndarray:
        context = self.tokenizer.encode(prefix_text, add_bos=True, add_eos=False)
        return self.distribution(context)

    def evaluate_lines(self, lines: list[str]) -> dict[str, Any]:
        self._ensure_fitted()
        sequences = [self.tokenizer.encode(line, add_bos=True, add_eos=True) for line in lines]
        rows: list[np.ndarray] = []
        targets: list[int] = []
        for sequence in sequences:
            for pos in range(len(sequence) - 1):
                context = sequence[: pos + 1]
                rows.append(self.distribution(context))
                targets.append(int(sequence[pos + 1]))

        metrics = _evaluate_rows(rows, targets)
        mass_errors = [abs(float(row.sum()) - 1.0) for row in rows]
        entropies = [_entropy(row) for row in rows]
        return {
            **metrics,
            "mass_error_max": float(np.max(np.asarray(mass_errors, dtype=np.float64))),
            "entropy_mean": float(np.mean(np.asarray(entropies, dtype=np.float64))),
            "support_mean@1e-4": float(
                np.mean(np.asarray([int(np.count_nonzero(row > 1e-4)) for row in rows], dtype=np.float64))
            ),
            "operations_per_token_estimate": float(self.config.default_profile.top_k + 4),
        }

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "phase_g_hybrid_runtime",
            "config": self.config.to_dict(),
            "token_to_id": self.tokenizer.token_to_id,
            "models": {
                "g2": self.g2.to_payload(),
                "local": self.local.to_payload(),
                "medium": self.medium.to_payload(),
                "global": self.global_model.to_payload(),
            },
        }
        with open(target, "wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> "PhaseGHybridRuntime":
        with open(Path(path), "rb") as file:
            payload = pickle.load(file)
        if payload.get("format") != "phase_g_hybrid_runtime":
            raise ValueError("Unsupported Phase G runtime artifact format.")
        tokenizer = WordTokenizer(token_to_id=payload["token_to_id"])
        config = PhaseGHybridConfig.from_dict(payload["config"])
        runtime = cls(tokenizer=tokenizer, config=config)
        runtime.g2 = SparseNGramModel.from_payload(payload["models"]["g2"])
        runtime.local = SparseNGramModel.from_payload(payload["models"]["local"])
        runtime.medium = SparseNGramModel.from_payload(payload["models"]["medium"])
        runtime.global_model = SparseNGramModel.from_payload(payload["models"]["global"])
        runtime._fitted = True
        return runtime
