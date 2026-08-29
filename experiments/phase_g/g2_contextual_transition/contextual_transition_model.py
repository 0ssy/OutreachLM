from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

PHASE_G_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE_G_ROOT) not in sys.path:
    sys.path.append(str(PHASE_G_ROOT))

from common.models import SparseNGramModel  # noqa: E402


@dataclass
class ContextualTransitionModel:
    vocab_size: int
    alpha: float = 0.1

    def __post_init__(self) -> None:
        self._model = SparseNGramModel(vocab_size=self.vocab_size, order=2, alpha=self.alpha)

    def fit(self, token_sequences: list[list[int]]) -> int:
        return self._model.fit(token_sequences)

    def distribution(self, context_tokens: list[int]):
        return self._model.distribution(context_tokens)

    def iter_probability_rows(self, sequences: list[list[int]]):
        return self._model.iter_probability_rows(sequences)

    @property
    def counts(self):
        return self._model.counts

    @property
    def parameter_count(self) -> int:
        return self._model.parameter_count

    @property
    def nonzero_parameters(self) -> int:
        return self._model.nonzero_parameters

    @property
    def model_storage_bytes(self) -> int:
        return self._model.model_storage_bytes

    def save(self, path: str | Path) -> None:
        self._model.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "ContextualTransitionModel":
        model = SparseNGramModel.load(path)
        wrapper = cls(vocab_size=model.vocab_size, alpha=model.alpha)
        wrapper._model = model
        return wrapper
