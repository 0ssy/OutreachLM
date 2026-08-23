from dataclasses import dataclass
from typing import Any, Callable

import torch


LossFn = Callable[[dict[str, Any]], torch.Tensor]


@dataclass(frozen=True)
class LossPlanResult:
    total_loss: torch.Tensor
    term_losses: dict[str, torch.Tensor]
    weighted_term_losses: dict[str, torch.Tensor]


class LossTerm:
    def __init__(
        self,
        *,
        name: str,
        weight: float,
        loss_fn: LossFn,
        enabled: bool = True,
    ):
        if not name:
            raise ValueError("Loss term name cannot be empty.")
        self.name = name
        self.weight = float(weight)
        self.loss_fn = loss_fn
        self.enabled = enabled

    def compute(self, context: dict[str, Any]) -> torch.Tensor:
        value = self.loss_fn(context)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Loss term '{self.name}' must return a torch.Tensor.")
        return value


class TeacherLossTerm(LossTerm):
    def __init__(self, *, loss_fn: LossFn, weight: float = 1.0, enabled: bool = True):
        super().__init__(name="teacher", weight=weight, loss_fn=loss_fn, enabled=enabled)


class RecoveryLossTerm(LossTerm):
    def __init__(self, *, loss_fn: LossFn, weight: float = 1.0, enabled: bool = True):
        super().__init__(name="recovery", weight=weight, loss_fn=loss_fn, enabled=enabled)


class PostErrorLossTerm(LossTerm):
    def __init__(self, *, loss_fn: LossFn, weight: float = 1.0, enabled: bool = True):
        super().__init__(name="post_error", weight=weight, loss_fn=loss_fn, enabled=enabled)


class RolloutCalibrationLossTerm(LossTerm):
    def __init__(self, *, loss_fn: LossFn, weight: float = 1.0, enabled: bool = True):
        super().__init__(
            name="rollout_calib",
            weight=weight,
            loss_fn=loss_fn,
            enabled=enabled,
        )


class LossPlan:
    def __init__(self, terms: list[LossTerm]):
        if not terms:
            raise ValueError("LossPlan requires at least one loss term.")
        self.terms = terms

    def compute(self, context: dict[str, Any]) -> LossPlanResult:
        term_losses: dict[str, torch.Tensor] = {}
        weighted_term_losses: dict[str, torch.Tensor] = {}
        total_loss = None

        for term in self.terms:
            if not term.enabled:
                continue

            raw = term.compute(context)
            weighted = raw * term.weight

            term_losses[term.name] = raw
            weighted_term_losses[term.name] = weighted
            total_loss = weighted if total_loss is None else total_loss + weighted

        if total_loss is None:
            raise ValueError("LossPlan has no enabled loss terms.")

        return LossPlanResult(
            total_loss=total_loss,
            term_losses=term_losses,
            weighted_term_losses=weighted_term_losses,
        )
