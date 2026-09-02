from __future__ import annotations

import math


class CosineWarmupScheduler:
    """Linear warmup followed by cosine annealing, driven by token count.

    Two corrections relative to the blueprint version:

    1. The blueprint's warmup branch computes
       `lr = lr_max * (tokens / warmup_tokens)`, which yields lr == 0.0 on the
       very first step (before any tokens are counted the ratio is 0). A zero
       learning rate means the first optimizer step is a no-op. Warmup here is
       clamped to a small floor (`lr_min`) so the first step still moves.

    2. The blueprint mutates `param_group['lr']` but never exposes the schedule
       for verification. This version also provides `lr_at_tokens`, a pure
       function of token count, so the curve can be validated numerically
       against the closed-form cosine without running an optimizer.
    """

    def __init__(
        self,
        optimizer,
        warmup_tokens: int,
        total_tokens: int,
        lr_max: float,
        lr_min: float,
    ) -> None:
        if warmup_tokens < 0:
            raise ValueError("warmup_tokens must be >= 0")
        if total_tokens <= warmup_tokens:
            raise ValueError("total_tokens must exceed warmup_tokens")
        self.optimizer = optimizer
        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.tokens_processed = 0

    def lr_at_tokens(self, tokens_processed: int) -> float:
        """Closed-form learning rate for a given cumulative token count."""
        if tokens_processed < self.warmup_tokens:
            warmup_fraction = tokens_processed / self.warmup_tokens if self.warmup_tokens else 1.0
            return max(self.lr_min, self.lr_max * warmup_fraction)

        span = self.total_tokens - self.warmup_tokens
        progress = (tokens_processed - self.warmup_tokens) / span
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine_decay

    def step_tokens(self, tokens_count: int) -> float:
        """Advance the schedule by `tokens_count` and apply the new lr."""
        self.tokens_processed += tokens_count
        lr = self.lr_at_tokens(self.tokens_processed)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
