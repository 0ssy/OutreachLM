from __future__ import annotations

from typing import Iterable, Iterator

import torch


class DynamicTokenPacker:
    """Chopped dynamic token packing: concatenate, then cut to exact blocks.

    Padding-based batching spends compute on `<pad>` positions that carry no
    information. This packer concatenates the incoming token stream into one
    continuous run and slices it into exact `block_size` chunks, so every
    position in every batch is a real token.

    A residual buffer carries leftover tokens across calls so no tokens are
    dropped at document boundaries.
    """

    def __init__(self, block_size: int) -> None:
        if block_size < 2:
            raise ValueError("block_size must be >= 2 to form (input, target) pairs")
        self.block_size = block_size
        self._residual: list[int] = []

    def reset(self) -> None:
        self._residual.clear()

    @property
    def residual_token_count(self) -> int:
        return len(self._residual)

    def pack(
        self,
        token_sequences: Iterable[list[int]],
        *,
        batch_size: int,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (inputs, targets) batches of exactly `block_size - 1` width."""
        buffer = self._residual
        pending: list[list[int]] = []

        for sequence in token_sequences:
            buffer.extend(sequence)
            while len(buffer) >= self.block_size:
                pending.append(buffer[: self.block_size])
                del buffer[: self.block_size]
                if len(pending) == batch_size:
                    yield self._to_tensors(pending)
                    pending = []

        if pending:
            yield self._to_tensors(pending)

        self._residual = buffer

    @staticmethod
    def _to_tensors(blocks: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.tensor(blocks, dtype=torch.long)
        return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous()


def padded_batch(
    token_sequences: list[list[int]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Baseline padded batching, kept for measuring what packing actually saves.

    Returns (inputs, targets, wasted_fraction) where `wasted_fraction` is the
    share of positions occupied by padding -- the compute that dynamic packing
    eliminates.
    """
    width = max(len(sequence) for sequence in token_sequences)
    padded = [sequence + [pad_id] * (width - len(sequence)) for sequence in token_sequences]
    tensor = torch.tensor(padded, dtype=torch.long)
    real = sum(len(sequence) for sequence in token_sequences)
    total = tensor.numel()
    wasted = (total - real) / total if total else 0.0
    return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous(), wasted
