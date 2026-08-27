from dataclasses import dataclass
from typing import Any

import torch

from outreachlm.datasets import LanguageModelDataset


@dataclass
class StreamPosition:
    epoch: int = 0
    position: int = 0


class ResumableShardedBatchSource:
    def __init__(
        self,
        dataset: LanguageModelDataset,
        *,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 42,
        sequence_packing: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size.")
        if world_size <= 0:
            raise ValueError("world_size must be > 0.")
        if sequence_packing <= 0:
            raise ValueError("sequence_packing must be > 0.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.sequence_packing = sequence_packing
        self.position = StreamPosition()

    def _epoch_indices(self) -> list[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.position.epoch)
            perm = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in perm]
        return indices[self.rank::self.world_size]

    def _packed_item(self, start: int, shard_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        x_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        end = min(start + self.sequence_packing, len(shard_indices))
        for i in range(start, end):
            x, y = self.dataset[shard_indices[i]]
            x_parts.append(x)
            y_parts.append(y)
        return torch.cat(x_parts, dim=0), torch.cat(y_parts, dim=0)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        while True:
            shard_indices = self._epoch_indices()
            if len(shard_indices) == 0:
                raise RuntimeError("Sharded dataset is empty for the configured rank/world_size.")
            if self.position.position >= len(shard_indices):
                self.position.epoch += 1
                self.position.position = 0
                continue

            items: list[tuple[torch.Tensor, torch.Tensor]] = []
            consumed = 0
            index = self.position.position
            while index < len(shard_indices) and len(items) < self.batch_size:
                item = self._packed_item(index, shard_indices)
                items.append(item)
                index += self.sequence_packing
                consumed += self.sequence_packing

            if self.drop_last and len(items) < self.batch_size:
                self.position.epoch += 1
                self.position.position = 0
                continue

            self.position.position += consumed
            x = torch.stack([pair[0] for pair in items], dim=0)
            y = torch.stack([pair[1] for pair in items], dim=0)
            return x, y

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.position.epoch,
            "position": self.position.position,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "sequence_packing": self.sequence_packing,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("rank") != self.rank:
            raise ValueError("rank mismatch while restoring stream state.")
        if payload.get("world_size") != self.world_size:
            raise ValueError("world_size mismatch while restoring stream state.")
        self.position = StreamPosition(
            epoch=int(payload.get("epoch", 0)),
            position=int(payload.get("position", 0)),
        )
