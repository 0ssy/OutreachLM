from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 8
    num_workers: int = 0
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = False
    drop_last: bool = False
    shuffle: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be > 0.")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataLoaderConfig":
        return cls(
            batch_size=payload.get("batch_size", 8),
            num_workers=payload.get("num_workers", 0),
            prefetch_factor=payload.get("prefetch_factor", 2),
            persistent_workers=payload.get("persistent_workers", False),
            pin_memory=payload.get("pin_memory", False),
            drop_last=payload.get("drop_last", False),
            shuffle=payload.get("shuffle", False),
        )


def build_data_loader(
    dataset: Dataset,
    config: DataLoaderConfig,
    *,
    generator: torch.Generator | None = None,
    distributed_rank: int | None = None,
    distributed_world_size: int | None = None,
    distributed_seed: int = 42,
) -> DataLoader:
    sampler = None
    if (
        distributed_rank is not None
        and distributed_world_size is not None
        and not isinstance(dataset, IterableDataset)
    ):
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed_world_size,
            rank=distributed_rank,
            shuffle=config.shuffle,
            drop_last=config.drop_last,
            seed=distributed_seed,
        )

    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": config.shuffle if sampler is None else False,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "drop_last": config.drop_last,
        "generator": generator,
        "sampler": sampler,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor
        kwargs["persistent_workers"] = config.persistent_workers
    return DataLoader(dataset, **kwargs)
