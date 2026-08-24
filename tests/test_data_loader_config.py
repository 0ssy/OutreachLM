import pytest
import torch

from outreachlm.data_loader_config import DataLoaderConfig, build_data_loader
from outreachlm.datasets import LanguageModelDataset, shard_indices


def test_data_loader_config_defaults_round_trip() -> None:
    config = DataLoaderConfig()
    restored = DataLoaderConfig.from_dict(config.to_dict())
    assert restored == config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"num_workers": -1},
        {"prefetch_factor": 0},
        {"num_workers": 0, "persistent_workers": True},
    ],
)
def test_data_loader_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DataLoaderConfig(**kwargs)


def test_build_data_loader_applies_scalability_options() -> None:
    token_ids = torch.arange(0, 80, dtype=torch.long)
    dataset = LanguageModelDataset(token_ids=token_ids, context_length=8)
    config = DataLoaderConfig(
        batch_size=4,
        num_workers=2,
        prefetch_factor=3,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )
    loader = build_data_loader(dataset, config)

    assert loader.batch_size == 4
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 3
    assert loader.persistent_workers is True
    assert loader.pin_memory is True
    assert loader.drop_last is True


def test_shard_indices_cover_without_overlap() -> None:
    total = 17
    all_indices = set()
    shard_sets = []
    for worker_id in range(4):
        shard = set(shard_indices(total, worker_id, 4))
        shard_sets.append(shard)
        all_indices.update(shard)

    for i in range(len(shard_sets)):
        for j in range(i + 1, len(shard_sets)):
            assert shard_sets[i].isdisjoint(shard_sets[j])
    assert all_indices == set(range(total))
