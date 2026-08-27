import torch

from outreachlm.data_pipeline import ResumableShardedBatchSource
from outreachlm.datasets import LanguageModelDataset


def _dataset() -> LanguageModelDataset:
    token_ids = (torch.arange(0, 300, dtype=torch.long) % 64).clone()
    return LanguageModelDataset(token_ids=token_ids, context_length=8)


def test_resumable_sharded_batch_source_splits_indices_deterministically():
    source_rank0 = ResumableShardedBatchSource(_dataset(), batch_size=2, rank=0, world_size=2, shuffle=False)
    source_rank1 = ResumableShardedBatchSource(_dataset(), batch_size=2, rank=1, world_size=2, shuffle=False)
    rank0_indices = set(source_rank0._epoch_indices())
    rank1_indices = set(source_rank1._epoch_indices())
    assert rank0_indices.isdisjoint(rank1_indices)
    assert rank0_indices | rank1_indices == set(range(len(_dataset())))


def test_resumable_sharded_batch_source_restores_position():
    source_a = ResumableShardedBatchSource(_dataset(), batch_size=2, rank=0, world_size=1, shuffle=True, seed=7)
    source_a.next_batch()
    source_a.next_batch()
    state = source_a.state_dict()
    expected_next = source_a.next_batch()

    source_b = ResumableShardedBatchSource(_dataset(), batch_size=2, rank=0, world_size=1, shuffle=True, seed=7)
    source_b.load_state_dict(state)
    restored_next = source_b.next_batch()

    assert torch.equal(expected_next[0], restored_next[0])
    assert torch.equal(expected_next[1], restored_next[1])


def test_resumable_sharded_batch_source_sequence_packing_changes_sequence_length():
    source = ResumableShardedBatchSource(
        _dataset(),
        batch_size=2,
        rank=0,
        world_size=1,
        sequence_packing=3,
        shuffle=False,
    )
    x, y = source.next_batch()
    assert x.shape == (2, 24)
    assert y.shape == (2, 24)
