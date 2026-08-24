import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class LanguageModelDataset(Dataset):

    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids.detach().clone()

        self.context_length = context_length

        if len(self.token_ids) <= context_length:
            raise ValueError(
                "Token sequence must be longer than context length."
            )

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):

        x = self.token_ids[
            index:index + self.context_length
        ]

        y = self.token_ids[
            index + 1:index + self.context_length + 1
        ]

        return x, y


def shard_indices(total_items: int, worker_id: int, num_workers: int) -> range:
    if total_items < 0:
        raise ValueError("total_items must be >= 0.")
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0.")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers.")
    return range(worker_id, total_items, num_workers)


class ShardedLanguageModelIterableDataset(IterableDataset):
    def __init__(self, token_ids: torch.Tensor, context_length: int):
        self.token_ids = token_ids.detach().clone()
        self.context_length = context_length
        if len(self.token_ids) <= context_length:
            raise ValueError("Token sequence must be longer than context length.")

    def __iter__(self):
        total_samples = len(self.token_ids) - self.context_length
        worker_info = get_worker_info()
        if worker_info is None:
            indices = range(total_samples)
        else:
            indices = shard_indices(
                total_items=total_samples,
                worker_id=worker_info.id,
                num_workers=worker_info.num_workers,
            )

        for index in indices:
            x = self.token_ids[index:index + self.context_length]
            y = self.token_ids[index + 1:index + self.context_length + 1]
            yield x, y