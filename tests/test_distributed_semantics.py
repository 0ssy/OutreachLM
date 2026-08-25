import pytest

from outreachlm.distributed_semantics import BatchSemantics


def test_effective_batch_size_formula() -> None:
    semantics = BatchSemantics(
        per_device_batch_size=4,
        gradient_accumulation_steps=8,
        world_size=2,
    )
    assert semantics.effective_batch_size == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"per_device_batch_size": 0, "gradient_accumulation_steps": 1, "world_size": 1},
        {"per_device_batch_size": 1, "gradient_accumulation_steps": 0, "world_size": 1},
        {"per_device_batch_size": 1, "gradient_accumulation_steps": 1, "world_size": 0},
    ],
)
def test_batch_semantics_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BatchSemantics(**kwargs)
