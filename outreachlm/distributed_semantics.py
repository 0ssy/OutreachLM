from dataclasses import dataclass


@dataclass(frozen=True)
class BatchSemantics:
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int

    def __post_init__(self) -> None:
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be > 0.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0.")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0.")

    @property
    def effective_batch_size(self) -> int:
        return (
            self.per_device_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )
