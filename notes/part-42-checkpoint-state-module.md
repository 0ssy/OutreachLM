# OutreachLM — Checkpoint State Module

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## What was added
Created [checkpoint.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/checkpoint.py) to centralize checkpoint state handling.

Functions added:
- `build_config(...)`
- `save_checkpoint(...)`
- `load_checkpoint(...)`
- `validate_config(...)`

Constants added:
- `CHECKPOINT_VERSION = 1`

## What checkpoint now represents
The checkpoint module is designed to persist experiment state, not only weights:
- model state dict
- optimizer (AdamW) state dict
- current step
- train loss
- best validation loss
- structured run config
- checkpoint version

## Config identity fields
`build_config(...)` captures:
- context length
- embedding dim
- batch size
- learning rate
- warmup steps
- min learning-rate ratio
- validation split
- seed
- absolute corpus path
- vocab size

## Validation behavior
`validate_config(...)` compares current run config to saved checkpoint config and raises a clear runtime error if mismatched.

This prevents silent resume with incompatible training/architecture/data settings.

## Train integration status for this step
Added checkpoint imports to [train.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py):

```python
from outreachlm.checkpoint import (
    build_config,
    save_checkpoint,
    load_checkpoint,
    validate_config,
)
```

Full wiring into the scheduler/resume flow is the next step.