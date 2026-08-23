# OutreachLM — Part 89: B1.4 TrainingConfig

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Scope
Implemented standalone typed training configuration seam (no trainer migration yet).

## Added
- [training_config.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/training_config.py)
  - `TrainingConfig` dataclass
  - fields:
    - `seed`
    - `steps`
    - `batch_size`
    - `learning_rate`
    - `warmup_steps`
    - `min_learning_rate_ratio`
    - `eval_interval`
    - `checkpoint_interval`
    - `label_smoothing`
  - validation guards:
    - `steps > 0`
    - `batch_size > 0`
    - `learning_rate > 0`
    - `warmup_steps >= 0`
    - `eval_interval > 0`
    - `checkpoint_interval > 0`
    - `0 < min_learning_rate_ratio <= 1`
    - `0 <= label_smoothing < 1`
  - serialization:
    - `to_dict()`
    - `from_dict(...)`

Defaults are aligned to current [train_v4.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train_v4.py) parser values.

## Added tests
- [test_training_config.py](C:/Users/josep/Desktop/OutreachLM/tests/test_training_config.py)
  - default-value match to `train_v4.py`
  - validation failure cases
  - round-trip serialization `config -> dict -> config`

## Validation
Full suite run:
- `python -m pytest`
- result: `38 passed`

## Separation status
- `TrainingConfig` currently includes only training-loop/runtime-like knobs.
- Experimental loss intervention fields remain excluded for future B1.6 `LossPlan`.
- [train_v4.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train_v4.py) was not modified in this step.