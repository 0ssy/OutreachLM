# OutreachLM — Part 103: B2.8 Resumable Training

## What it is
Extended resume support to cover richer trainer/process state and added deterministic resume-equivalence validation.

## Why it is there
B2.8 requires that interrupted training can restart from checkpoint and continue as the same run, not a partial re-initialization.

## Why it is important
- Restores RNG state for deterministic continuation.
- Supports scheduler/scaler restoration paths through checkpoint load.
- Adds explicit trainer-state serialization/load hooks for accumulation and throughput counters.
- Proves resume correctness with a continuous-vs-restart equivalence test.

## What would happen without it
- Resume behavior could drift from uninterrupted training.
- Gradient-accumulation and scheduler state could reset silently.
- Crash recovery confidence would remain low.

## Updated
- [checkpoint.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/checkpoint.py)
  - `load_checkpoint(...)` now accepts optional `scheduler` and `runtime`
  - restores scheduler and scaler state when provided
  - returns `scheduler_state`, `scaler_state`, and `rng_state` in load result
- [trainer_core.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/trainer_core.py)
  - added `state_dict()` export for trainer + scheduler + scaler state
  - added `load_state_dict(...)` to restore trainer progression and attached states
- [train.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train.py)
  - enriches checkpoint save with trainer-state metadata (`optimizer_step`, `micro_step`, `last_learning_rate`, `interval_loss_count`)
  - stores resume metadata on checkpoint writes

## Added/updated tests
- [test_checkpoint.py](C:/Users/josep/Desktop/OutreachLM/tests/test_checkpoint.py)
  - added continuous-vs-checkpoint-resume deterministic equivalence test
- [test_trainer_core.py](C:/Users/josep/Desktop/OutreachLM/tests/test_trainer_core.py)
  - added trainer state export/load round-trip test

## Validation
- Targeted:
  - `python -m pytest tests\test_checkpoint.py tests\test_trainer_core.py`
  - result: `20 passed`
- Full:
  - `python -m pytest`
  - result: `124 passed`
