# OutreachLM — Part 97: B2.2 Scalable Checkpoint Format

## What it is
Refactored checkpoint handling into a dedicated training-checkpoint contract that is explicitly separate from model artifacts.

## Why it is there
B2.2 requires a checkpoint format that can resume training process state, not just store model weights. This creates the training-state seam needed for long-running, restartable jobs.

## Why it is important
- Distinguishes "trained model artifact" from "resume training state."
- Adds explicit fields for trainer state, RNG state, and metadata.
- Preserves backward compatibility by loading legacy v1/v2 checkpoint payloads.
- Provides a stable format to extend later with scheduler/scaler/runtime state.

## What would happen without it
- Resume semantics would stay implicit and fragile.
- Training-state evolution would continue as ad-hoc fields.
- Future B2/B3 resumability features would require risky rewrites.

## Updated
- [checkpoint.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/checkpoint.py)
  - introduces `TrainingCheckpoint` contract
  - `CHECKPOINT_VERSION` advanced to `3`
  - saves explicit fields:
    - `model_state`
    - `optimizer_state`
    - `scheduler_state` (optional)
    - `scaler_state` (optional)
    - `trainer_state`
    - `rng_state`
    - `config`
    - `metadata`
  - restores RNG state on load
  - normalizes legacy payloads (v1/v2) into the new format for compatibility

## Added tests
- [test_checkpoint.py](C:/Users/josep/Desktop/OutreachLM/tests/test_checkpoint.py)
  - v3 checkpoint round-trip
  - legacy v2 compatibility load
  - missing-required-field validation error

## Validation
- Targeted:
  - `python -m pytest tests\test_checkpoint.py`
  - result: `3 passed`
- Full:
  - `python -m pytest`
  - result: `102 passed`
