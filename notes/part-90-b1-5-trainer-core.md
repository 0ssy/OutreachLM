# OutreachLM — Part 90: B1.5 TrainerCore

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Scope
Extracted a generic trainer seam without touching existing [train.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train.py) or [train_v4.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train_v4.py).

## Added
- [trainer_core.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/trainer_core.py)
  - `SingleDeviceRuntime`
  - `TrainerHooks`
  - `TrainerState`
  - `Trainer`
    - generic step mechanics: zero-grad, forward, loss, backward, optimizer step
    - optional scheduler step
    - optional evaluation/checkpoint callbacks with intervals
    - hook callbacks for step lifecycle and evaluation/checkpoint events

## Intentionally excluded
Per B1.5 boundaries, this module does **not** encode:
- recovery loss logic,
- post-error loss logic,
- rollout-calibration logic,
- forced-error/boundary indices,
- tokenizer assumptions.

Those remain in current training scripts and future seam modules (`LossPlan`, `EvaluationProfile`).

## Added tests
- [test_trainer_core.py](C:/Users/josep/Desktop/OutreachLM/tests/test_trainer_core.py)
  - trainer construction
  - parameter updates on one step
  - scheduler execution
  - loss value changes across deterministic steps
  - checkpoint callback + hook invocation
  - evaluation callback + hook invocation
  - single-device runtime batch movement
  - step lifecycle hook call counts

## Validation
Full suite run:
- `python -m pytest`
- result: `46 passed`

## B1 progression
- B1.1 ModelArtifact ✅
- B1.2 ModelRegistry ✅
- B1.3 ModelConfig ✅
- B1.4 TrainingConfig ✅
- B1.5 TrainerCore ✅
- next: B1.6 LossPlan