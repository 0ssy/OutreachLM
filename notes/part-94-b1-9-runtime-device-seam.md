# OutreachLM — Part 94: B1.9 Runtime/Device Seam

## What it is
`runtime.py` introduces an explicit runtime abstraction for training execution concerns (device placement, batch placement, backward/step operations, and synchronization hook), with a concrete `SingleDeviceRuntime` implementation.

## Why it is there
Before this seam, runtime/device behavior was effectively embedded in trainer logic. This change creates the plug-in point needed for future distributed runtimes while preserving current single-device behavior.

## Why it is important
- Decouples `Trainer` from direct device/runtime assumptions.
- Establishes stable extension point for future DDP/FSDP implementations.
- Preserves current behavior while reducing migration risk for Phase C.
- Makes runtime semantics testable independently from model/trainer details.

## What would happen without it
- `Trainer` would continue to absorb runtime concerns and become harder to evolve.
- Distributed runtime adoption would require invasive trainer rewrites later.
- Device behavior would remain implicit and more error-prone during refactors.

## Scope
Implemented B1.9 seam only. No distributed runtime implementation yet.

## Added
- [runtime.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/runtime.py)
  - `RuntimeInfo`
  - `Runtime` protocol
  - `SingleDeviceRuntime`
    - `prepare_model`
    - `prepare_batch`
    - `zero_grad`
    - `backward`
    - `optimizer_step`
    - `synchronize` (no-op for single-device)

## Updated
- [trainer_core.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/trainer_core.py)
  - uses runtime seam types from `runtime.py`
  - trainer logic remains behavior-equivalent

## Added tests
- [test_runtime.py](C:/Users/josep/Desktop/OutreachLM/tests/test_runtime.py)
  - runtime metadata defaults
  - model placement
  - batch placement
  - backward + optimizer update flow
  - synchronize no-op

## Validation
Full suite run:
- `python -m pytest`
- result: `90 passed`

## B1 progression
- B1.1 ModelArtifact ✅
- B1.2 ModelRegistry ✅
- B1.3 ModelConfig ✅
- B1.4 TrainingConfig ✅
- B1.5 TrainerCore ✅
- B1.6 LossPlan ✅
- B1.7 EvaluationProfile ✅
- B1.8 TokenizerArtifact ✅
- B1.9 Runtime/Device seam ✅
- next: B1.10 migrate `train.py` and `train_v4.py` onto seams with no behavior change
