# OutreachLM — Part 100: B2.5 Memory Accounting

## What it is
Added memory accounting to the runtime seam and exposed it in trainer step metrics.

## Why it is there
B2.5 requires the training system to report memory usage and model-state footprint, not just fail on OOM.

## Why it is important
- Gives measurable visibility into parameter/gradient/optimizer memory.
- Provides CUDA allocated/reserved/peak values when running on GPU.
- Keeps memory logic in runtime (deep module) instead of scattering checks in trainer code.

## What would happen without it
- Scaling decisions would remain guesswork.
- Memory regressions would be hard to detect early.
- Later B2/B3 optimization work would lack baseline instrumentation.

## Updated
- [runtime.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/runtime.py)
  - added `collect_memory_stats(model, optimizer)` to runtime interface
  - `SingleDeviceRuntime` now reports:
    - `parameter_count`
    - `trainable_parameter_count`
    - `parameter_bytes`
    - `gradient_bytes`
    - `buffer_bytes`
    - `optimizer_state_bytes`
    - `model_and_grad_bytes`
    - `estimated_training_state_bytes`
    - CUDA allocated/reserved/peak fields (when device is CUDA)
- [trainer_core.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/trainer_core.py)
  - includes memory snapshot in per-step metrics and finalize-accumulation metrics

## Added/updated tests
- [test_runtime.py](C:/Users/josep/Desktop/OutreachLM/tests/test_runtime.py)
  - verifies CPU memory-stat shape and sane values
- [test_trainer_core.py](C:/Users/josep/Desktop/OutreachLM/tests/test_trainer_core.py)
  - verifies trainer metrics include memory stats

## Validation
- Targeted:
  - `python -m pytest tests\test_runtime.py tests\test_trainer_core.py`
  - result: `22 passed`
- Full:
  - `python -m pytest`
  - result: `113 passed`
