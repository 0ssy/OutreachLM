# OutreachLM — Part 98: B2.3 Gradient Accumulation

## What it is
Added gradient accumulation support to the trainer seam so training can run multiple micro-steps before one optimizer step.

## Why it is there
B2.3 is the first memory-scaling feature: increase effective batch size without requiring the full batch to fit in memory at once.

## Why it is important
- Separates micro-step progression from optimizer-step progression in trainer state.
- Makes optimizer/eval/checkpoint cadence accumulation-aware.
- Preserves default behavior (`gradient_accumulation_steps=1`) while enabling larger effective batch sizes.

## What would happen without it
- Effective batch size scaling would remain memory-bound to per-step batch size.
- Step accounting would stay tied to micro updates, which breaks scalable scheduling semantics.

## Updated
- [training_config.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/training_config.py)
  - added `gradient_accumulation_steps` with validation and serialization support
- [trainer_core.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/trainer_core.py)
  - added accumulation-aware execution in `train_step`
  - tracks `micro_step` and `optimizer_step` in trainer state
  - evaluates/checkpoints on optimizer-step boundaries
  - added `finalize_accumulation` and optional `flush_partial_accumulation` in `run_steps`

## Added/updated tests
- [test_training_config.py](C:/Users/josep/Desktop/OutreachLM/tests/test_training_config.py)
  - default and validation coverage for `gradient_accumulation_steps`
- [test_trainer_core.py](C:/Users/josep/Desktop/OutreachLM/tests/test_trainer_core.py)
  - optimizer step delayed until accumulation boundary
  - accumulated gradients match equivalent larger-batch update
  - partial accumulation flush behavior
  - eval/checkpoint cadence follows optimizer steps under accumulation

## Validation
- Targeted:
  - `python -m pytest tests\test_training_config.py tests\test_trainer_core.py`
  - result: `25 passed`
- Full:
  - `python -m pytest`
  - result: `107 passed`
