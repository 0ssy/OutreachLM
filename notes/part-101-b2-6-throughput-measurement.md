# OutreachLM — Part 101: B2.6 Throughput Measurement

## What it is
Added throughput and timing instrumentation to trainer-step metrics, including accumulation-aware optimizer-step rates and lifetime token accounting.

## Why it is there
B2.6 requires measurable training speed so configurations can be compared objectively.

## Why it is important
- Reports machine-readable throughput (`tokens/sec`, `samples/sec`, micro/optimizer step rates).
- Tracks `tokens_processed` and `samples_processed` across run lifetime.
- Exposes timing breakdowns (`data`, `forward`, `backward`, `optimizer`, total step).
- Preserves accumulation semantics by computing optimizer-step throughput from optimizer-step progression.

## What would happen without it
- Performance tuning would remain guesswork.
- Accumulation changes could not be compared fairly on throughput.
- Later telemetry and scaling validation would lack core speed baselines.

## Updated
- [trainer_core.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/trainer_core.py)
  - added cumulative counters in trainer state:
    - `total_tokens_processed`
    - `total_samples_processed`
    - `run_start_time`
  - added throughput computation and timing breakdown per step
  - adds `throughput` block to both `train_step` and finalize-accumulation metrics

## Added/updated tests
- [test_trainer_core.py](C:/Users/josep/Desktop/OutreachLM/tests/test_trainer_core.py)
  - verifies throughput keys and non-negative timings
  - verifies accumulation-aware progression (`tokens_processed` grows by micro-step, optimizer-step rate stays zero until optimizer step)

## Validation
- Targeted:
  - `python -m pytest tests\test_trainer_core.py`
  - result: `15 passed`
- Full:
  - `python -m pytest`
  - result: `115 passed`
