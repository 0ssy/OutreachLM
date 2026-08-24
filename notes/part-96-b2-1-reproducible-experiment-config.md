# OutreachLM — Part 96: B2.1 Reproducible Experiment Configuration

## What it is
Introduced a new `ExperimentConfig` seam and wired `--config` support into both training entrypoints so one JSON file can define reproducible experiment defaults.

## Why it is there
B2.1 requires a single source of truth for experiment setup instead of scattered CLI-only settings. This allows consistent, replayable runs and clearer provenance.

## Why it is important
- Centralizes model/training/evaluation/runtime/metadata settings into one module.
- Preserves behavior-safe CLI ergonomics by keeping explicit flags as highest priority.
- Creates a deep module seam for future B2 steps (telemetry/checkpoint/resume) to consume shared experiment context.

## What would happen without it
- Reproducibility would continue depending on manual CLI reconstruction.
- Config drift risk would remain high across repeated runs.
- Later scalability features would duplicate configuration glue in multiple scripts.

## Added
- [experiment_config.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/experiment_config.py)
  - `ExperimentConfig`
  - `RuntimeConfig`
  - `ExperimentMetadata`
  - `save_experiment_config` / `load_experiment_config`
  - `to_train_cli_defaults` / `to_train_v4_cli_defaults`

## Updated
- [train.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train.py)
  - adds `--config`
  - applies config defaults before final parse
  - preserves CLI-over-config precedence
- [train_v4.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train_v4.py)
  - adds `--config`
  - applies config defaults before final parse
  - preserves CLI-over-config precedence

## Added tests
- [test_experiment_config.py](C:/Users/josep/Desktop/OutreachLM/tests/test_experiment_config.py)
  - serialization/deserialization round-trip
  - validation failures
  - train/train_v4 default mapping checks
- [test_experiment_config_cli.py](C:/Users/josep/Desktop/OutreachLM/tests/test_experiment_config_cli.py)
  - config-driven arg defaults for both entrypoints
  - explicit CLI override precedence

## Validation
- Targeted:
  - `python -m pytest tests\test_experiment_config.py tests\test_experiment_config_cli.py`
  - result: `9 passed`
- Full:
  - `python -m pytest`
  - result: `99 passed`
