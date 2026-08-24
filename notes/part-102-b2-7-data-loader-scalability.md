# OutreachLM — Part 102: B2.7 Data-Loader Scalability

## What it is
Added a typed data-loader seam with scalable worker/prefetch options, wired into `train.py` validation loading, and added dataset worker-sharding support.

## Why it is there
B2.7 requires one-machine data path scalability and explicit control over loader behavior instead of hard-coded `DataLoader(...)` parameters.

## Why it is important
- Exposes loader knobs (`num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory`, `drop_last`, `shuffle`) through a typed module.
- Keeps loader construction centralized and testable.
- Adds worker-sharding logic for iterable dataset consumption so worker partitions are disjoint and complete.
- Keeps default behavior unchanged (`num_workers=0`) for behavior safety.

## What would happen without it
- Data loading would remain mostly fixed and harder to tune.
- Worker/prefetch optimization would require repeated ad-hoc script edits.
- Worker partition correctness would stay implicit.

## Added
- [data_loader_config.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/data_loader_config.py)
  - `DataLoaderConfig`
  - `build_data_loader(...)`
- [datasets.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/datasets.py)
  - `shard_indices(...)`
  - `ShardedLanguageModelIterableDataset`

## Updated
- [train.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train.py)
  - evaluation data loader now uses `DataLoaderConfig` + `build_data_loader`
  - new CLI flags:
    - `--eval-num-workers`
    - `--eval-prefetch-factor`
    - `--eval-persistent-workers`
    - `--eval-pin-memory`
    - `--eval-drop-last`
  - checkpoint run-config now records `eval_loader_config`
- [checkpoint.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/checkpoint.py)
  - `build_config(...)` accepts optional `eval_loader_config`
- [experiment_config.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/experiment_config.py)
  - `to_train_cli_defaults(...)` now applies all `script_args` keys (with non-None values)

## Added/updated tests
- [test_data_loader_config.py](C:/Users/josep/Desktop/OutreachLM/tests/test_data_loader_config.py)
  - config round-trip
  - validation
  - loader option wiring
  - shard coverage/disjointness
- [test_experiment_config.py](C:/Users/josep/Desktop/OutreachLM/tests/test_experiment_config.py)
  - train default mapping includes eval-loader script args
- [test_experiment_config_cli.py](C:/Users/josep/Desktop/OutreachLM/tests/test_experiment_config_cli.py)
  - train CLI config path maps eval-loader flags

## Validation
- Targeted:
  - `python -m pytest tests\test_data_loader_config.py tests\test_experiment_config.py tests\test_experiment_config_cli.py`
  - result: `16 passed`
- Full:
  - `python -m pytest`
  - result: `122 passed`
