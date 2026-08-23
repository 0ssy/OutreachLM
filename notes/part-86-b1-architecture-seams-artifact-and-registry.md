# OutreachLM — Part 86: B1 Architecture Seams (Artifact + Model Registry)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Scope
Continue Phase B1 refactor logging from Part 85 with no training behavior changes.

This part records:
- B1.1 artifact seam completion,
- B1.2 model registry seam completion,
- constructor snapshot for legacy and V4 architectures.

## B1.1 completed
Added canonical artifact seam module:
- [model_artifacts.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/model_artifacts.py)

Added tests:
- [test_model_artifacts.py](C:/Users/josep/Desktop/OutreachLM/tests/test_model_artifacts.py)

Contract fields:
- `artifact_version`
- `model_type`
- `model_config`
- `tokenizer_config`
- `training_config`
- `state_dict`

Status:
- targeted tests passed (`2 passed`)
- committed: `32920fa` (`refactor: establish model artifact contract`)

## B1.2 completed
Added model registry seam:
- [model_registry.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/model_registry.py)

Added tests:
- [test_model_registry.py](C:/Users/josep/Desktop/OutreachLM/tests/test_model_registry.py)

Registry interface:
- `register_model(model_type, factory)`
- `create_model(model_type, model_config)`

Registered architectures:
- `legacy_v1` -> [OutreachModel](C:/Users/josep/Desktop/OutreachLM/outreachlm/model.py)
- `v4` -> [OutreachV4Model](C:/Users/josep/Desktop/OutreachLM/outreachlm/v4_model.py)

Compatibility checks added in tests:
- V4 direct constructor vs registry constructor parameter names/shapes/count match
- legacy direct constructor vs registry constructor parameter names/shapes/count match
- unknown model type raises clear error

Status:
- full pytest run passed (`7 passed`)
- committed: `3f3201e` (`refactor: add model registry seam`)

## Constructor snapshot (as requested)
Legacy constructor in [model.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/model.py):
- `OutreachModel(vocab_size, context_length, embedding_dim, num_layers=1, num_heads=4)`

V4 constructor in [v4_model.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/v4_model.py):
- `OutreachV4Model(vocab_size, context_length=256, embedding_dim=256, num_layers=4, num_heads=8, ffn_dim=684)`

Key architecture difference preserved:
- legacy uses explicit positional embedding + explicit output head,
- V4 uses RoPE within attention + RMSNorm + tied output projection.

## Phase status
- V2 remains frozen production leader.
- Refactor is code-architecture-only (no model training behavior changes).
- B1 progression now at: `B1.2 complete`, next is `B1.3 ModelConfig`.