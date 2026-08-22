# OutreachLM — Part 88: B1.3 ModelConfig

## Scope
Implemented typed model configuration seam for current architectures while preserving existing behavior and registry compatibility.

## Added
- [model_config.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/model_config.py)
  - `LegacyV1Config`
  - `V4Config`
  - validation in `__post_init__`
  - `to_dict()` and `from_dict(...)`

## Updated
- [model_registry.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/model_registry.py)
  - still supports existing dictionary API:
    - `create_model("legacy_v1", config_dict)`
    - `create_model("v4", config_dict)`
  - now also supports typed config input:
    - `create_model(LegacyV1Config(...))`
    - `create_model(V4Config(...))`
  - no architecture behavior changes

## Tests added
- [test_model_config.py](C:/Users/josep/Desktop/OutreachLM/tests/test_model_config.py)
  - defaults for `LegacyV1Config` and `V4Config`
  - validation failures (non-positive dims and v4 divisibility rule)
  - serialization round-trip (`to_dict` -> `from_dict`)
  - registry compatibility (typed config path vs dict path)

## Validation result
Full suite run:
- `python -m pytest`
- `26 passed`

## B1 progression
- B1.1 ModelArtifact seam: complete
- B1.2 ModelRegistry seam: complete
- B1.3 ModelConfig: complete
- next: B1.4 TrainingConfig
