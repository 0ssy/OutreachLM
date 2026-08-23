# OutreachLM — Self-Describing Artifacts and Checkpoint Integrity

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Why this step
The model architecture audit showed that training/generation correctness depended on shared constants and implicit assumptions.

This step hardens reproducibility by making checkpoints and model artifacts carry explicit configuration identity.

## What changed

### 1) Checkpoint config now captures architecture identity
Updated [checkpoint.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/checkpoint.py):

- `CHECKPOINT_VERSION` bumped to `2`
- `build_config(...)` now includes:
  - `num_layers`
  - `num_heads`
  - `vocab_size`
  - existing training/scheduler/data fields

This means resume compatibility now includes architecture and vocabulary shape, not only optimizer/step state.

### 2) Train loop now validates resume compatibility
Updated [train.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py):

- Added model config constants:
  - `NUM_LAYERS = 1`
  - `NUM_HEADS = 4`
- Extended model creation path to use `num_layers` / `num_heads`
- Built `run_config` via `build_config(...)` using **actual runtime arguments**
- On `--resume`, loads checkpoint state and runs:
  - `validate_config(checkpoint_state["config"], run_config)`
- Resume now fails fast on incompatible run configs instead of silently continuing.

### 3) Model artifact is now self-describing
`outreachlm_model.pt` save format in [train.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py) now includes:

- `model_state_dict`
- `model_config`:
  - vocab size
  - context length
  - embedding dim
  - num layers
  - num heads
- `training_config`
- `tokenizer_config`:
  - token list
  - pad token
  - unk token

This allows generation to reconstruct the exact model/tokenizer contract from the artifact.

### 4) Generation now prefers artifact metadata
Updated [generate.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/generate.py):

- Loads model/tokenizer from self-describing artifact when metadata is present
- Uses artifact `context_length` via `model.context_length` during generation
- Keeps a legacy fallback path for old state-dict-only model files

## Compatibility updates
Updated CLI/train wiring for architecture parameters:
- `--num-layers`
- `--num-heads`

These are now part of the run and checkpoint identity.

## Result
OutreachLM now has a stronger reproducibility contract:

- checkpoint resume is config-validated
- model artifact declares architecture/tokenizer metadata
- generation can consume that metadata directly

This is the right base before scaling model capacity.