# OutreachLM — Part 93: B1.8 Tokenizer Artifact Seam

## What it is
`TokenizerArtifact` is a typed, serializable seam that records tokenizer identity and structure (tokens, special tokens, vocab size) as an explicit artifact.

## Why it is there
Tokenizer assumptions were previously implicit (global paths and fixed vocabulary expectations). This seam makes tokenizer provenance explicit and loadable/savable in a controlled way.

## Why it is important
- Removes hidden dependency on fixed vocab assumptions.
- Enables reliable model/tokenizer compatibility checks.
- Supports future artifact-contract integration without redesigning tokenizer internals.
- Improves reproducibility by treating tokenizer metadata as first-class serialized state.

## What would happen without it
- Tokenizer configuration drift could silently break training/evaluation compatibility.
- Hard-coded expectations (like fixed vocab sizes) would remain scattered.
- Future migration of training/evaluation to artifact-driven configuration would be fragile.

## Scope
Implemented B1.8 seam only. No migration of train/evaluation implementations yet.

## Added
- [tokenizer_artifacts.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/tokenizer_artifacts.py)
  - `TokenizerArtifact` dataclass
  - validation for required fields and token integrity
  - derived `vocab_size`
  - `to_dict()` / `from_dict(...)`
  - `from_tokenizer(...)` / `to_tokenizer(...)` compatibility bridge
  - `save_tokenizer_artifact(...)` / `load_tokenizer_artifact(...)`

## Added tests
- [test_tokenizer_artifacts.py](C:/Users/josep/Desktop/OutreachLM/tests/test_tokenizer_artifacts.py)
  - artifact construction
  - vocab-size derivation
  - special-token configuration
  - save/load round trip
  - missing required fields
  - invalid vocabulary
  - duplicate tokens
  - serialization/deserialization
  - compatibility with current tokenizer representation
  - serialized vocab-size mismatch detection

## Validation
Full suite run:
- `python -m pytest`
- result: `85 passed`

## B1 progression
- B1.1 ModelArtifact ✅
- B1.2 ModelRegistry ✅
- B1.3 ModelConfig ✅
- B1.4 TrainingConfig ✅
- B1.5 TrainerCore ✅
- B1.6 LossPlan ✅
- B1.7 EvaluationProfile ✅
- B1.8 TokenizerArtifact ✅
- next: B1.9 Artifact contract integration
