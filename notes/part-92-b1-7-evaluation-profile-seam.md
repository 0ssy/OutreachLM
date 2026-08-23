# OutreachLM — Part 92: B1.7 EvaluationProfile Seam

## What it is
`EvaluationProfile` is a typed configuration seam that centralizes shared evaluation parameters used across boundary/rollout analysis and gating workflows.

## Why it is there
Evaluation settings were previously scattered and hard-coded across multiple scripts. This seam introduces one explicit place to represent those parameters without modifying existing evaluation scripts yet.

## Why it is important
- Standardizes evaluation defaults and range semantics.
- Reduces duplicated assumptions (for example boundary windows and sample settings).
- Makes future migration of evaluation scripts safer and incremental.
- Improves reproducibility by giving a serializable, validated profile object.

## What would happen without it
- Hard-coded evaluation knobs would continue drifting across scripts.
- Subtle default mismatches could invalidate model comparisons.
- Refactors toward shared evaluator infrastructure would remain high-risk and noisy.

## Scope
Implemented B1.7 seam only (no migration of existing evaluation scripts yet).

## Added
- [evaluation_profiles.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/evaluation_profiles.py)
  - `EvaluationProfile` dataclass
  - validation rules for ranges/counts/seeds/windows
  - `to_dict()` / `from_dict(...)`

Default profile values aligned to current scripts:
- prompt/eval lengths: `40`, `80`
- boundary window: `40..52`
- sample settings: `seed=42`, `count=4096`, `batch=256`
- fallback top-k: `5`
- heldout slices: `4`
- hidden/output test windows: `38..45` and `39..43`
- output top-k: `5`

## Added tests
- [test_evaluation_profiles.py](C:/Users/josep/Desktop/OutreachLM/tests/test_evaluation_profiles.py)
  - defaults
  - validation failures (invalid ranges/counts/seeds)
  - serialization round-trip (`to_dict` -> `from_dict`)

## Validation
Full suite run:
- `python -m pytest`
- result: `74 passed`

## B1 progression
- B1.1 ModelArtifact ✅
- B1.2 ModelRegistry ✅
- B1.3 ModelConfig ✅
- B1.4 TrainingConfig ✅
- B1.5 TrainerCore ✅
- B1.6 LossPlan ✅
- B1.7 EvaluationProfile ✅
- next: B1.8 Tokenizer artifact seam
