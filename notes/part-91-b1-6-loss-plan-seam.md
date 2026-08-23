# OutreachLM — Part 91: B1.6 LossPlan Seam

## Scope
Implemented a composable loss seam without migrating [train_v4.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train_v4.py) yet.

## What it is
`LossPlan` is the objective-composition seam for training. It standardizes how independent loss terms are combined into one `total_loss` while preserving per-term diagnostics.

## Added
- [loss_plans.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/loss_plans.py)
  - `LossTerm`
  - `TeacherLossTerm`
  - `RecoveryLossTerm`
  - `PostErrorLossTerm`
  - `RolloutCalibrationLossTerm`
  - `LossPlan`
  - `LossPlanResult` with:
    - `total_loss`
    - `term_losses`
    - `weighted_term_losses`

## Design constraints preserved
- No architecture-specific logic inside `LossPlan`.
- No trainer logic inside loss terms.
- Weighted composition follows current V4-style objective math.
- Optional/disabled terms supported without changing trainer core.
- Per-term diagnostics preserved for `teacher`, `recovery`, `post_error`, `rollout_calib`.

## Why it is there
Before this seam, objective logic lived inline in trainer code, which couples loop mechanics to experiment math. `LossPlan` separates:
- **how** training runs (`TrainerCore`)
- **what** training optimizes (`LossPlan`)

This is required for B1 migration safety and for future controlled experiments.

## Why it is important
- Prevents objective logic from being duplicated across trainers/scripts.
- Makes intervention terms composable and testable without rewriting loop code.
- Preserves term-level observability needed for debugging and gating.
- Provides a stable interface before B1.10 trainer migrations.

## What would happen without it
- Loss behavior would stay entangled inside trainer scripts.
- Small experiment changes would require editing core loop code repeatedly.
- Regression risk would increase because objective math would be harder to isolate/test.
- Later distributed/runtime refactors (Phase C) would carry hidden objective coupling and become riskier.

## Added tests
- [test_loss_plans.py](C:/Users/josep/Desktop/OutreachLM/tests/test_loss_plans.py)
  - teacher-only plan
  - weighted combination correctness
  - disabled terms omitted
  - multi-term composition
  - diagnostics contain per-term losses
  - zero-weight term contributes zero
  - gradient flow through composed loss
  - rollout-calibration term pluggability
  - regression-style V4 weighted objective math with representative logits/targets tensors

## Validation
Full suite run:
- `python -m pytest`
- result: `55 passed`

## B1 progression
- B1.1 ModelArtifact ✅
- B1.2 ModelRegistry ✅
- B1.3 ModelConfig ✅
- B1.4 TrainingConfig ✅
- B1.5 TrainerCore ✅
- B1.6 LossPlan ✅
- next: B1.7 EvaluationProfile
