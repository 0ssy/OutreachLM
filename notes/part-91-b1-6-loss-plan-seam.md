# OutreachLM — Part 91: B1.6 LossPlan Seam

## Scope
Implemented a composable loss seam without migrating [train_v4.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train_v4.py) yet.

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
