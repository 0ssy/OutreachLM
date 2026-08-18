# OutreachLM — Part 82: V6 Rollout Calibration (Seed 1 only)

## Goal
Implement V6 with one controlled change:
- add rollout-distribution-preservation loss around boundary `41..43`,
- keep V4 architecture and core training setup unchanged.

No second seed run in this step (per protocol).

## Code change
- Updated [train_v4.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train_v4.py)
  - added optional rollout-calibration loss on output distributions under self-conditioned context drift,
  - no hidden-state matching term added,
  - default behavior unchanged unless new weight flag is enabled.

New flags:
- `--rollout-calibration-loss-weight`
- `--rollout-calibration-forced-error-index`
- `--rollout-calibration-rollout-steps`
- `--rollout-calibration-start-index`
- `--rollout-calibration-end-index`

## Seed-1 run command
```powershell
python -m outreachlm.train_v4 `
  --output-dir experiments/v6-training-rollout-calibration-seed1 `
  --seed 1 `
  --steps 1500 `
  --eval-interval 250 `
  --checkpoint-interval 500 `
  --batch-size 8 `
  --learning-rate 0.0005 `
  --warmup-steps 250 `
  --min-learning-rate-ratio 0.1 `
  --label-smoothing 0.05 `
  --recovery-start-index 40 `
  --recovery-loss-weight 2.0 `
  --post-error-loss-weight 0.0 `
  --rollout-calibration-loss-weight 1.0 `
  --rollout-calibration-forced-error-index 40 `
  --rollout-calibration-rollout-steps 8 `
  --rollout-calibration-start-index 41 `
  --rollout-calibration-end-index 43
```

## Artifacts
- [v4_config.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1/v4_config.json)
- [v4_training_summary.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1/v4_training_summary.json)
- [v4-best-rollout.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1/v4-best-rollout.pt)
- [v4-final.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1/v4-final.pt)

## Result
From [v4_training_summary.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1/v4_training_summary.json):
- selected best step: `250`
- selected free-match: `0.2250`
- selected teacher_top1: `0.2250`
- first divergence: `41`
- stopped early after sustained degradation from best.

## Gate precondition check
Frozen rule for spending seed-2 was: seed-1 must beat leader free-match `0.2000`.

Seed-1 achieved `0.2250` (`> 0.2000`), so it **passes the precondition** to run seed-2 next.
