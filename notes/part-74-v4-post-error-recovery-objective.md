# OutreachLM — Part 74: V4 Post-Error Recovery Objective (Implementation)

## Goal
Implement a V4 training objective that explicitly trains recovery from the model's own wrong token, without changing architecture.

## Changes
- Updated [train_v4.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train_v4.py)
  - added post-error rollout construction using model-generated tokens after a forced wrong token,
  - added optional post-error recovery loss term,
  - integrated post-error loss into training logs, checkpoints, and training summary.
- Updated [v4-architecture.md](C:/Users/josep/OneDrive/Desktop/OutreachLM/docs/v4-architecture.md)
  - documented optional post-error recovery objective and run command.

## New training flags
- `--post-error-loss-weight` (default `0.0`)
- `--post-error-start-index` (default `40`)
- `--post-error-rollout-steps` (default `8`)
- `--post-error-loss-window` (default `32`; negative value means full suffix)

With defaults unchanged, baseline V4 behavior is preserved.

## Objective behavior
When enabled (`--post-error-loss-weight > 0`):
1. Force a wrong token at the boundary index using top-1 prediction unless it equals gold, then use top-2.
2. Roll out a short model-generated continuation from that perturbed state.
3. Compute CE loss on the continuation window against gold targets.
4. Add this term to total loss:

`total = teacher_loss + recovery_weight * recovery_loss + post_error_weight * post_error_loss`

## Validation
Short end-to-end smoke run succeeded with post-error objective enabled:

```bash
python -m outreachlm.train_v4 --steps 80 --eval-interval 40 --checkpoint-interval 80 --log-interval 20 --post-error-loss-weight 1.0 --post-error-start-index 40 --post-error-rollout-steps 4 --post-error-loss-window 16
```

Key validation signals:
- post-error loss computed and logged each step,
- checkpoint summaries include `post_error_loss`,
- training summary includes `first_post_error_loss` and `last_post_error_loss`,
- rollout-aware final selection still works.
