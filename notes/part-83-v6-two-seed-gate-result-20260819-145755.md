# OutreachLM — Part 83: V6 Two-Seed Gate Result (20260819-145755)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Scope
Completed V6 seed-2 and ran the frozen two-seed leader gate using best-rollout checkpoints for both seeds.

## Commands used
```powershell
python -m outreachlm.train_v4 `
  --output-dir experiments/v6-training-rollout-calibration-seed1337 `
  --seed 1337 `
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

python -m outreachlm.leader_gating_protocol `
  --leader-artifact experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate seed1=experiments/v6-training-rollout-calibration-seed1/v4-best-rollout.pt `
  --candidate seed1337=experiments/v6-training-rollout-calibration-seed1337/v4-best-rollout.pt `
  --min-seeds 2 `
  --required-free-match-margin 0.0 `
  --minimum-free-match 0.2 `
  --require-not-below-leader `
  --require-divergence-or-recovery `
  --minimum-post-divergence-next12-delta 0.0 `
  --output-dir experiments
```

## Artifacts (full data)
- Gate report (full JSON): [leader-gate-20260819-145755.json](C:/Users/josep/Desktop/OutreachLM/experiments/leader-gate-20260819-145755.json)
- Gate report (text): [leader-gate-20260819-145755.txt](C:/Users/josep/Desktop/OutreachLM/experiments/leader-gate-20260819-145755.txt)
- Seed-1 training summary: [v4_training_summary.json](C:/Users/josep/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1/v4_training_summary.json)
- Seed-1337 training summary: [v4_training_summary.json](C:/Users/josep/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1337/v4_training_summary.json)
- Consolidated report (this run): [v6-rollout-calibration-two-seed-gate-20260819-145755.txt](C:/Users/josep/Desktop/OutreachLM/experiments/v6-rollout-calibration-two-seed-gate-20260819-145755.txt)
- Consolidated report JSON: [v6-rollout-calibration-two-seed-gate-20260819-145755.json](C:/Users/josep/Desktop/OutreachLM/experiments/v6-rollout-calibration-two-seed-gate-20260819-145755.json)

## Key result
- `promotion_pass = False`

Per-seed gate snapshot:
- seed1: free-match `0.2250`, heldout `0.140625`, first_div `41`, post_div_next12 `0.166667`, divergence_or_recovery `False`
- seed1337: free-match `0.2250`, heldout `0.143750`, first_div `41`, post_div_next12 `0.250000`, divergence_or_recovery `False`

Aggregate:
- candidate_mean_free_match: `0.225000`
- candidate_mean_heldout_free_match: `0.142188`
- candidate_mean_post_div_next12: `0.208333`
- leader post_div_next12: `0.3333`

## Interpretation
V6 rollout-calibration loss improved free-match across both seeds, but failed the frozen gate condition requiring divergence improvement or post-divergence recovery improvement relative to the leader baseline.