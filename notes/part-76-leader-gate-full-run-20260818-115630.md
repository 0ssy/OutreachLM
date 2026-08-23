# OutreachLM — Part 76: Leader Gate Full Run (20260818-115630)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Run the full leader-gating protocol on the current baseline and two candidate seeds:
- `seed1`: V4 post-error best-rollout checkpoint,
- `seed1337`: reproduction seed checkpoint.

## Command
```powershell
python -m outreachlm.leader_gating_protocol `
  --leader-artifact experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate seed1=experiments/v4-training-post-error-w1/v4-best-rollout.pt `
  --candidate seed1337=experiments/v2-test3-boundary-consistency-20260818-093816.pt `
  --min-seeds 2 `
  --required-free-match-margin 0.0 `
  --output-dir experiments
```

## Artifacts
- [leader-gate-20260818-115630.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-20260818-115630.json)
- [leader-gate-20260818-115630.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-20260818-115630.txt)

## Gate summary
Leader:
- teacher_top1: `0.4625`
- free_match: `0.2000`
- heldout_mean_free: `0.1344`

Candidates:
- `seed1`: teacher `0.2000`, free_match `0.2250`, heldout `0.1375`, seed_pass=`True`, heldout_pass=`True`
- `seed1337`: teacher `0.4750`, free_match `0.1875`, heldout `0.13125`, seed_pass=`False`, heldout_pass=`False`

Aggregate:
- candidate mean free_match: `0.20625`
- candidate mean heldout free_match: `0.134375`

## Decision
- `promotion_pass = False`
- reason: not all seeds beat leader free-match threshold (`seed1337` failed).

## Interpretation
The gating protocol confirms the earlier conclusion:
- single-seed `0.225` is promising,
- but the intervention is **not robust across seeds yet**,
- baseline leader remains **V2 w=2.0**.