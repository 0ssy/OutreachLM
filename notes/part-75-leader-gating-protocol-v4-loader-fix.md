# OutreachLM — Part 75: Leader Gating Protocol V4 Loader Fix

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Fix the new leader-gating protocol so it can evaluate mixed candidate sets that include:
- classic OutreachLM artifacts (V2-style),
- V4 artifacts (`model_type=outreachlm_v4`).

## Issue observed
Running:

```powershell
python -m outreachlm.leader_gating_protocol `
  --leader-artifact experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate seed1=experiments/v4-training-post-error-w1/v4-best-rollout.pt `
  --candidate seed1337=experiments/v2-test3-boundary-consistency-20260818-093816.pt `
  --min-seeds 2 `
  --required-free-match-margin 0.0 `
  --output-dir experiments
```

failed with `state_dict` key mismatch because V4 checkpoints use a different model module layout than V2 checkpoints.

## Code fix
- Updated [leader_gating_protocol.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/leader_gating_protocol.py)
  - added artifact-type-aware loader:
    - if `model_config.model_type == "outreachlm_v4"`, load via [v4_generate.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/v4_generate.py),
    - otherwise load via existing V2 artifact loader.

This mirrors the compatibility approach already used by [evaluate_suite_compare.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/evaluate_suite_compare.py).

## Validation
Smoke validation with mixed candidates now succeeds:

```powershell
python -m outreachlm.leader_gating_protocol `
  --leader-artifact experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate seed1=experiments/v4-training-post-error-w1/v4-best-rollout.pt `
  --candidate seed1337=experiments/v2-test3-boundary-consistency-20260818-093816.pt `
  --min-seeds 2 `
  --required-free-match-margin 0.0 `
  --heldout-slices 2 `
  --systematic-sample-count 64 `
  --systematic-batch-size 64 `
  --output-dir experiments `
  --report-prefix leader-gate-smoke2
```

Generated:
- [leader-gate-smoke2-20260818-113803.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-smoke2-20260818-113803.json)
- [leader-gate-smoke2-20260818-113803.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-smoke2-20260818-113803.txt)

## Next run
You can now rerun your full command (without smoke reductions) and it should proceed past model loading.