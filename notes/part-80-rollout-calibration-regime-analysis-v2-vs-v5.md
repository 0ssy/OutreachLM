# OutreachLM — Part 80: Rollout-Calibration Regime Analysis (V2 leader vs failed V5 seeds)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Analyze mechanism differences without new training:
- position `40->41`, `41->42`, and `42..52`,
- confidence/margins/entropy,
- hidden/logit movement,
- gold-token and fallback-token behavior,
- post-error recovery and cross-seed stability.

## New analysis script
- [rollout_calibration_regime_analysis.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/rollout_calibration_regime_analysis.py)

It compares one leader plus multiple candidates and exports:
- suite metrics,
- position-level diagnostics (`40..52`),
- fallback-token mass under context-drift,
- post-divergence recovery summary,
- held-out stability and cross-seed stability stats.

## Run used
```powershell
python -m outreachlm.rollout_calibration_regime_analysis `
  --leader leader=experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate v5_seed1=experiments/v5-boundary-rollout-intervention-20260818-123304.pt `
  --candidate v5_seed2=experiments/v5-boundary-rollout-intervention-20260818-125542.pt `
  --output-dir experiments
```

## Artifacts
- [rollout-calibration-analysis-20260818-175332.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/rollout-calibration-analysis-20260818-175332.json)
- [rollout-calibration-analysis-20260818-175332.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/rollout-calibration-analysis-20260818-175332.txt)

## Key findings
1. **First divergence remains fixed at 41** in leader and failed V5 seeds.
2. Both failed V5 seeds show **lower free-match** than leader despite comparable teacher accuracy:
   - leader `0.2000`
   - V5 seeds `0.1000`, `0.1875`
3. Around `41..43`, failed V5 seeds show:
   - lower free gold probability (negative deltas),
   - more negative margin-shift deltas,
   - higher free entropy,
   - larger fallback-token probability mass under context drift.
4. Cross-seed candidate stability remains weak:
   - free_match mean/std/min/max = `0.14375 / 0.04375 / 0.1000 / 0.1875`
   - held-out mean free = `0.115625` (below leader held-out mean from gate runs).

## Interpretation
V2 w=2.0 preserves a rollout-calibration property that failed V5 seeds do not: under post-boundary context drift, V2 maintains better gold-token usability with less harmful fallback drift, even when teacher-side metrics are not dominant.