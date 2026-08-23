# OutreachLM — w=2.0 Continuation Stability Pass

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Build directly on the best current intervention (recovery weight `2.0`) and test whether additional training can preserve/improve free-running recovery while regaining teacher-forced quality.

## Setup
- Resume from: [v2-divergence-intervention-20260815-135123.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-135123.pt)
- Same recipe:
  - balanced CE + label smoothing `0.05`
  - divergence-window recovery start index `40`
  - recovery loss weight `2.0`
  - batch `8`, LR `0.0005`, warmup `250`, min LR ratio `0.1`
- Additional steps: `1500`

## Artifacts
- [v2-divergence-intervention-20260816-113809.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.json)
- [v2-divergence-intervention-20260816-113809.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.txt)
- [v2-divergence-intervention-20260816-113809.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.pt)

## Metrics (before vs after continuation)
| Metric | Before (w=2.0) | After +1500 |
|---|---:|---:|
| teacher_top1 | 0.4375 | **0.4625** |
| free_match | 0.2000 | **0.2000** |
| prompt_logit_cosine | 0.9337 | **0.9191** |
| rollout_mean_entropy | 3.8536 | 3.8613 |
| first_repeated_bigram_step | 22 | 22 |
| first_repeated_trigram_step | 36 | **39** |
| first_free_divergence | 41 | 41 |

## Interpretation
This continuation preserves the best free-running match (`0.2000`) while recovering some teacher-forced quality and improving logit separation/repetition delay further.  
Divergence onset remains fixed at `41`, but post-divergence behavior remains improved and slightly more stable.