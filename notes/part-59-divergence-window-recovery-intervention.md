# OutreachLM — Divergence-Window Recovery Intervention (V2)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Target the teacher/free mismatch at the known divergence area (around position 41) without changing architecture, tokenizer, or base objective family.

## Intervention
Starting from V2@4500 checkpoint:
- keep balanced CE + label smoothing (`0.05`)
- add one extra **recovery loss** term computed on model-perturbed histories:
  - for each batch, build mixed input by replacing tokens from index `40` onward with model-predicted tokens from teacher logits
  - compute CE on the mixed-history forward pass for target positions from index `40` onward
  - total loss: `teacher_loss + 1.0 * recovery_loss`

## Script and artifacts
Script:
- [divergence_window_intervention.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/divergence_window_intervention.py)

Raw outputs:
- [v2-divergence-intervention-20260815-134536.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-134536.json)
- [v2-divergence-intervention-20260815-134536.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-134536.txt)
- Checkpoint:
  - [v2-divergence-intervention-20260815-134536.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-134536.pt)

## Metric comparison
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeated_bigram_step | first_repeated_trigram_step | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline V1 | 0.5125 | 0.1000 | 0.9944 | 1.7168 | 19 | 20 | 41 |
| V2 before intervention | 0.4500 | 0.1000 | 0.9177 | 3.0624 | 25 | 29 | 41 |
| V2 after intervention | 0.4750 | 0.1625 | 0.9241 | 3.6348 | 22 | 37 | 41 |

## Training progression
Checkpoint snapshots:
- step 375: teacher `0.4375`, free `0.1500`, cosine `0.9430`, trigram step `20`
- step 750: teacher `0.4500`, free `0.1250`, cosine `0.9330`, trigram step `38`
- step 1125: teacher `0.4875`, free `0.1750`, cosine `0.9241`, trigram step `38`
- step 1500: teacher `0.4750`, free `0.1625`, cosine `0.9241`, trigram step `37`

Loss behavior:
- total loss first/last/mean: `9.4062 / 8.3652 / 8.3695`
- teacher loss first->last: `3.7731 -> 3.7778` (flat)
- recovery loss first->last: `5.6331 -> 4.5873` (improving)

## Interpretation
- This is the first intervention to materially improve **free-running match** (`0.1000 -> 0.1625`) while keeping architecture fixed.
- It also increases rollout entropy and delays trigram collapse further (`29 -> 37` vs V2 pre-intervention).
- First divergence position remained `41`, so onset is unchanged, but downstream recovery appears improved.

Net: targeted mismatch training shows a meaningful positive signal compared with prior objective-only and scheduled-sampling runs.