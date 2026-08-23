# OutreachLM — Divergence Recovery Weight 3.0

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Single controlled run with only one change:
- `recovery_loss_weight: 2.0 -> 3.0`

All else held constant (same V2@4500 base, same objective, same optimizer/schedule, same evaluation).

## Artifacts
- [v2-divergence-intervention-20260815-135834.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-135834.json)
- [v2-divergence-intervention-20260815-135834.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-135834.txt)
- [v2-divergence-intervention-20260815-135834.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-135834.pt)

## Final metrics
| Weight | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeat_bigram | first_repeat_trigram | first_free_divergence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.4750 | 0.1625 | 0.9241 | 3.6348 | 22 | 37 | 41 |
| 2.0 | 0.4375 | 0.2000 | 0.9337 | 3.8536 | 22 | 36 | 41 |
| 3.0 | 0.4250 | 0.2000 | 0.9411 | 3.8613 | 25 | 36 | 41 |

## Interpretation
- Moving from `2.0` to `3.0` did **not** improve free-match (`0.2000` flat).
- Teacher accuracy decreased further (`0.4375 -> 0.4250`).
- Prompt-logit cosine worsened (`0.9337 -> 0.9411`).

This suggests the useful recovery-weight range has likely been reached around `2.0` for the current setup.