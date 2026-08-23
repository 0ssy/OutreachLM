# OutreachLM — First-Error Prefix Intervention

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Test the proposed first-error intervention directly on model-generated pre-error state:
- keep successful recovery objective (`recovery weight = 2.0`),
- add auxiliary loss on prediction from model-generated prefix immediately before target position ~41.

## Intervention details
Per training batch:
1. standard teacher loss (balanced CE + label smoothing),
2. recovery loss from divergence window (start index 40),
3. **first-error prefix loss**:
   - take true prefix length 40,
   - generate token at position 40 with model argmax,
   - feed `[true prefix + generated token]`,
   - train next prediction against gold token at position 41.

Total loss:
- `teacher + 2.0 * recovery + 1.0 * first_error_prefix`

## Script and artifacts
Script:
- [first_error_prefix_intervention.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/first_error_prefix_intervention.py)

Outputs:
- [v2-first-error-intervention-20260816-115848.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-first-error-intervention-20260816-115848.json)
- [v2-first-error-intervention-20260816-115848.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-first-error-intervention-20260816-115848.txt)
- [v2-first-error-intervention-20260816-115848.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-first-error-intervention-20260816-115848.pt)

## Final metric comparison
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeat_bigram | first_repeat_trigram | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 w=2.0 checkpoint (start) | 0.4625 | **0.2000** | 0.9191 | 3.8613 | 22 | 39 | 41 |
| After first-error intervention | 0.4750 | 0.1500 | 0.9333 | 3.6379 | 22 | 43 | 41 |

## Interpretation
- Teacher-forced accuracy improved slightly.
- Free-running match degraded (`0.2000 -> 0.1500`), so this does **not** beat the current leader.
- First divergence remained fixed at 41.

This branch does not replace the current best checkpoint.