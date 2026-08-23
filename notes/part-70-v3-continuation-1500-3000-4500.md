# OutreachLM — V3 Continuation (1500 -> 4500) and Leader Comparison

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Continue V3 training from step 1500 to step 4500 with **no recipe changes**, then compare:
- V3 @1500
- V3 @3000
- V3 @4500
against the current V2 leader using the same evaluation suite.

## Fixed training recipe
- architecture: V3 (`context=256`, `emb=256`, `layers=4`, `heads=8`, vocab=490)
- objective: balanced CE + label smoothing `0.05`
- recovery loss weight: `2.0`
- batch size: `8`
- learning rate: `0.0005`
- warmup steps: `250`
- min LR ratio: `0.1`

## Scripts and artifacts
Continuation/comparison script:
- [v3_continue_to_4500_compare.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/v3_continue_to_4500_compare.py)

Comparison outputs:
- [v3-continue-and-compare.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-continue-20260816-131317/v3-continue-and-compare.json)
- [v3-continue-and-compare.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-continue-20260816-131317/v3-continue-and-compare.txt)

Saved V3 checkpoints:
- [v3-step-3000.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-continue-20260816-131317/v3-step-3000.pt)
- [v3-step-4500.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-continue-20260816-131317/v3-step-4500.pt)

## Metrics
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeat_bigram | first_repeat_trigram | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 leader | 0.4625 | **0.2000** | 0.9191 | 3.8613 | 22 | 39 | 41 |
| V3 @1500 | 0.2625 | 0.1875 | 0.9531 | 3.8824 | 19 | 20 | 41 |
| V3 @3000 | 0.3750 | 0.1875 | 0.8643 | 3.7783 | 19 | 20 | 41 |
| V3 @4500 | **0.4875** | 0.1500 | 0.8146 | 3.4192 | 24 | 35 | 41 |

## Readout
- V3 teacher accuracy recovered strongly and surpassed the V2 leader by step 4500.
- Prompt-logit cosine improved substantially with training.
- However, free-running match degraded at step 4500 (`0.1500`), staying below the leader (`0.2000`).

So under this fixed recipe, V3 is improving language-model fit but not yet matching the leader on the rollout objective.