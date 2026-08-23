# OutreachLM — V2 Continuation (Additional 3000 Steps)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Continue the existing V2 pilot checkpoint for 3000 additional steps with **no recipe changes**:
- same architecture (ctx 128, emb 128, layers 2, heads 4),
- same objective family (balanced CE + label smoothing 0.05),
- same batch size/LR/warmup/min-LR-ratio,
- same evaluation suite.

## Script and artifacts
Script:
- [architecture_capacity_continuation.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/architecture_capacity_continuation.py)

Raw outputs:
- [architecture-v2-continuation-20260815-133859.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/architecture-v2-continuation-20260815-133859.json)
- [architecture-v2-continuation-20260815-133859.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/architecture-v2-continuation-20260815-133859.txt)
- Continued checkpoint:
  - [architecture-v2-continuation-20260815-133859.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/architecture-v2-continuation-20260815-133859.pt)

## Metric comparison
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeated_bigram_step | first_repeated_trigram_step | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5125 | 0.1000 | 0.9944 | 1.7168 | 19 | 20 | 41 |
| V2 before continuation (1500-step pilot) | 0.2875 | 0.1125 | 0.9534 | 3.3459 | 19 | 20 | 41 |
| V2 after +3000 continuation | 0.4500 | 0.1000 | 0.9177 | 3.0624 | 25 | 29 | 41 |

## Training progression during continuation
- loss first/last/mean: `4.1674 / 3.6576 / 3.8787`
- checkpoint snapshots:
  - step 750: teacher `0.3875`, free `0.1000`, cosine `0.9536`, trigram step `20`
  - step 1500: teacher `0.3750`, free `0.0875`, cosine `0.9315`, trigram step `29`
  - step 2250: teacher `0.4000`, free `0.0625`, cosine `0.9204`, trigram step `29`
  - step 3000: teacher `0.4500`, free `0.1000`, cosine `0.9177`, trigram step `29`

## Interpretation
- V2 continued learning materially (teacher top-1 recovered from `0.2875` to `0.4500`).
- Output alignment improved further (`0.9534 -> 0.9177`).
- Repetition onset moved later (`20 -> 29` trigram).
- But free-running target match remained at baseline-level (`0.1000`), and first divergence stayed fixed at `41`.

This continuation supports the view that V2 was undertrained at 1500, but also indicates that simply adding training time/capacity (within this setup) is still insufficient to resolve the core rollout instability.