# OutreachLM — Part 73: V4 Rollout-Aware Selection Rerun

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Run one controlled V4 rerun with **architecture and recipe unchanged**, and change only training control:
- evaluate every 250 steps,
- select/save best checkpoint by validation `free_match`,
- conservative early-stop on sustained rollout degradation.

## Code updates
- [train_v4.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train_v4.py)
  - added rollout-aware eval/selection policy,
  - added `v4-best-rollout.pt`,
  - final artifact now comes from best rollout checkpoint.
- [v4-architecture.md](C:/Users/josep/OneDrive/Desktop/OutreachLM/docs/v4-architecture.md)
  - updated training-policy documentation.
- [evaluate_suite_compare.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/evaluate_suite_compare.py)
  - now supports V4 artifact loading for standard suite comparison.

## Rerun artifacts
- [v4_config.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v4-training/v4_config.json)
- [v4_training_summary.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v4-training/v4_training_summary.json)
- [v4-best-rollout.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v4-training/v4-best-rollout.pt)
- [v4-final.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v4-training/v4-final.pt)

Comparison output:
- [eval-compare-20260816-201446.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v4-eval-compare/eval-compare-20260816-201446.txt)

## Rollout trajectory (eval interval = 250)
- step 250: free_match `0.1000`
- step 500: free_match `0.1250`
- step 750: free_match `0.1625`
- step 1000: free_match `0.1500`
- step 1250: free_match `0.1875`
- step 1500: free_match `0.2000` (**best**)
- step 1750: free_match `0.1875`
- step 2000: free_match `0.1875`
- step 2250: free_match `0.1375`
- step 2500: free_match `0.1750`

Early stop triggered at step 2500 after 4 degraded evals since best at step 1500.

## Final selected checkpoint
From [v4_training_summary.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v4-training/v4_training_summary.json):
- selected step: `1500`
- selected free_match: `0.2000`
- selected teacher_top1: `0.3750`
- first divergence: `41`
- first repeated trigram step: `20`

## Leader comparison
From [eval-compare-20260816-201446.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v4-eval-compare/eval-compare-20260816-201446.txt):

- V2 leader: teacher_top1 `0.4625`, free_match `0.2000`, first_repeat_tri `39`, divergence `41`
- V4 rollout-selected final: teacher_top1 `0.3750`, free_match `0.2000`, first_repeat_tri `20`, divergence `41`

## Interpretation
Rollout-aware checkpoint selection recovered the useful V4 regime (step 1500) and prevented late-training collapse from being exported as final.  
V4 now ties the best free-match (`0.2000`) but still does not beat the V2 leader overall.