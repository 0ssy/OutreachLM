# OutreachLM — V2 Recovery Weight 2.5 (from Current Leader)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Single controlled run from the current leader checkpoint, changing only:
- `recovery_loss_weight: 2.0 -> 2.5`

Everything else fixed:
- architecture unchanged (V2),
- same tokenizer/data/context,
- same balanced CE + label smoothing `0.05`,
- same recovery window start index `40`,
- same optimizer/schedule and evaluation suite.

## Artifacts
- [v2-divergence-intervention-20260816-142617.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-142617.json)
- [v2-divergence-intervention-20260816-142617.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-142617.txt)
- [v2-divergence-intervention-20260816-142617.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-142617.pt)

## Result
From current leader -> after weight 2.5:
- teacher_top1: `0.4625 -> 0.4875`
- free_match: `0.2000 -> 0.1000`
- prompt_logit_cosine: `0.9191 -> 0.9085`
- rollout_mean_entropy: `3.8613 -> 3.8044`
- first_repeated_trigram_step: `39 -> 36`
- first_free_divergence: `41 -> 41`

## Checkpoint trajectory (during run)
- step 375: free_match `0.1875`
- step 750: free_match `0.1750`
- step 1125: free_match `0.2000`
- step 1500: free_match `0.1000`

## Interpretation
Weight `2.5` does not improve the leader and is unstable for rollout objective under this schedule; final free-running match collapses back to baseline-level performance.  
Current leader (`recovery weight 2.0`) remains best for autonomous behavior.