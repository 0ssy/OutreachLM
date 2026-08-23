# OutreachLM — Divergence Recovery Weight 2.0

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Keep the same V2 divergence-window intervention and change only:
- `recovery_loss_weight: 1.0 -> 2.0`

Everything else unchanged:
- same resume checkpoint (V2@4500),
- same architecture/tokenizer/dataset/context,
- same balanced CE + label smoothing setup,
- same evaluation suite.

## Artifacts
- [v2-divergence-intervention-20260815-135123.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-135123.json)
- [v2-divergence-intervention-20260815-135123.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-135123.txt)
- [v2-divergence-intervention-20260815-135123.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260815-135123.pt)

## Final metrics
### Weight 1.0 (previous run)
- teacher_top1: `0.4750`
- free_match: `0.1625`
- prompt_logit_cosine: `0.9241`
- rollout_mean_entropy: `3.6348`
- first_repeated_bigram_step: `22`
- first_repeated_trigram_step: `37`
- first_free_divergence: `41`

### Weight 2.0 (current run)
- teacher_top1: `0.4375`
- free_match: `0.2000`
- prompt_logit_cosine: `0.9337`
- rollout_mean_entropy: `3.8536`
- first_repeated_bigram_step: `22`
- first_repeated_trigram_step: `36`
- first_free_divergence: `41`

## Delta (2.0 - 1.0)
- teacher_top1: `-0.0375`
- free_match: `+0.0375`
- prompt_logit_cosine: `+0.0096` (slightly worse alignment)
- rollout_mean_entropy: `+0.2189`
- first_repeated_bigram_step: `0`
- first_repeated_trigram_step: `-1`
- first_free_divergence: `0`

## Interpretation
- Increasing recovery weight to 2.0 improved the key target metric again (`free_match 0.1625 -> 0.2000`).
- Tradeoff: teacher-forced accuracy declined and logit alignment became slightly worse.
- Divergence onset remains fixed at position 41, but post-divergence recovery continues to improve.