# OutreachLM — Part 78: V5 Two-Seed Gate Result (20260818-173842)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Runs completed
V5 seed runs:
- [v5-boundary-rollout-intervention-20260818-123304.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-123304.json)
- [v5-boundary-rollout-intervention-20260818-123304.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-123304.txt)
- [v5-boundary-rollout-intervention-20260818-123304.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-123304.pt)
- [v5-boundary-rollout-intervention-20260818-125542.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-125542.json)
- [v5-boundary-rollout-intervention-20260818-125542.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-125542.txt)
- [v5-boundary-rollout-intervention-20260818-125542.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-125542.pt)

Gate run:
- [leader-gate-20260818-173842.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-20260818-173842.json)
- [leader-gate-20260818-173842.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-20260818-173842.txt)

## Gate config
- min_seeds: `2`
- required_free_match_margin: `0.0`
- minimum_free_match: `0.2`
- require_not_below_leader: `True`
- require_divergence_or_recovery: `True`
- minimum_post_divergence_next12_delta: `0.0`

## Result
- `promotion_pass = False`

Candidate snapshot:
- seed1: teacher `0.4625`, free_match `0.1000`, heldout `0.103125`, first_div `41`
- seed2: teacher `0.4500`, free_match `0.1875`, heldout `0.128125`, first_div `41`

Aggregate:
- candidate_mean_free_match: `0.14375`
- candidate_mean_heldout_free_match: `0.115625`

## Interpretation
V5 boundary-aware rollout loss (current settings) does not pass the frozen acceptance gate and is below the V2 leader on rollout quality for both seeds.