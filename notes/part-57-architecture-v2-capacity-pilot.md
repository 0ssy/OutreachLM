# OutreachLM — Architecture V2 Capacity Pilot (Fixed Objective)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Test a sequence-capacity/architecture intervention (V2) with everything else controlled to the Balanced+LS objective family:
- tokenizer/dataset fixed,
- objective fixed (balanced CE + label smoothing),
- optimizer family fixed (AdamW),
- metric suite fixed.

## V2 configuration
- context length: `128` (from `32`)
- embedding dim: `128` (from `64`)
- layers: `2` (from `1`)
- heads: `4` (unchanged)

## Script and artifacts
Script:
- [architecture_capacity_pilot.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/architecture_capacity_pilot.py)

Raw outputs:
- [architecture-v2-pilot-20260815-133251.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/architecture-v2-pilot-20260815-133251.json)
- [architecture-v2-pilot-20260815-133251.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/architecture-v2-pilot-20260815-133251.txt)
- V2 pilot checkpoint:
  - [architecture-v2-pilot-20260815-133251.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/architecture-v2-pilot-20260815-133251.pt)

## Results (1500-step pilot)
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeated_bigram_step | first_repeated_trigram_step | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (current checkpoint) | 0.5125 | 0.1000 | 0.9944 | 1.7168 | 19 | 20 | 41 |
| Architecture V2 pilot | 0.2875 | 0.1125 | 0.9534 | 3.3459 | 19 | 20 | 41 |

Additional observations:
- V2 pilot loss moved `7.3524 -> 4.1371` (still clearly undertrained in this budget).
- Free-running continuation remained repetitive (`"the the the ... and the ..."` pattern).

## Interpretation
- At this short budget, V2 did **not** improve teacher-forced quality and did not move first-divergence/repeat-onset metrics.
- Slight free-match lift (`0.1000 -> 0.1125`) is not accompanied by broader stability gains.
- This pilot is likely compute-limited/undertrained for the larger model; it does not yet provide evidence that capacity increase alone solves rollout failure.