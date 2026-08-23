# OutreachLM — Pre-Divergence Window Weight Sweep

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Run the requested controlled sweep on pre-divergence emphasis (positions `37-40`) while keeping the successful recovery objective fixed:
- resume from strengthened w=2.0 checkpoint,
- keep recovery loss weight `2.0`,
- sweep only pre-window weight: `1.0`, `0.5`, `2.0`.

## Setup
Resume checkpoint:
- [v2-divergence-intervention-20260816-113809.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.pt)

Sweep script:
- [pre_divergence_window_weight_sweep.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/pre_divergence_window_weight_sweep.py)

Artifacts:
- [v2-pre-window-sweep-20260816-114348.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-pre-window-sweep-20260816-114348.json)
- [v2-pre-window-sweep-20260816-114348.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-pre-window-sweep-20260816-114348.txt)
- [v2-pre-window-1.00-20260816-114348.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-pre-window-1.00-20260816-114348.pt)
- [v2-pre-window-0.50-20260816-114348.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-pre-window-0.50-20260816-114348.pt)
- [v2-pre-window-2.00-20260816-114348.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-pre-window-2.00-20260816-114348.pt)

## Final metrics comparison
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeat_bigram | first_repeat_trigram | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 w=2.0 checkpoint (start) | 0.4625 | 0.2000 | 0.9191 | 3.8613 | 22 | 39 | 41 |
| pre-window weight 1.0 | 0.4875 | 0.1750 | 0.9150 | 3.7818 | 22 | 38 | 41 |
| pre-window weight 0.5 | 0.4875 | 0.1625 | 0.9121 | 3.7840 | 22 | 38 | 41 |
| pre-window weight 2.0 | 0.4875 | 0.1875 | 0.9195 | 3.7671 | 22 | 38 | 41 |

## Interpretation
- All three pre-window variants increased teacher top-1 (`0.4625 -> 0.4875`).
- None beat the starting checkpoint on free-running match (`0.2000` remained best).
- First divergence stayed fixed at `41` for all variants.

Net result: this pre-divergence weighting pass improved teacher-forced accuracy but did not improve the key free-running metric beyond the current best regime.