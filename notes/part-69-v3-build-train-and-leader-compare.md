# OutreachLM — V3 Build, Training Run, and Leader Comparison

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Requested V3 configuration
- vocab: `490` (existing tokenizer)
- context length: `256`
- embedding dim: `256`
- transformer layers: `4`
- attention heads: `8`
- objective: balanced CE + label smoothing `0.05`
- recovery loss (perturbed histories): weight `2.0`
- shifted-target training objective: unchanged

## Implementation
Training script:
- [v3_train_and_compare.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/v3_train_and_compare.py)

Comparison script:
- [evaluate_suite_compare.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/evaluate_suite_compare.py)

## Training run (from scratch)
Run directory:
- [v3-training-20260816-123145](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-training-20260816-123145)

Saved checkpoints:
- [v3-checkpoint-step-00500.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-training-20260816-123145/v3-checkpoint-step-00500.pt)
- [v3-checkpoint-step-01000.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-training-20260816-123145/v3-checkpoint-step-01000.pt)
- [v3-checkpoint-step-01500.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v3-training-20260816-123145/v3-checkpoint-step-01500.pt)

## Same evaluation suite vs current leader
Comparison artifacts:
- [eval-compare-20260816-130942.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/eval-compare-20260816-130942.json)
- [eval-compare-20260816-130942.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/eval-compare-20260816-130942.txt)

Metrics:
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeat_bigram | first_repeat_trigram | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Leader (V2 w=2.0) | 0.4625 | 0.2000 | 0.9191 | 3.8613 | 22 | 39 | 41 |
| V3 checkpoint @ step 1500 | 0.2625 | 0.1875 | 0.9531 | 3.8824 | 19 | 20 | 41 |

## Readout
At step 1500, V3 is underperforming the current leader on key rollout metrics and teacher accuracy.  
The saved checkpoints make it straightforward to continue V3 training from this scratch run if desired.