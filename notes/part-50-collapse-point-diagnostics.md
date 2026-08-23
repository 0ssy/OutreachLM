# OutreachLM — Collapse Point Diagnostics (Experiments 1/2/3)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Identify whether collapse happens at:
1. prompt representation stage,
2. prompt-to-logit projection stage,
3. autoregressive rollout stage,
4. train vs inference distribution shift.

## Script and artifacts
Script:
- [collapse_diagnostics.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/collapse_diagnostics.py)

Raw outputs:
- [collapse-diagnostics-20260815-125132.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/collapse-diagnostics-20260815-125132.json)
- [collapse-diagnostics-20260815-125132.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/collapse-diagnostics-20260815-125132.txt)

## Experiment 1 — hidden-state vs logit collapse
Selected pair results:

- `Machine learning allows` vs `A computer system can`
  - hidden cosine: `0.0808`
  - logit cosine: `0.9918`
  - symmetric KL: `0.8336`

- `Machine learning allows computers to` vs `The purpose of a transformer`
  - hidden cosine: `0.1174`
  - logit cosine: `0.9925`
  - symmetric KL: `1.8386`

- `OutreachLM is` vs `Machine`
  - hidden cosine: `0.4488`
  - logit cosine: `0.9957`
  - symmetric KL: `0.9762`

Interpretation:
- hidden states are clearly separated for many prompt pairs,
- but final logits are very similar (very high cosine),
- indicating strong convergence at output distribution level.

## Experiment 2 — attractor transition trace
Prompt: `OutreachLM is` (greedy rollout)

- first repeated bigram step: `19`
- first repeated trigram step: `20`

Early-step dynamics show the orbit entry:
- step 1: `' '` with high confidence and low entropy
- steps 2–4: `a -> n -> d` with confidence spike on `d`
- step 5: `' '` high confidence
- then recurrence repeats and drifts into repetitive phrase attractor.

Interpretation:
- collapse begins early in rollout,
- with a repeatable low-entropy local pathway (`" and "`-like transition).

## Experiment 3 — teacher forcing vs free running
Validation sequence source (auto-selected): substring near:
- `the content on the Xbox 360 Marketplace, ...`

Metrics:
- teacher-forcing top-1 accuracy: `0.5125`
- teacher-forcing average gold probability: `0.3411`
- free-running match rate vs target continuation: `0.1000`
- first divergence position: `41`

Teacher-forced continuation quality is materially better than free-running match behavior.

Interpretation:
- classic train/inference divergence is present:
  - model predicts reasonably under true-history conditioning,
  - but drifts rapidly under self-generated history.

## Conclusion
The failure mode is now localized:
- not pure hidden representation collapse,
- substantial collapse occurs in autoregressive rollout and output-level dynamics,
- with evidence of train/inference distribution divergence.

This supports next experiments focused on capacity/context and rollout robustness rather than blind extra steps on the current architecture.