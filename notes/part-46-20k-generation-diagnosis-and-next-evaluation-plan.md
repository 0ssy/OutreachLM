# OutreachLM — 20k Generation Diagnosis and Next Evaluation Plan

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Current diagnosis (from token-level generation trace)
At ~20,000 steps, OutreachLM shows clear character-level learning but limited semantic coherence.

Observed behavior:
- character prediction: strong
- spacing and common word-shape transitions: strong
- real-word fragments: frequent
- semantic continuation: weak/inconsistent
- long-range coherence: weak

Example pattern:
- outputs are often near-words and plausible local syntax, but meaning drifts quickly.

## Why this is happening
Current model capacity is very small:
- context length: 32
- embedding dim: 64
- transformer layers: 1
- attention heads: 4
- character-level tokenizer

This setup can learn short-range character statistics well, but has limited representational power for robust semantic continuation.

## Training itself is working
Validation trend (approximate progression):
- 1k: loss 2.393, ppl 10.94, acc 31.73%
- 2k: loss 2.253, ppl 9.51, acc 35.18%
- 5k: loss 2.073, ppl 7.95, acc 39.76%
- 6k: loss 2.032, ppl 7.63, acc 40.87%
- 10k: loss 1.960, ppl 7.10, acc 42.75%
- 15k: loss 1.919, ppl 6.81
- 20k: loss 1.893, ppl 6.64, acc 44.67%

Conclusion:
- optimization is improving held-out metrics
- generation limitations are now mostly capacity/context/tokenization/evaluation bottlenecks, not “no learning”

## Immediate strategy
Do **not** blindly push to 50k+ steps yet.

Before architecture changes, run a stronger no-training evaluation suite to locate the primary bottleneck:

1. deterministic generation baseline  
2. multiple temperatures  
3. prompt set diversity  
4. repetition-rate checks  
5. malformed-word rate / word integrity  
6. context sensitivity tests  
7. next-token probability behavior by prompt

Then decide whether the next highest-impact change is:
- architecture size (layers/dim/heads),
- context length increase (likely first major candidate),
- tokenizer upgrade,
- dataset quality/coverage,
- optimization schedule,
- sampling strategy.

## Guardrail
No hardcoded prompt-answer behavior.  
Improvements must come from learned representations, not handcrafted outputs.