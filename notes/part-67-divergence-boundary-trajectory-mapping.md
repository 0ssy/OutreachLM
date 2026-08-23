# OutreachLM — Divergence-Boundary Trajectory Mapping (Leader Checkpoint)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Test whether, at the teacher->free transition boundary, hidden-state differences are being compressed in output space.

## Checkpoint analyzed
- [v2-divergence-intervention-20260816-113809.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.pt)

## Script and artifacts
Script:
- [divergence_trajectory_mapping_analysis.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/divergence_trajectory_mapping_analysis.py)

Outputs:
- [divergence-trajectory-mapping-20260816-121303.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/divergence-trajectory-mapping-20260816-121303.json)
- [divergence-trajectory-mapping-20260816-121303.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/divergence-trajectory-mapping-20260816-121303.txt)

## Key findings
Canonical divergence slice:
- position 41 remains the first divergence.
- At position 41, teacher and free contexts are still identical and both predict the same wrong token:
  - hidden cosine = `1.0`, logit cosine = `1.0`, KL = `0.0`.
- After divergence (position 42+), teacher/free hidden states separate strongly while logits stay much more aligned:
  - e.g. position 42: hidden cosine `0.7380`, logit cosine `0.9752`.
  - e.g. position 48: hidden cosine `0.3699`, logit cosine `0.9287`.

Systematic mapping over 4096 validation windows:
- pos40 prediction mismatch rate: `0.593262`.
- pos41 teacher/free match rates: `0.400391 / 0.246094`.
- pos41 free match conditioned on context:
  - context same: `0.409364`
  - context diff: `0.134156`
- teacher/free hidden cosine mean:
  - all/same/diff = `0.857386 / 1.000000 / 0.759611`
- teacher/free logit cosine mean:
  - all/same/diff = `0.977240 / 1.000000 / 0.961636`
- KL_sym mean:
  - all/same/diff = `0.392433 / 0.000000 / 0.661484`

## Interpretation
This confirms a two-part failure pattern:
1. **First error at position 41 is local and teacher-visible** (not caused by immediate context drift).
2. **After context diverges, hidden states separate more than output logits**, indicating continued output-space alignment/compression under perturbed histories.

So the persistent rollout weakness is not just exposure bias or just local token classification; it is the interaction of local first-error risk plus post-error mapping behavior.