# OutreachLM — Logit Direction and Trajectory Instability

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Test whether near-0.99 prompt logit cosine is dominated by a global logit direction, and localize teacher-forced vs free-running distribution divergence.

## Script and artifacts
Script:
- [logit_alignment_diagnostics.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/logit_alignment_diagnostics.py)

Raw outputs:
- [logit-alignment-diagnostics-20260815-130044.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/logit-alignment-diagnostics-20260815-130044.json)
- [logit-alignment-diagnostics-20260815-130044.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/logit-alignment-diagnostics-20260815-130044.txt)

## Experiment 4 — dominant logit direction (validation contexts)
Using 2048 sampled validation contexts:

- Raw logit matrix explained variance:
  - PC1: `0.990257`
  - PC2: `0.003007`
  - PC5: `0.000486`
  - PC10: `0.000192`
  - cumulative PC1/PC2/PC5/PC10: `0.990257 / 0.993264 / 0.996539 / 0.997921`

- Mean-centered logit matrix explained variance:
  - PC1: `0.797498`
  - PC2: `0.062543`
  - PC5: `0.010301`
  - PC10: `0.003692`
  - cumulative PC1/PC2/PC5/PC10: `0.797498 / 0.860040 / 0.927854 / 0.956481`

- Cosine(logits_i, mean_logits) mean/std/min/max:
  - `0.993869 / 0.004489 / 0.945436 / 0.998859`

Interpretation:
- raw logits live in a very narrow dominant direction,
- subtracting the global mean reveals larger context-specific structure.

## Experiment 5 — raw vs centered prompt similarities
Across prompt pairs:
- raw cosine mean/min/max: `0.994433 / 0.991302 / 0.997458`
- centered cosine mean/min/max: `0.186568 / -0.673381 / 0.942933`

Key pairs:
- `Machine learning allows` vs `A computer system can`
  - raw: `0.991788`
  - centered: `-0.131523`
- `Machine learning allows computers to` vs `The purpose of a transformer`
  - raw: `0.992509`
  - centered: `-0.659817`
- `OutreachLM is` vs `Machine`
  - raw: `0.995749`
  - centered: `0.529006`

Interpretation:
- the ~0.99 raw cosine was strongly dominated by a shared global component,
- centering exposes meaningful context-dependent directional differences.

## Experiment 6 — top tokens of mean logit direction
Top mean-logit tokens are mostly global high-frequency character priors:
- `i`, `e`, `a`, `t`, `s`, `o`, `r`, `l`, space, `n`, ...

Interpretation:
- dominant direction is aligned with a broad character prior, not a single fixed phrase.

## Experiment 7 — teacher-forced vs free-run distribution drift
Validation slice source: `fallback-start-of-validation`  
Prompt: `'the content on the Xbox 360 Marketplace,'`

Summary:
- teacher top-1 accuracy: `0.480000`
- free top-1 accuracy: `0.090000`
- generated match rate: `0.090000`
- first mismatch position: `41`
- first teacher/free top-1 disagreement: `42`
- first KL_sym > 0.5: `42`
- first KL_sym > 1.0: `42`
- mean/max KL_sym: `3.458544 / 9.210614`
- mean/max TV distance: `0.718486 / 0.990655`

Local transition around divergence:
- pos 41: both teacher/free top-1 are `'a'` (wrong vs gold `'n'`), KL_sym `0.0`
- pos 42 onward: teacher/free distributions split sharply (e.g. KL_sym `6.5673`, TV `0.9722`)

Interpretation:
- first error appears before trajectory split,
- once in self-generated history, distribution drift becomes severe almost immediately.

## Conclusion
This run supports a refined diagnosis:
- hidden states are distinct,
- output directions are strongly dominated by a global mean/prior component,
- context signal exists but is comparatively weaker in raw cosine geometry,
- free-running instability is a trajectory problem that spikes right after early local errors.

This is better described as **context-to-logit alignment around a dominant global direction + autoregressive trajectory drift**, not literal output-matrix rank collapse.