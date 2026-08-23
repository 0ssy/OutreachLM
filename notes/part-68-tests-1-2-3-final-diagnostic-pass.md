# OutreachLM — Final Diagnostic Pass (Tests 1, 2, 3)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Goal
Run the final three tests in order:
1. hidden-state transition around 38–45,
2. output-head sensitivity around 39–43,
3. one controlled intervention based on tests 1–2.

Leader checkpoint under test:
- [v2-divergence-intervention-20260816-113809.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.pt)

---

## Test 1 — Hidden-state transition
Script:
- [hidden_output_transition_tests.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/hidden_output_transition_tests.py)

Artifacts:
- [tests1-2-hidden-output-transition-20260816-121636.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/tests1-2-hidden-output-transition-20260816-121636.json)
- [tests1-2-hidden-output-transition-20260816-121636.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/tests1-2-hidden-output-transition-20260816-121636.txt)

Key results (positions 38–45):
- divergence onset position: `41`
- hidden cosine distance mean:
  - pos40: `0.0000`
  - pos41: `0.1426`
  - pos42: `0.2220`
  - pos45: `0.3705`
- context mismatch rate:
  - pos41: `0.5933`
  - pos42: `0.8335`
  - pos45: `0.9934`

Interpretation:
- hidden trajectory divergence begins at 41 and grows rapidly after.
- this confirms failure is not only output-space; hidden states themselves separate once context diverges.

---

## Test 2 — Output-head sensitivity
Same artifacts as Test 1 (shared run), using positions 39–43.

Key results:
- at pos41:
  - teacher gold prob `0.1778` -> free gold prob `0.1171`
  - teacher margin(pred-gold) `0.7651` -> free margin `1.2449`
  - logit cosine teacher/free `0.9772`
  - KL symmetric `0.3924`
  - argmax change rate `0.4492`
- as positions advance (42, 43):
  - free gold probability continues dropping (`0.0793`, `0.0639`)
  - top-k overlap drops (`0.5553`, `0.4923`)
  - argmax change rises (`0.6462`, `0.6897`)
  - fallback token under context-diff remains dominated by `' '`.

Interpretation:
- after hidden divergence, output decisions shift in a way that increases wrong-margin confidence, especially toward frequent fallback tokens.

---

## Test 3 — Controlled intervention based on Tests 1–2
Hypothesis from Tests 1–2:
- post-divergence output distributions may need stability against context perturbations.

Intervention:
- Keep leader recipe unchanged (Balanced CE/LS + recovery weight 2.0).
- Add boundary consistency KL loss (teacher vs free logits) at indices `[40,41,42]` for context-diff cases.

Script:
- [test3_boundary_consistency_intervention.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/test3_boundary_consistency_intervention.py)

Artifacts:
- [test3-boundary-consistency-20260816-122240.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/test3-boundary-consistency-20260816-122240.json)
- [test3-boundary-consistency-20260816-122240.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/test3-boundary-consistency-20260816-122240.txt)
- [v2-test3-boundary-consistency-20260816-122240.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-test3-boundary-consistency-20260816-122240.pt)

Decision metrics:
- teacher_top1: `0.4625 -> 0.4750`
- free_match: `0.2000 -> 0.1875` (worse)
- first divergence: `41 -> 41` (unchanged)
- first repeated trigram: `39 -> 22` (worse)

Boundary mapping:
- pos41 free match (all): `0.2461 -> 0.2041` (worse)
- context-diff logit cosine: `0.9616 -> 0.9972` (much tighter alignment)
- context-diff KL: `0.6615 -> 0.0456` (collapsed teacher/free separation)

Interpretation:
- intervention over-regularized boundary logits, reducing useful discriminative differences.
- this branch is rejected on rollout objective.

---

## Final status after Tests 1–3
- Tests 1–2 confirmed hidden divergence onset at 41 and output-side decision drift under context mismatch.
- Test 3 (targeted boundary consistency) did not improve free-running behavior and degraded the leader metric.
- Current leader remains:
  - [v2-divergence-intervention-20260816-113809.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.pt)
  - free-running match = `0.2000`