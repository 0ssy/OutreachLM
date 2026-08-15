# OutreachLM — Objective Intervention (Balanced CE + Label Smoothing)

## Goal
Run one training/objective intervention with **fixed architecture** (no model-size changes), replacing only the training loss behavior.

## Intervention
Used weighted cross-entropy with label smoothing:
- class weights from training character frequencies: `w_i ∝ p_i^-0.5`, normalized and clipped
- label smoothing: `0.05`

Architecture, optimizer family (AdamW), and LR scheduler form were unchanged.

## Script and artifacts
Script:
- [objective_intervention_experiment.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/objective_intervention_experiment.py)

Raw outputs:
- [objective-intervention-20260815-131950.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/objective-intervention-20260815-131950.json)
- [objective-intervention-20260815-131950.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/objective-intervention-20260815-131950.txt)

## Results

Baseline snapshot:
- teacher top-1: `0.512500`
- free-running match: `0.100000`
- prompt logit cosine mean: `0.994433`
- rollout mean entropy: `1.716832`
- first repeated trigram step: `20`

After intervention training:
- teacher top-1: `0.525000`
- free-running match: `0.100000`
- prompt logit cosine mean: `0.820768`
- rollout mean entropy: `2.974022`
- first repeated trigram step: `36`

Delta:
- teacher top-1: `+0.012500`
- free-running match: `+0.000000`
- prompt logit cosine mean: `-0.173665`

Observed continuation shift (`OutreachLM is`):
- baseline: repetitive `"and the ... and the company ..."` pattern
- intervention: still repetitive, but later and with a different attractor phrase (`"and the contraction ..."`-like loop)

## Interpretation
- The intervention substantially weakens global logit alignment and increases rollout entropy.
- It improves teacher-forced accuracy slightly.
- It does **not** yet improve free-running target match rate.

So this objective change helps representation-to-logit separation, but by itself does not fully solve autoregressive trajectory instability.
