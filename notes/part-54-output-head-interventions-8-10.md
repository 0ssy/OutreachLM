# OutreachLM — Experiments 8, 9, 10 (Output-Head Intervention Phase)

## Goal
Run the first intervention phase:
1. output-head intervention test,
2. training intervention under the existing objective/optimizer,
3. controlled generation comparison (greedy + sampling).

## Script and artifacts
Script:
- [output_head_intervention_experiments.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/output_head_intervention_experiments.py)

Raw outputs:
- [interventions-8-10-20260815-131511.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/interventions-8-10-20260815-131511.json)
- [interventions-8-10-20260815-131511.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/interventions-8-10-20260815-131511.txt)
- Corrected checkpoint:
  - [corrected-model-20260815-131511.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/corrected-model-20260815-131511.pt)

## Experiment 8 — output-head intervention
Baseline:
- prompt off-diagonal logit cosine mean: `0.994433`

Intervention candidates:
- `centered_output_head` (subtract mean row direction from output-head weights and mean from bias):
  - cosine mean: `0.974546`
  - teacher-forced top-1: `0.512500`
  - free-running match: `0.100000`

- `reinitialized_output_head` (Xavier reinit + unigram-log bias):
  - cosine mean: `0.998394`
  - teacher-forced top-1: `0.212500`
  - free-running match: `0.162500`

Selected candidate:
- `centered_output_head` (best tradeoff: materially reduced cosine without collapsing teacher-forced accuracy).

## Experiment 9 — training intervention (existing objective/optimizer)
Setup:
- objective: unchanged cross-entropy
- optimizer: unchanged AdamW
- LR schedule: unchanged warmup+cosine form
- steps: `1200`
- learning rate: `0.0005`

Results:
- loss first/last: `1.802182 / 1.854177`
- final teacher-forced top-1: `0.512500`
- final free-running match: `0.075000`
- final prompt logit cosine mean: `0.981093`
- final rollout mean entropy: `1.749765`

Interpretation:
- prompt-logit alignment improved vs baseline (`0.9944 -> 0.9811`),
- teacher-forced top-1 stayed flat,
- free-running target match did not improve in this run.

## Experiment 10 — controlled generation comparison
Metrics across the 8 probe prompts.

### Baseline
- greedy:
  - unique continuations: `7`
  - context similarity mean: `0.775000`
  - mean repeated bigram ratio: `0.751582`
  - mean repeated trigram ratio: `0.705128`
- sampling:
  - unique continuations: `8`
  - context similarity mean: `0.248661`
  - mean repeated bigram ratio: `0.202532`
  - mean repeated trigram ratio: `0.062500`

### Corrected
- greedy:
  - unique continuations: `7`
  - context similarity mean: `0.792411`
  - mean repeated bigram ratio: `0.729430`
  - mean repeated trigram ratio: `0.671474`
- sampling:
  - unique continuations: `8`
  - context similarity mean: `0.249554`
  - mean repeated bigram ratio: `0.208861`
  - mean repeated trigram ratio: `0.056090`

Interpretation:
- greedy repetition reduced modestly after correction,
- context-conditioned diversity did not improve in greedy mode,
- sampling behavior remained similar overall.

## Conclusion
This intervention phase provides mixed evidence:
- output-head centering reduces raw prompt-logit alignment,
- but does not by itself fix trajectory instability under free-running generation after short retraining.

The failure mechanism appears to involve more than static output-head geometry and likely includes rollout dynamics that are not resolved by this isolated head intervention.
