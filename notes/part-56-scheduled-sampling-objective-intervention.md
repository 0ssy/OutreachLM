# OutreachLM — Scheduled Sampling Intervention (Architecture Fixed)

## Goal
Test whether exposure-bias-focused training improves free-running behavior when starting from the same baseline and objective family:
- architecture fixed,
- tokenizer/dataset/context fixed,
- objective baseline = balanced CE + label smoothing,
- intervention = add scheduled sampling during training.

## Script and artifacts
Script:
- [scheduled_sampling_intervention_experiment.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/scheduled_sampling_intervention_experiment.py)

Raw outputs:
- [scheduled-sampling-intervention-20260815-132617.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/scheduled-sampling-intervention-20260815-132617.json)
- [scheduled-sampling-intervention-20260815-132617.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/scheduled-sampling-intervention-20260815-132617.txt)

## Procedure
- Trained for 1500 steps.
- Kept balanced CE + label smoothing (`0.05`) unchanged.
- Added scheduled sampling rate linearly from `0.0 -> 0.4` across steps.
- Scheduled sampling implementation: for each batch, replace a fraction of input tokens (except first) with model-predicted previous-token outputs.

## Metric comparison
| Condition | teacher_top1 | free_match | prompt_logit_cosine | rollout_mean_entropy | first_repeated_bigram_step | first_repeated_trigram_step | first_free_divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5125 | 0.1000 | 0.9944 | 1.7168 | 19 | 20 | 41 |
| Balanced+LS (control) | 0.5250 | 0.1000 | 0.8208 | 2.9740 | 22 | 36 | 41 |
| Balanced+LS+ScheduledSampling | 0.4750 | 0.1000 | 0.9023 | 3.4721 | 30 | 32 | 41 |

## Additional observations
- Scheduled sampling raised entropy further (`3.47`) and delayed first repeated bigram (`30`), but did not improve free-match.
- Compared with Balanced+LS control, scheduled sampling degraded:
  - teacher top-1 (`0.5250 -> 0.4750`)
  - prompt-logit separation (`0.8208 -> 0.9023`, higher is worse here).
- Mid-training checkpoints showed temporary free-match improvement (`0.1375` at steps 750/1125) that regressed by step 1500.

## Conclusion
This specific scheduled-sampling setup did **not** produce a durable free-running improvement.  
Exposure-bias remains plausible, but this intervention (as configured) is not yet the right fix.
