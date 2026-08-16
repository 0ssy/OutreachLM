# OutreachLM — Targeted Position-41 Token-Loss Intervention

## Goal
Run one surgical intervention on the isolated local failure:
- keep current leader recipe (V2 + Balanced CE/LS + recovery loss weight 2.0),
- add an extra loss only for token prediction at index `40` (position 41),
- evaluate whether this moves:
  1. position-41 teacher accuracy,
  2. fallback-token bias,
  3. first divergence,
  4. and free-running match beyond 20%.

## Script and artifacts
Script:
- [position41_token_loss_intervention.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/position41_token_loss_intervention.py)

Outputs:
- [v2-position41-intervention-20260816-120905.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-position41-intervention-20260816-120905.json)
- [v2-position41-intervention-20260816-120905.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-position41-intervention-20260816-120905.txt)
- [v2-position41-intervention-20260816-120905.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-position41-intervention-20260816-120905.pt)

## Intervention setup
- Resume: `v2-divergence-intervention-20260816-113809.pt`
- Keep:
  - balanced CE + label smoothing `0.05`
  - recovery loss start index `40`
  - recovery weight `2.0`
- Add:
  - position-41 loss term on teacher logits at index `40`
  - position loss weight `1.0`

## Result summary
| Metric | Before (leader) | After intervention |
|---|---:|---:|
| teacher_top1 | 0.4625 | **0.5000** |
| free_match | **0.2000** | 0.1875 |
| prompt_logit_cosine | 0.9191 | 0.9303 |
| rollout_mean_entropy | 3.8613 | 3.8154 |
| first_repeated_trigram_step | 39 | 35 |
| first_free_divergence | 41 | 41 |

Position-41 systematic stats (4096 windows):
- teacher match @pos41: `0.4004 -> 0.4355` (improved)
- free match @pos41: `0.2461 -> 0.2383` (slightly worse)
- most common wrong predicted token:
  - before: `' '` (fraction `0.2821`)
  - after: `'t'` (fraction `0.2215`)

## Interpretation
- This intervention improves teacher-facing local classification at the problematic position.
- But it does **not** improve the primary rollout objective; free-running match drops below the current 20% leader.
- First divergence remains fixed at 41.

Conclusion: this branch is not a replacement for the current leader checkpoint.
