# OutreachLM — Part 77: V5 Boundary-Aware Rollout + Frozen Acceptance Gate

## Goal
Start V5 without architecture changes and target the observed failure mechanism:
- wrong decision near position 41,
- context drift,
- rollout degradation and fallback behavior.

## New V5 training script
- [v5_boundary_rollout_intervention.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/v5_boundary_rollout_intervention.py)

### What changes (and what does not)
Unchanged:
- V2 architecture, tokenizer, optimizer schedule, dataset split, evaluation suite.

Changed:
- add **boundary-aware rollout loss** around indices `40..43` on top of:
  - teacher loss,
  - recovery loss.

Mechanism:
1. Force a wrong token at `forced_error_index` (default `40`) using model top-1 unless top-1 equals gold, then top-2.
2. Roll forward autoregressively from that perturbed boundary.
3. Compute CE against gold targets for boundary window (`40..43`).
4. Add to objective:

`total = teacher_loss + recovery_weight * recovery_loss + boundary_weight * boundary_rollout_loss`

This directly optimizes **useful predictive behavior after boundary error**, not hidden-state matching.

## Gate protocol updates (frozen acceptance rules support)
Updated:
- [leader_gating_protocol.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/leader_gating_protocol.py)

New options:
- `--minimum-free-match` (absolute per-seed floor, strict `>`),
- `--require-not-below-leader` (per-seed `>= leader`),
- `--require-divergence-or-recovery` (per-seed: first divergence improves **or** post-divergence next-12 recovery improves),
- `--minimum-post-divergence-next12-delta` (required recovery improvement margin).

## V5 training command (seed 1)
```powershell
python -m outreachlm.v5_boundary_rollout_intervention `
  --resume-artifact experiments/v2-divergence-intervention-20260816-113809.pt `
  --seed 1 `
  --steps 1500 `
  --recovery-loss-weight 2.0 `
  --boundary-loss-weight 1.0 `
  --boundary-start-index 40 `
  --boundary-end-index 43 `
  --forced-error-index 40 `
  --output-dir experiments
```

## V5 gate command (two seeds, frozen rules)
```powershell
python -m outreachlm.leader_gating_protocol `
  --leader-artifact experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate seed1=experiments/<v5-seed1-artifact>.pt `
  --candidate seed2=experiments/<v5-seed2-artifact>.pt `
  --min-seeds 2 `
  --required-free-match-margin 0.0 `
  --minimum-free-match 0.2 `
  --require-not-below-leader `
  --require-divergence-or-recovery `
  --minimum-post-divergence-next12-delta 0.0 `
  --output-dir experiments
```

## Validation
Implemented script checks:
- `python -m outreachlm.v5_boundary_rollout_intervention --help`
- `python -m outreachlm.leader_gating_protocol --help`
- `python -m py_compile outreachlm\v5_boundary_rollout_intervention.py outreachlm\leader_gating_protocol.py`

End-to-end smoke run succeeded:
- [v5-boundary-rollout-intervention-20260818-122036.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-122036.json)
- [v5-boundary-rollout-intervention-20260818-122036.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-122036.txt)
- [v5-boundary-rollout-intervention-20260818-122036.pt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-122036.pt)

Gate-rule smoke run succeeded:
- [leader-gate-v5-rule-smoke-20260818-122526.json](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-v5-rule-smoke-20260818-122526.json)
- [leader-gate-v5-rule-smoke-20260818-122526.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-v5-rule-smoke-20260818-122526.txt)
