# OutreachLM — Part 81: V6 Mechanism Hypothesis + Pre-Training Success Criteria

## Status
No V6 training run in this step.  
This note freezes the mechanism hypothesis and the acceptance criteria before any V6 implementation.

## Frozen mechanism hypothesis
**Rollout Calibration / Distribution Preservation** failure at boundary `41..43`.

Observed reproducible pattern (across failed V5 seeds):
- position `40`: mostly normal
- position `41`: first wrong decision
- positions `42..43`: lower gold-token probability, worse margins, more fallback mass, larger hidden/logit movement deltas vs V2
- `44+`: trajectory damage propagates

So the primary question is:

> after context drift begins at 41, how do we keep the model’s output distribution useful?

Not:
- maximizing teacher accuracy alone,
- forcing teacher/free hidden-state equality,
- moving first divergence position by itself.

## V6 scientific question (pre-registered)
Can one controlled calibration intervention around `41..43` preserve post-error predictive distributions under self-conditioned rollout well enough to beat the V2 leader across seeds?

## V6 constraints (frozen)
For first V6 test:
- one controlled change only,
- keep architecture/tokenizer/optimizer/data split/eval suite fixed,
- run exactly two seeds first (`seed1`, `seed1337`) before any expansion,
- evaluate only via gate protocol (no manual cherry-picking).

## Success criteria (must all pass)
1. Per-seed free-match: each required seed `> 0.2000`.
2. Held-out slices: each required seed passes held-out threshold vs leader.
3. Divergence/recovery criterion: each seed either
   - improves first divergence, or
   - improves post-divergence next-12 match rate.
4. No seed below leader free-match.
5. Two-seed robustness: seed set passes as a group (`promotion_pass=True`).

## Failure interpretation rule
If V6 fails these criteria, conclude:
- this specific loss-level calibration approach is insufficient under current setup,
- next step should shift to architecture/decoding-level mechanism changes rather than more loss-weight sweeps.

## Primary references
- [rollout-calibration-analysis-20260818-175332.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/rollout-calibration-analysis-20260818-175332.txt)
- [leader-gate-20260818-173842.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-20260818-173842.txt)
- [part-80-rollout-calibration-regime-analysis-v2-vs-v5.md](C:/Users/josep/OneDrive/Desktop/OutreachLM/notes/part-80-rollout-calibration-regime-analysis-v2-vs-v5.md)
