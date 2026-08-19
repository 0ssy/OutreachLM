# OutreachLM — Part 84: V6 Failure Analysis (Positions 40..52)

## Scope
Diagnostic-only run (no training) to explain why V6 fails gate recovery despite reproducible free-match gains.

Compared:
- V2 leader: [v2-divergence-intervention-20260816-113809.pt](C:/Users/josep/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.pt)
- V6 seed1: [v4-best-rollout.pt](C:/Users/josep/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1/v4-best-rollout.pt)
- V6 seed1337: [v4-best-rollout.pt](C:/Users/josep/Desktop/OutreachLM/experiments/v6-training-rollout-calibration-seed1337/v4-best-rollout.pt)

## Command run
```powershell
python -m outreachlm.rollout_calibration_regime_analysis `
  --leader v2=experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate v6_seed1=experiments/v6-training-rollout-calibration-seed1/v4-best-rollout.pt `
  --candidate v6_seed1337=experiments/v6-training-rollout-calibration-seed1337/v4-best-rollout.pt `
  --sample-count 1024 `
  --sample-seed 42 `
  --sample-batch-size 128 `
  --position-start 40 `
  --position-end 52 `
  --fallback-topk 5 `
  --output-dir experiments `
  --report-prefix v6-failure-analysis
```

## Artifacts
- Full report text: [v6-failure-analysis-20260819-154639.txt](C:/Users/josep/Desktop/OutreachLM/experiments/v6-failure-analysis-20260819-154639.txt)
- Full report JSON: [v6-failure-analysis-20260819-154639.json](C:/Users/josep/Desktop/OutreachLM/experiments/v6-failure-analysis-20260819-154639.json)

## Key results
- V2 leader: `free_match=0.2000`, `post_div_next12=0.3333`, `first_div=41`
- V6 seed1: `free_match=0.2250`, `post_div_next12=0.1667`, `first_div=41`
- V6 seed1337: `free_match=0.2250`, `post_div_next12=0.2500`, `first_div=41`

So V6 remains better on overall free-match, but still worse than V2 on post-divergence recovery.

## Position-window interpretation (40..52)
- At position 41 and immediately after, both V6 seeds show:
  - lower free gold-token probability vs V2,
  - worse margin shift (more negative) vs V2,
  - increased fallback probability mass under context-diff.
- Hidden/logit movement deltas grow strongly negative from 42 onward in both seeds, indicating trajectory drift remains amplified after first error.
- First divergence remains fixed at 41 for all three models.

## Conclusion
V6 rollout-calibration improves global free-match and held-out mean, but does not repair the post-error transition dynamics needed by the gate. V2 remains production leader.
