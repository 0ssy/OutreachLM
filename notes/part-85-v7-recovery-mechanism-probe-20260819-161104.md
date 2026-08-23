# OutreachLM — Part 85: V7 Recovery Mechanism Probe (No Training)

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## Scope
Diagnostic-only probe comparing post-error behavior at positions `40..52` for:
- V2 leader
- V6 seed1 (best-rollout)
- V6 seed1337 (best-rollout)

No training or architecture changes.

## Command
```powershell
python -m outreachlm.rollout_calibration_regime_analysis `
  --leader v2=experiments/v2-divergence-intervention-20260816-113809.pt `
  --candidate v6_seed1=experiments/v6-training-rollout-calibration-seed1/v4-best-rollout.pt `
  --candidate v6_seed1337=experiments/v6-training-rollout-calibration-seed1337/v4-best-rollout.pt `
  --sample-count 4096 `
  --sample-seed 42 `
  --sample-batch-size 256 `
  --position-start 40 `
  --position-end 52 `
  --fallback-topk 10 `
  --output-dir experiments `
  --report-prefix v7-recovery-mechanism-probe
```

## Artifacts
- Probe text: [v7-recovery-mechanism-probe-20260819-161104.txt](C:/Users/josep/Desktop/OutreachLM/experiments/v7-recovery-mechanism-probe-20260819-161104.txt)
- Probe JSON: [v7-recovery-mechanism-probe-20260819-161104.json](C:/Users/josep/Desktop/OutreachLM/experiments/v7-recovery-mechanism-probe-20260819-161104.json)
- Consolidated text: [v7-recovery-mechanism-probe-consolidated-20260819-161104.txt](C:/Users/josep/Desktop/OutreachLM/experiments/v7-recovery-mechanism-probe-consolidated-20260819-161104.txt)
- Consolidated JSON: [v7-recovery-mechanism-probe-consolidated-20260819-161104.json](C:/Users/josep/Desktop/OutreachLM/experiments/v7-recovery-mechanism-probe-consolidated-20260819-161104.json)

## Key result
- V2: `free_match=0.2000`, `first_div=41`, `post_div_next12=0.3333`
- V6 seed1: `free_match=0.2250`, `first_div=41`, `post_div_next12=0.1667`
- V6 seed1337: `free_match=0.2250`, `first_div=41`, `post_div_next12=0.2500`

Interpretation:
- V6 is reproducibly better on global free-match and held-out mean.
- V6 remains worse than V2 on post-divergence recovery.
- Divergence onset is unchanged at position 41.
- Position deltas show stronger post-error trajectory drift in V6 (hidden/logit movement) and persistently worse margin shifts across 41..52.

## Decision
Keep V2 as production leader under the frozen gate. Keep V6 as a research candidate checkpoint.