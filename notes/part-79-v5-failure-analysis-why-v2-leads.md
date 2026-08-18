# OutreachLM — Part 79: V5 Failure Analysis (Why V2 w=2.0 Still Leads)

## Central question
Why does V2 `recovery_weight=2.0` remain best on free-running rollout while several higher-capacity or higher-teacher-accuracy variants underperform?

## Evidence snapshot
Leader:
- [v2-divergence-intervention-20260816-113809.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v2-divergence-intervention-20260816-113809.txt)
  - teacher `0.4625`, free-match `0.2000`, divergence `41`, rollout entropy `3.8613`

V5 boundary-aware rollout (2 seeds):
- [v5-boundary-rollout-intervention-20260818-123304.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-123304.txt)
  - teacher `0.4625`, free-match `0.1000`, divergence `41`, entropy `3.8372`
- [v5-boundary-rollout-intervention-20260818-125542.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/v5-boundary-rollout-intervention-20260818-125542.txt)
  - teacher `0.4500`, free-match `0.1875`, divergence `41`, entropy `3.3319`

Gate result:
- [leader-gate-20260818-173842.txt](C:/Users/josep/OneDrive/Desktop/OutreachLM/experiments/leader-gate-20260818-173842.txt)
  - promotion fail; both seeds below leader free-match and held-out criteria.

## Main findings
1. **First divergence location is not moving**  
   Across leader, V4, and V5 runs, first divergence stays at `41`.  
   So current interventions are not fixing the first-error trigger.

2. **Higher teacher quality does not guarantee better rollout**  
   We continue to see teacher accuracy and free-match decouple.  
   The failure is not plain under-training.

3. **Post-boundary optimization can improve local recovery but still hurt sequence-level objective**  
   In V5 gate report, one seed improved post-divergence next-12 recovery (`0.4167` vs leader `0.3333`) yet free-match collapsed to `0.1000`.  
   This indicates local boundary improvements can be offset by broader trajectory damage.

4. **Lower rollout entropy correlates with weaker rollout robustness in failed V5 seeds**  
   Failed V5 seeds show reduced entropy (`3.84`, `3.33`) vs leader (`3.86`), consistent with over-confident/free-running brittleness rather than robust corrective behavior.

## Working explanation
V2 w=2.0 sits in a **stability regime** where it is less optimal under teacher forcing but more tolerant to its own off-policy inputs.  
Recent losses (including V5 boundary rollout) likely over-shape local behavior, reducing global rollout calibration and causing earlier collapse modes despite similar or better teacher-side metrics.

## V6 implication
V6 should target **rollout calibration under perturbed contexts** (sequence-level robustness objective), not just teacher alignment or local boundary correction pressure.
