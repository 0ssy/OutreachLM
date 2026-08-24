# OutreachLM — Part 95: B1.10 Train Script Seam Migration

## What it is
`train.py` and `train_v4.py` were migrated to consume the new seams (model config/registry, runtime, tokenizer artifact, training config, and V4 loss plan composition) while preserving behavior.

## Why it is there
B1.1–B1.9 introduced deep modules at clear seams. B1.10 wires real training entrypoints onto those seams so the architecture work is actually used in production paths, not just in isolated tests.

## Why it is important
- Converts seam design into live leverage at the script interface.
- Improves locality: model construction, runtime placement, tokenizer serialization, and loss composition now route through dedicated modules.
- Reduces future migration risk by removing duplicated script-local implementations.
- Keeps current training behavior intact while making Phase C changes safer.

## What would happen without it
- The new seam modules would remain partially unused.
- Training scripts would continue carrying duplicated construction/serialization/objective glue.
- Future architecture evolution would require touching multiple script-specific code paths again.

## Scope
Implemented B1.10 migration for `train.py` and `train_v4.py` only, with behavior-preserving wiring.

## Updated
- [train.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train.py)
  - model creation moved to `LegacyV1Config` + model registry
  - runtime placement moved through `SingleDeviceRuntime`
  - tokenizer save path moved to `TokenizerArtifact` seam
- [train_v4.py](C:/Users/josep/Desktop/OutreachLM/outreachlm/train_v4.py)
  - V4 model creation moved to `V4Config` + model registry
  - runtime placement moved through `SingleDeviceRuntime`
  - tokenizer save path moved to `TokenizerArtifact` seam
  - base training loop fields sourced from `TrainingConfig`
  - objective aggregation routed through `LossPlan` terms

## Validation
Full suite run:
- `python -m pytest`
- result: `90 passed`

## B1 progression
- B1.1 ModelArtifact ✅
- B1.2 ModelRegistry ✅
- B1.3 ModelConfig ✅
- B1.4 TrainingConfig ✅
- B1.5 TrainerCore ✅
- B1.6 LossPlan ✅
- B1.7 EvaluationProfile ✅
- B1.8 TokenizerArtifact ✅
- B1.9 Runtime/Device seam ✅
- B1.10 train script seam migration ✅
