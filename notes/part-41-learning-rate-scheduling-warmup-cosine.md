# OutreachLM — Learning-Rate Scheduling (Warmup + Cosine Decay)

## Goal
Keep AdamW unchanged and control update magnitude with a step-dependent learning rate schedule.

## Scheduler configuration
Added in [train.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py):

```python
WARMUP_STEPS = 500
MIN_LEARNING_RATE_RATIO = 0.1
```

Meaning:
- base LR is still configured by `LEARNING_RATE` / `--learning-rate`
- minimum LR is derived as:

```python
min_lr = base_lr * MIN_LEARNING_RATE_RATIO
```

So with base LR `0.001`, minimum LR becomes `0.0001`.

## New scheduler function
Added:

```python
get_learning_rate(...)
```

with:
- linear warmup
- cosine decay
- clamped progress in `[0, 1]`

This gives:

```
warmup -> peak/base lr -> gradual decay -> smaller lr
```

## Applied in training loop
Before `optimizer.step()`, trainer now computes and applies LR per step:

```python
current_learning_rate = get_learning_rate(...)
for parameter_group in optimizer.param_groups:
    parameter_group["lr"] = current_learning_rate
```

AdamW remains the optimizer; only LR changes over time.

## Logging update
Training logs now expose LR directly:

```python
Step XXXXX | Train Loss: ... | LR: 0.00......
```

This makes scheduler behavior observable in live training output.

## CLI control added
New arguments:
- `--warmup-steps`
- `--min-learning-rate-ratio`

Both are wired into `train(...)`, so schedule shape can be changed without code edits.

## Deterministic resume implication
Scheduler state is deterministic from:
- step
- max training steps
- base LR
- warmup steps
- min LR ratio

No separate scheduler checkpoint is required.

Important caveat remains:
changing `--steps` on resume changes the schedule curve, so run-configuration compatibility checks are the next hardening step.

---

## Run results (resume to 10,000 steps)
Command:

```bash
python -m outreachlm.train --resume --steps 10000
```

Observed scheduler behavior:
- LR decayed smoothly during resumed run:
  - step 6100: `0.00042531`
  - step 7000: `0.00030400`
  - step 8000: `0.00019498`
  - step 9000: `0.00012443`
  - step 10000: `0.00010000` (configured minimum)

Validation outcome at step 10,000:
- Validation loss: `1.959803`
- Perplexity: `7.0979`
- Accuracy: `42.75%`
- Best checkpoint updated: `outreachlm_best.pt`

Progression snapshot:

| Step | Validation loss | Perplexity | Accuracy |
|---|---:|---:|---:|
| 1,000 | 2.3926 | 10.9416 | 31.73% |
| 2,000 | 2.2527 | 9.5130 | 35.18% |
| 5,000 | 2.0727 | 7.9466 | 39.76% |
| 6,000 | 2.0324 | 7.6326 | 40.87% |
| 10,000 | **1.9598** | **7.0979** | **42.75%** |

Interpretation:
held-out performance continued improving while LR decayed toward the configured floor.

---

## Instructor interpretation and next control point
This run is valid and scheduler behavior is correct.

Key improvements from step 6000 to 10000:
- Validation perplexity: `7.6326 -> 7.0979` (about 7.0% improvement)
- Validation accuracy: `40.87% -> 42.75%`

Observed LR decay sequence:

```
6100   0.00042531
7000   0.00030400
8000   0.00019498
9000   0.00012443
10000  0.00010000
```

Decision:
pause additional training steps and harden checkpoint/run-state compatibility before further scaling.

Next implementation target:
- store full training-run state in checkpoint
- verify resume compatibility against current config before training
- fail fast on mismatch with a clear error instead of silently continuing

Compatibility scope to validate on resume:
- model shape-driving config (context length, embedding dim)
- optimizer/scheduler-driving config (base LR, warmup, min LR ratio, batch size, total steps policy)
- data identity (corpus source and tokenizer/vocabulary identity)
