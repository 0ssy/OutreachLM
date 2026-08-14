# OutreachLM — Resume Training Checkpoint + Next Metrics

## Confirmed result
Resume training is working and producing valid progress.

### Observed metrics

| Step | Train loss | Validation loss | Validation perplexity | Accuracy |
|---|---:|---:|---:|---:|
| 1000 | 2.4244 | 2.3926 | 10.9431 | 31.73% |
| 2000 | 2.2518 | 2.2527 | 9.5130 | 35.18% |

## What this means
Most important signal:

```
Validation loss: 2.3926 -> 2.2527
```

This indicates improvement on unseen validation text, not just memorization of training windows.

Resume path is verified:

```
step 1000
  -> checkpoint saved
  -> model + AdamW state reloaded
  -> resumed at step 1100
  -> continued to step 2000
```

So OutreachLM now has a persistent training process.

---

## Current issue to address before longer training
Training loss logs are noisy because they are single-minibatch snapshots.

Example fluctuations:

```
2.324
2.254
2.390
2.391
2.233
2.389
2.336
2.348
2.279
2.252
```

This does not imply failure; it implies coarse measurement.

---

## Next engineering upgrade (before long runs)

### 1) Running/interval train-loss averaging
Add averaged train metrics over a window/interval, not only raw per-batch loss.

Target style:

```
Step 1000 | Train Loss: 2.4244 | Avg: ...
Step 1100 | Train Loss: 2.3241 | Avg: ...
```

### 2) Regular validation monitoring during training
Evaluate validation at scheduled intervals and log:

- validation loss
- perplexity
- accuracy

Target style:

```
Step 1000
  Validation Loss: ...
  Perplexity: ...

Step 2000
  Validation Loss: ...
  Perplexity: ...
```

### 3) Best-checkpoint tracking
Do not overwrite best model when validation degrades.

Maintain:

```
outreachlm_checkpoint.pt   # latest resumable state
outreachlm_best.pt         # best by validation loss only
outreachlm_model.pt        # final/export artifact
```

Update `outreachlm_best.pt` only when validation loss improves.

Decision logic:

```
train -> update weights -> evaluate
    if val_loss improves:
        save BEST
    else:
        continue
```

---

## Current structure checkpoint

### Training engine status
- Data-driven corpus loading from `corpus/fineweb`
- Tokenizer built from training split
- Random minibatch sampling
- Forward -> loss -> backward -> AdamW
- Validation evaluation
- Resumable checkpoint flow (`--resume`)

### Artifacts now in lifecycle
- `outreachlm_checkpoint.pt` (resume)
- `outreachlm_model.pt` (model state)
- `outreachlm_tokenizer.json` (tokenizer metadata)

---

## Operational note
Do not start another long training run yet.

First implement:
1. interval-averaged training metrics
2. scheduled validation monitoring
3. best-checkpoint logic

Then resume from the current step-2000 checkpoint.
