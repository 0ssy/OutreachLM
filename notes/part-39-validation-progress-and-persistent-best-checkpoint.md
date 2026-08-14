# OutreachLM — Validation Progress + Persistent Best Checkpoint

## Confirmed learning progression
Current measured results:

| Step | Train loss | Validation loss | Perplexity | Accuracy |
|---|---:|---:|---:|---:|
| 1,000 | 2.4244 | 2.3926 | 10.9416 | 31.73% |
| 2,000 | 2.2518 | 2.2527 | 9.5130 | 35.18% |
| 5,000 | 2.0859 | **2.0727** | **7.9466** | **39.76%** |
| 6,000 | 2.0211 | **2.0324** | **7.6326** | **40.87%** |

Key signal:

```
Validation loss: 2.3926 -> 2.2527 -> 2.0727 -> 2.0324
```

This is real held-out improvement, not just training memorization.

Resume run confirmation:

```
--resume from step 5000 to 6000
validation loss improved to 2.032433
best checkpoint updated (outreachlm_best.pt)
```

---

## Small stability fix: tensor copy warning removed
Observed warning during training:

```
UserWarning: To copy construct from a tensor...
```

Source was [datasets.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/datasets.py) constructing a new tensor from an existing tensor.

Applied fix:

```python
self.token_ids = token_ids.detach().clone()
```

This removes the warning and makes ownership/copy intent explicit without changing training behavior.

Best checkpoint creation confirmed:

```
outreachlm_best.pt
```

---

## Subtle state persistence issue (now addressed)
If `best_validation_loss` is reset to `inf` on resume, the trainer can incorrectly mark a later worse model as “new best” relative to historical runs.

Correct approach:
- save `best_validation_loss` inside checkpoint
- load it when resuming
- continue best tracking from persisted value

Required checkpoint state:

```python
checkpoint = {
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
    "loss": float(loss),
    "best_validation_loss": float(best_validation_loss),
}
```

Load contract:

```python
return (
    checkpoint["step"],
    checkpoint.get("best_validation_loss", float("inf"))
)
```

Train-state behavior:

```python
start_step, best_validation_loss = load_checkpoint(...)
```

Fresh run behavior:

```python
start_step = 0
best_validation_loss = float("inf")
```

Checkpoint save call:

```python
save_checkpoint(
    CHECKPOINT_PATH,
    model,
    optimizer,
    step,
    config,
    loss.item(),
    best_validation_loss
)
```

---

## Current lifecycle (stable baseline)

```
train step
  -> forward
  -> loss
  -> backward
  -> AdamW update
  -> checkpoint save
  -> periodic validation
       -> if val improves: save outreachlm_best.pt
```

---

## Next stage direction
After this infrastructure stabilization:
- stop adding basic training plumbing
- improve sampling efficiency for the large corpus regime

Target flow:

```
3.3M characters
   -> token IDs
   -> random minibatch sampling
   -> transformer
   -> loss
   -> backprop
   -> AdamW
   -> checkpoint
   -> validation
```

Do not increase model size yet; prioritize pipeline efficiency and reliability first.

---

## Update: direct random minibatch sampling enabled
Training now samples minibatches directly from tokenized training IDs each step, instead of sampling via dataset indexing patterns intended for full shuffled passes.

### New training sampling flow

```
training_text
  -> tokenizer.encode(...)
  -> training_token_ids
  -> random start positions
  -> fixed context windows (inputs/targets)
  -> minibatch
  -> model
```

`get_random_batch(...)` now operates on:
- `token_ids`
- `context_length`
- `batch_size`
- `device`

and returns:
- `inputs` shape `[B, T]`
- `targets` shape `[B, T]`

with next-token shift preserved.

### Deterministic validation preserved
Validation is intentionally still deterministic using validation token IDs and a validation dataset/evaluator path, so step-to-step validation metrics remain comparable.

### Split consistency preserved
The same corpus split remains:

```
full corpus text
  -> train/validation split
  -> training_token_ids (random minibatch sampling)
  -> validation_token_ids (deterministic evaluation)
```

No split logic was moved into minibatch sampling.

---

## Immediate validation plan
Run a short continuation window first (about +1000 steps) to confirm:

1. loss/logging behavior remains stable
2. checkpoint/resume still works
3. validation trend remains coherent relative to prior baseline

Only then proceed to longer schedules.
