# OutreachLM — Complete Training-State Checkpoint

## Goal
Upgrade checkpointing from a partial snapshot to a full training-state record so `--resume` reflects the real optimization state.

## What changed
In [train.py](C:/Users/josep/OneDrive/Desktop/OutreachLM/outreachlm/train.py), checkpoint payload now stores explicit training metrics instead of one ambiguous `loss` field.

Checkpoint fields now include:

```python
{
    "step": step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "last_loss": float(last_loss),
    "average_train_loss": float(average_train_loss),
    "validation_loss": float(validation_loss),
    "best_validation_loss": float(best_validation_loss),
    "config": {
        "context_length": CONTEXT_LENGTH,
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    },
}
```

## Why this matters
`last_loss` and averaged interval training loss represent different things:

- `last_loss`: one minibatch at the current step
- `average_train_loss`: mean over the reporting interval

Separating them removes bookkeeping ambiguity during resume and analysis.

## Resume behavior
`load_checkpoint(...)` now restores:

- model weights
- AdamW internal optimizer state
- resume step
- persisted `best_validation_loss`

and uses:

```python
torch.load(..., weights_only=False)
```

`--resume` now also checks checkpoint existence before loading.

## Interval tracking update
Training now tracks interval loss with:

```python
interval_losses = []
```

Each step appends `loss.item()`. At log boundaries, the engine computes:

```python
average_train_loss = sum(interval_losses) / len(interval_losses)
```

then clears interval storage.

## Checkpoint timing behavior
Checkpoint saves still occur on save interval/final step, but now include:

- current `last_loss`
- latest computed `average_train_loss`
- latest known `validation_loss`
- persisted `best_validation_loss`

If `average_train_loss` or `validation_loss` has not yet been computed at save time, it is computed before saving so checkpoint state remains complete.

## Artifact separation remains clean
- training continuation state: `outreachlm_checkpoint.pt`
- best validation model weights: `outreachlm_best.pt`
- deployable model weights: `outreachlm_model.pt`
- tokenizer state: `outreachlm_tokenizer.json`

This preserves a clean separation between training-state continuity and deployable artifacts.
