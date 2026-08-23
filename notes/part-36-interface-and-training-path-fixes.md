# OutreachLM — Notes Update: Interface + Training Path Fixes

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.

## What was fixed

### 1) Model/block interface consistency (tuple propagation bug)
We fixed the architecture mismatch where a transformer block returned:

```
(x, attention_weights)
```

but the model loop treated it as:

```
x = block(x)
```

This caused:

```
TypeError: layer_norm(): argument 'input' must be Tensor, not tuple
```

#### Correct flow now

```
for block in self.transformer_blocks:
    x, block_attention_weights = block(x)
    attention_weights.append(block_attention_weights)
```

and model forward returns:

```
return logits, attention_weights
```

So attention data is preserved for inspection while only tensors flow into LayerNorm and subsequent layers.

---

### 2) Experiment scripts aligned to model API
Because the model now returns a tuple:

```
(logits, attention_weights)
```

scripts that only need logits now unpack with:

```
logits, _ = model(...)
```

This was applied in transition/generalization/analysis experiment scripts where needed.

---

### 3) Windows console encoding issue in transition experiment
The transition script previously used Unicode status symbols (`✓`, `✗`, `△`) which failed on cp1252 terminals.

They were replaced with ASCII-safe markers:

```
[OK], [X], [~]
```

Result: the transition experiment now runs to completion on the current Windows environment.

---

### 4) Robust corpus/model paths in training script
`train.py` previously used fragile working-directory-relative strings.

Now it uses file-relative absolute resolution:

```
PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_PATH = PROJECT_DIR / "data1" / "train.txt"
MODEL_PATH = PROJECT_DIR / "outreachlm_model.pt"
```

This removes ambiguity when running:

```
python -m outreachlm.train
```

from different working directories.

---

## Current technical status

### Architecture/API
- Transformer block returns `(tensor, attention_weights)` intentionally.
- Model aggregates per-block attention weights and returns `(logits, attention_weights)`.
- Call sites now unpack appropriately.

### Training/evaluation scripts
- Tuple interface mismatch resolved.
- Transition experiment executes end-to-end in current terminal.
- Training path resolution is now robust and deterministic.

---

## Key engineering lesson from this phase
Most recent failures were **integration/interface issues**, not failures of learning theory.

We validated that:
1. Model outputs and consumer expectations must stay synchronized.
2. Observability data (attention weights) should be explicit in interfaces.
3. Runtime environment details (console encoding, working directory) are part of production reliability.

---

## Next checkpoint
Continue curriculum from gradient flow into:

```
forward
→ loss
→ backward
→ per-parameter gradient norms
→ optimizer update
→ second forward
→ observed loss change
```

with direct instrumentation on the actual OutreachLM training engine.