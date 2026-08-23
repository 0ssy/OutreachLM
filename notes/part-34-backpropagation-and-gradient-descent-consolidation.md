# OUTREACHLM — BACKPROPAGATION + GRADIENT DESCENT

## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.
You already have the forward pass:

```
tokens
  ↓
token embeddings
  ↓
position embeddings
  ↓
LayerNorm
  ↓
Q K V
  ↓
scaled dot-product attention
  ↓
softmax
  ↓
attention values
  ↓
output projection
  ↓
residual
  ↓
LayerNorm
  ↓
MLP
  ↓
residual
  ↓
output head
  ↓
logits
  ↓
cross entropy
  ↓
LOSS
```

Now we're going **backward through the exact same graph**:

```
LOSS
  │
  ▼
dL/dlogits
  │
  ▼
output head
  │
  ▼
MLP
  │
  ▼
residual
  │
  ▼
LayerNorm
  │
  ▼
attention output projection
  │
  ▼
attention
  │
  ├── dL/dV
  ├── dL/dAttention
  │       │
  │       ▼
  │     softmax
  │       │
  │       ▼
  │     scores
  │       │
  │       ├── Q
  │       └── K
  │
  ▼
token embeddings
  │
  ▼
PARAMETER GRADIENTS
  │
  ▼
GRADIENT DESCENT
  │
  ▼
UPDATED PARAMETERS
```

That is the system we're implementing.

---

# 1. The central idea
Suppose the model has a parameter:

```
w = 3
```

and produces:

```
y = wx
```

For:

```
x = 2
target = 10
```

we get:

```
y = 3 × 2
  = 6
```

Loss:

```
L = (y - target)²
  = (6 - 10)²
  = 16
```

The model is wrong.

Backpropagation asks:

> **How much did each parameter contribute to that error?**
For `w`:

[
L=(wx-t)^2
]

Therefore:

# [
\frac{\partial L}{\partial w}
2(wx-t)x
]

giving:

```
dL/dw = -16
```

The negative sign means:

> Increasing `w` would reduce the loss.
So gradient descent does:

# [
w_{\text{new}}
w-\eta\frac{\partial L}{\partial w}
]

where `eta` is the learning rate.

If:

```
w = 3
gradient = -16
learning_rate = 0.1
```

then:

```
w_new
= 3 - (0.1 × -16)
= 4.6
```

This is the entire idea behind training neural networks.

---

# 2. Gradient descent is NOT backpropagation
This distinction is important.

### Backpropagation
Calculates:

```
"Which direction should every parameter move?"
```

It produces:

```
∂L/∂W
∂L/∂b
∂L/∂embedding
...
```

### Gradient descent
Actually changes the parameters:

```
W ← W - learning_rate × ∂L/∂W
```

So:

```
FORWARD
   ↓
LOSS
   ↓
BACKPROP
   ↓
GRADIENTS
   ↓
GRADIENT DESCENT
   ↓
NEW PARAMETERS
```

Then the cycle repeats.

---

# 3. Implemented files
Created:

```
outreachlm/
│
├── gradient_descent.py
├── backprop_lab.py
│
├── attention.py
├── transformer_block.py
├── model.py
│
└── ...
```

The first two files are our **learning/verification infrastructure**.

---

# 4. Immediate implementation sequence
Run these **in this order**:

```
python -m outreachlm.gradient_descent
```

then:

```
python -m outreachlm.backprop_lab
```

The second program is intentionally comprehensive: **linear → MLP → LayerNorm → softmax → Q/K/V → attention → embeddings → complete Transformer**.

---

# 5. What to learn from output
For the full Transformer, confirm gradient path coverage:

```
Loss
 │
 ▼
Output Head
 │
 ▼
Transformer
 │
 ├───────────────┐
 ▼               ▼
MLP           Attention
 │               │
 ▼               ▼
LayerNorm      Output Projection
                 │
                 ▼
              Context
                 │
                 ▼
             Attention
             /   |   \
            Q    K    V
            │    │    │
            └────┴────┘
                 │
                 ▼
          Token representations
                 │
                 ▼
           Token Embeddings
```

Every trainable parameter should receive a gradient.

---

# 6. Most important formulas to keep
### Linear layer backward
Forward:

[
Y=XW^T+b
]

Backward:

# [
\frac{\partial L}{\partial W}
\frac{\partial L}{\partial Y}^T X
]

and:

# [
\frac{\partial L}{\partial X}
\frac{\partial L}{\partial Y}W
]

### Softmax + cross entropy key simplification

# [
\boxed{
\frac{\partial L}{\partial z}
p-y
}
]

### Q/K/V input gradient accumulation

# [
\frac{\partial L}{\partial X}
\frac{\partial L}{\partial X}_Q
+
\frac{\partial L}{\partial X}_K
+
\frac{\partial L}{\partial X}_V
]

---

# 7. Current status checkpoint
We have reached the training mechanism:

```
FORWARD PASS
    ↓
LOSS
    ↓
BACKPROPAGATE
    ↓
PARAMETER GRADIENTS
    ↓
GRADIENT DESCENT / OPTIMIZER
    ↓
UPDATED PARAMETERS
```

Next implementation target:

```
forward
→ loss
→ backward
→ per-parameter gradients
→ gradient norms
→ optimizer step
→ second forward
→ loss reduction
```