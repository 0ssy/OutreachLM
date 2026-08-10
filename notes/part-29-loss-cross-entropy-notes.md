# Next. We now make the model **measure its own error**.

So far:

```
Input
  ↓
Embeddings
  ↓
Position
  ↓
Attention
  ↓
Feed-forward
  ↓
Transformer
  ↓
Output head
  ↓
Logits
```

The missing piece is:

```
Logits + Target → Loss
```

## 1. What is the loss?
Your dataset gives us the correct answer:

```
Input:
[0, 1, 2, 3, 4]

Target:
[1, 2, 3, 4, 5]
```

The model produces 20 scores at every position:

```
position 0 → 20 possible tokens
position 1 → 20 possible tokens
position 2 → 20 possible tokens
position 3 → 20 possible tokens
position 4 → 20 possible tokens
```

We want:

```
position 0 → token 1
position 1 → token 2
position 2 → token 3
position 3 → token 4
position 4 → token 5
```

The **loss** converts the model's mistakes into a number.

For example:

```
Loss = 3.21
```

means the current parameters are producing relatively poor predictions.

After training we hope to see something like:

```
3.21
 ↓
2.47
 ↓
1.63
 ↓
0.82
 ↓
0.21
 ↓
...
```

The exact values will depend on initialization and training.

---

# 2. Cross-entropy loss
For our classification problem, we'll use:

```
nn.CrossEntropyLoss()
```

Why classification?

Because at every position the model chooses among:

```
20 possible token IDs
```

So each position is effectively a 20-class prediction problem.

The model doesn't directly say:

```
"I predict token 3."
```

Instead it produces:

```
token 0 → score
token 1 → score
token 2 → score
...
token 19 → score
```

Cross-entropy tells us how much probability mass the model effectively assigned to the **correct answer**.

---

# 3. Implement the first loss calculation
For now, don't build the optimizer.

Let's modify `main.py` to calculate the loss.

```
import torch
import torch.nn as nn

from outreachlm.model import OutreachModel

VOCAB_SIZE = 20
CONTEXT_LENGTH = 5
EMBEDDING_DIM = 16

model = OutreachModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM
)

# Training example
input_ids = torch.tensor([
    [0, 1, 2, 3, 4]
])

target_ids = torch.tensor([
    [1, 2, 3, 4, 5]
])

# Forward pass
logits = model(input_ids)

print("=" * 60)
print("OUTREACHLM LOSS TEST")
print("=" * 60)

print()
print("Input:")
print(input_ids)

print()
print("Target:")
print(target_ids)

print()
print("Logits shape:")
print(logits.shape)

# Cross-entropy loss
loss_function = nn.CrossEntropyLoss()

# CrossEntropyLoss expects:
#
# predictions:
# [batch, classes, sequence]
#
# targets:
# [batch, sequence]
#
# Our logits are:
# [batch, sequence, classes]
#
# Therefore we transpose dimensions 1 and 2.

loss = loss_function(
    logits.transpose(1, 2),
    target_ids
)

print()
print("Loss:")
print(loss.item())
```

Run:

```
python -m outreachlm.main
```

You should get something approximately like:

```
============================================================
OUTREACHLM LOSS TEST
============================================================

Input:
tensor([[0, 1, 2, 3, 4]])

Target:
tensor([[1, 2, 3, 4, 5]])

Logits shape:
torch.Size([1, 5, 20])

Loss:
something around 3
```

Don't worry about the exact number.

Because the model is **randomly initialized**, it hasn't learned the sequence yet.

---

# 4. Why do we transpose?
This is worth understanding.

Our model produces:

```
[B, T, V]
```

where:

```
B = batch
T = sequence length
V = vocabulary size
```

Currently:

```
[1, 5, 20]
```

But `CrossEntropyLoss` expects the class dimension in position 1:

```
[B, C, T]
```

So:

```
logits.transpose(1, 2)
```

changes:

```
[1, 5, 20]
```

into:

```
[1, 20, 5]
```

The targets remain:

```
[1, 5]
```

Therefore:

```
Predictions:
[1, 20, 5]

Targets:
[1, 5]
```

and PyTorch understands:

```
5 positions
×
20 possible classes
```

---

# 5. What is actually being compared?
Imagine the model produces:

```
Position 0:

token 0 → 0.3
token 1 → 0.8   ← correct
token 2 → -0.4
...
token 19 → 0.1
```

The target says:

```
target = 1
```

So the loss asks:

> Did the model give a strong enough score to token 1?
Then:

```
Position 1
```

target is:

```
2
```

and the same process happens.

Therefore the total loss combines the error across all five predictions.

Conceptually:

```
                Target
                  │
                  ▼
Model ────────→ Prediction
                  │
                  ▼
              Compare
                  │
                  ▼
                 LOSS
```

---

# 6. We have now completed another major piece
Our system is becoming:

```
                 INPUT
                   │
                   ▼
             Token IDs
                   │
                   ▼
           Token Embeddings
                   │
                   +
                   │
          Positional Embeddings
                   │
                   ▼
          Transformer Block
                   │
          ┌────────┴─────────┐
          │                  │
      Attention         Feed Forward
          │                  │
          └────────┬─────────┘
                   │
                   ▼
              [B,T,D]
                   │
                   ▼
              Output Head
                   │
                   ▼
              [B,T,V]
                   │
                   ▼
                LOGITS
                   │
                   │
            ┌──────┴──────┐
            │             │
         Targets       Predictions
            │             │
            └──────┬──────┘
                   ▼
               CROSS
               ENTROPY
                   │
                   ▼
                  LOSS
```

### Current status
We have built:

- Dataset
- Token embeddings
- Positional embeddings
- Causal self-attention
- Causal masking
- Feed-forward network
- Layer normalization
- Residual connections
- Transformer block
- Output projection
- Logits
- Cross-entropy loss
And we've tested the important shape transitions.

---

# The next step is different
**The machine currently knows how wrong it is.**

But it doesn't know how to become less wrong.

That's where gradients enter.

Next we'll perform:

```
loss
  ↓
loss.backward()
  ↓
gradients
  ↓
inspect what changed
```

Before introducing an optimizer, I want you to **see the gradients themselves**.

That is the transition from:

> "We built a neural network."
to:

> **"We built a system capable of modifying itself from error."**
