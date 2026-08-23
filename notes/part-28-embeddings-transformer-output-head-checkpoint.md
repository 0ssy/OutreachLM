## What it is
This part documents a concrete step in the OutreachLM build/refactor/research timeline and records the implementation or experiment state reached at that point.

## Why it is there
This note exists to preserve chronological traceability of decisions, commands, outputs, and outcomes so later phases can build on verified history instead of assumptions.

## Why it is important
It provides continuity across phases, supports reproducibility of results, and makes architecture/training decisions auditable when comparing future changes.

## What would happen without it
Without this record, decision context and result provenance would degrade, making regressions harder to diagnose and increasing risk of repeating failed approaches.
notes
You're right. We continued implementing without maintaining the **study notes**, and that breaks the way we're approaching this project.

From this point forward, we'll do both:

1. **Implement the next component.**
2. **Maintain cumulative engineering notes** so you have a record of what you actually learned and built.
I won't restart the notes from zero. I'll continue from the last checkpoint.

# OutreachLM — Engineering Notes

## Checkpoint: From Embeddings → Transformer → Position → Output

### 1. Dataset and next-token objective
We started with a deliberately tiny dataset:

```
Total tokens: 20
Context length: 5
Dataset length: 15
```

Example:

```
Input : [0, 1, 2, 3, 4]
Target: [1, 2, 3, 4, 5]
```

The important idea is that the target is shifted one position forward.

For:

```
[0, 1, 2, 3, 4]
```

the model is learning:

```
0 → 1
1 → 2
2 → 3
3 → 4
4 → 5
```

This gives us a controlled environment for testing whether the architecture can actually learn a predictable relationship.

---

# 2. Token Embeddings
We built:

```
class TokenEmbedding(nn.Module):
```

using:

```
nn.Embedding(
    vocab_size,
    embedding_dim
)
```

The purpose is to transform discrete token IDs into continuous vectors.

For example:

```
Token ID
   3
   │
   ▼
Embedding table
   │
   ▼
[16-dimensional vector]
```

With:

```
embedding_dim = 16
```

our input:

```
[1, 5]
```

becomes:

```
[1, 5, 16]
```

where:

```
B = batch size
T = sequence/context length
D = embedding dimension
```

Therefore:

```
[B, T] → [B, T, D]
```

---

# 3. Self-Attention
We then built `CausalSelfAttention`.

The architecture creates three projections:

```
Q = Query
K = Key
V = Value
```

using learned linear transformations.

We calculate:

```
Q @ Kᵀ
```

which produces an attention-score matrix:

```
[B, T, T]
```

The matrix answers:

> How strongly should each position interact with every other position?
We then scale the scores by:

```
√D
```

to prevent the dot products from becoming excessively large.

---

# 4. Causal Mask
We added the lower-triangular causal mask:

```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

This prevents a position from accessing information from the future.

We actually tested it.

Your output showed:

```
Future-token attention values:

0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
```

and:

```
✓ Causal mask is working.
✓ No token is attending to a future token.
```

That was an important verification rather than an assumption.

---

# 5. Feed-Forward Network
We added the feed-forward component inside the Transformer block.

Its role is different from attention.

Attention allows tokens to **exchange information**.

The feed-forward network then performs a learned nonlinear transformation on the representation at each position.

Conceptually:

```
Attention
   │
   ▼
contextual information
   │
   ▼
Feed Forward
   │
   ▼
transformed representation
```

---

# 6. Transformer Block
We combined attention and feed-forward processing into:

```
TransformerBlock
```

Your current block follows:

```
x
│
▼
LayerNorm
│
▼
CausalSelfAttention
│
▼
Residual addition
│
▼
LayerNorm
│
▼
FeedForward
│
▼
Residual addition
│
▼
output
```

We verified:

```
✓ Transformer block preserves shape.
✓ Causal attention is inside the block.
✓ Feed-forward network is inside the block.
✓ Residual connections are working.
✓ Layer normalization is working.
```

The representation remains:

```
[B, T, D]
```

---

# 7. Residual Connections
We use:

```
x = x + attention_output
```

and:

```
x = x + feed_forward_output
```

These are residual/skip connections.

Instead of forcing every sublayer to completely replace the representation, the network learns a modification to the existing representation.

Conceptually:

```
original representation
        │
        ├───────────────┐
        │               │
        ▼               ▼
     sublayer       original
        │               │
        └───────+───────┘
                │
                ▼
             result
```

This becomes extremely important as networks become deeper.

---

# 8. Positional Embeddings
We then discovered a fundamental problem.

A token embedding alone doesn't inherently encode **where the token occurs**.

The same token:

```
1
```

has the same embedding whether it appears at:

```
position 0
```

or:

```
position 4
```

Your experiment proved this:

```
Token 1 at position 0
=
Token 1 at position 4

Maximum difference:
0.0
```

Then we added positional embeddings.

The representation becomes:

```
token representation
        +
position representation
        =
position-aware representation
```

Your second experiment showed:

```
Token 1 at position 0
Token 1 at position 4

Maximum difference:
2.1953368186950684
```

Therefore:

```
✓ Same token tested at two positions.
✓ Token embeddings are identical.
✓ Position changes the representation.
✓ Positional information is being added.
```

This is another component we **experimentally verified**.

---

# 9. Current Architecture
At this point our machine is:

```
                  TOKEN IDs
                     │
                     ▼
              Token Embedding
                     │
                     +
                     │
             Positional Embedding
                     │
                     ▼
          Position-aware representation
                     │
                     ▼
             Transformer Block
                     │
          ┌──────────┴──────────┐
          │                     │
     Causal Attention      Feed Forward
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
             contextual vectors
```

Current shape:

```
Input IDs
[B, T]

        ↓

Embeddings
[B, T, 16]

        ↓

Transformer
[B, T, 16]
```

---

# 10. Where We Are NOW
The Transformer currently produces a **representation**.

It does not yet produce a prediction.

We have:

```
[B, T, 16]
```

but our vocabulary contains:

```
20 tokens
```

So we need to transform:

```
16 dimensions
```

into:

```
20 output scores
```

This gives:

```
[B, T, 16]
       │
       ▼
Linear(16 → 20)
       │
       ▼
[B, T, 20]
```

Those 20 values are **logits**.

---

# NEXT IMPLEMENTATION — Output Head
Now we actually implement that.

In your `OutreachModel.__init__()` add:

```
self.output_head = nn.Linear(
    embedding_dim,
    vocab_size
)
```

Then at the end of `forward()`:

```
logits = self.output_head(x)

return logits
```

So the complete forward path becomes:

```
Input IDs
   │
   ▼
Token Embedding
   │
   +
Positional Embedding
   │
   ▼
Transformer Block
   │
   ▼
[B, T, 16]
   │
   ▼
Output Head
Linear(16 → 20)
   │
   ▼
[B, T, 20]
```

---

## What we're going to test
Don't train yet.

First run the model and print:

```
Input shape
Token embedding shape
Position-aware shape
Transformer output shape
Logits shape
```

We want:

```
Input:
torch.Size([1, 5])

Token embeddings:
torch.Size([1, 5, 16])

Position-aware:
torch.Size([1, 5, 16])

Transformer:
torch.Size([1, 5, 16])

Logits:
torch.Size([1, 5, 20])
```

Then we'll inspect something like:

```
Position 0:
token 0 → score ...
token 1 → score ...
token 2 → score ...
...
token 19 → score ...
```

That will let us see what an **untrained model's prediction space** actually looks like.

### And only after that:

```
logits
   ↓
target alignment
   ↓
cross-entropy
   ↓
loss
   ↓
backpropagation
   ↓
optimizer
   ↓
parameter update
```

That is the point where the machine will stop merely **processing information** and start **learning from error**.

We'll keep the notes synchronized with every component from here onward.
