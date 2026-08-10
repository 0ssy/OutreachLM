notes continue 
You're right. **OutreachLM should not be framed as "only a language model."** What we're building is a trainable neural architecture whose current training objective is next-token prediction. The same core machinery can later become part of other ML systems.

And yes—physics is useful when the concept genuinely connects to what we're building. I won't artificially avoid a field just because we're working on AI.

Now let's continue exactly where we stopped.

# NEXT — Causal Masking
Our current attention implementation has a serious problem.

Right now, every token can attend to **every other token**:

```
1 → 1 2 3 4 5
2 → 1 2 3 4 5
3 → 1 2 3 4 5
4 → 1 2 3 4 5
5 → 1 2 3 4 5
```

For an autoregressive model, that's cheating.

If token 2 is supposed to predict token 3, it cannot already see token 3.

We therefore impose a **causal constraint**.

genui{"physics_motion_forces":{"type_id":"NEWTON_FIRST_LAW","locale_override":"en-US"}}

The useful connection here isn't that attention follows a physical law—it doesn't. Rather, we're imposing a directional constraint on information flow, much like many physical systems have constraints on what states can influence what other states. In our model, the mask is an explicit computational constraint.

---

## 1. The causal mask
For a sequence of length 5:

```
        key
        ↓
       1  2  3  4  5
    ┌───────────────
q 1 │  ✓  ✗  ✗  ✗  ✗
  2 │  ✓  ✓  ✗  ✗  ✗
  3 │  ✓  ✓  ✓  ✗  ✗
  4 │  ✓  ✓  ✓  ✓  ✗
  5 │  ✓  ✓  ✓  ✓  ✓
```

Mathematically:

```
M[i,j] =
    0       if j ≤ i
   -∞       if j > i
```

Why `-∞`?

Because we're going to add the mask to the attention scores **before softmax**.

Suppose:

```
scores = [2.0, 1.0, 3.0, 4.0, 2.5]
```

If position 1 isn't allowed to see positions 2–5:

```
masked scores =
[2.0, -∞, -∞, -∞, -∞]
```

After softmax:

```
[1.0, 0.0, 0.0, 0.0, 0.0]
```

The forbidden positions effectively disappear.

That's the key trick.

---

# 2. Build the mask
Let's modify `SelfAttention`.

Your current version was approximately:

```
class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (K.size(-1) ** 0.5)

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = attention_weights @ V

        return output
```

We're going to change it.

---

# 3. Causal Self-Attention
Replace it with:

```
class SelfAttention(nn.Module):

    def __init__(self, embedding_dim, context_length):
        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        mask = torch.tril(
            torch.ones(
                context_length,
                context_length
            )
        )

        self.register_buffer(
            "mask",
            mask
        )

    def forward(self, x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        scores = scores / (K.size(-1) ** 0.5)

        sequence_length = x.size(1)

        causal_mask = self.mask[
            :sequence_length,
            :sequence_length
        ]

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        output = attention_weights @ V

        return output
```

There are several things worth understanding here.

---

# 4. `torch.tril`
This:

```
torch.tril(
    torch.ones(
        5,
        5
    )
)
```

produces:

```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

That's exactly our causal structure.

The lower triangle is allowed.

The upper triangle is forbidden.

---

# 5. Why `register_buffer`?
This line:

```
self.register_buffer("mask", mask)
```

is important.

The mask isn't a trainable parameter.

We don't want gradient descent modifying it.

But we **do** want PyTorch to treat it as part of the model's state.

So:

```
Parameter
→ learned

Buffer
→ stored with model, but not learned
```

The causal mask belongs in the second category.

---

# 6. Update `main.py`
Because `SelfAttention` now needs `context_length`, change:

```
attention = SelfAttention(
    embedding_dim=16
)
```

to:

```
attention = SelfAttention(
    embedding_dim=16,
    context_length=5
)
```

Run:

```
python -m outreachlm.main
```

You should still get:

```
Input shape:
torch.Size([1, 5, 16])

Attention output shape:
torch.Size([1, 5, 16])
```

The shape hasn't changed.

The **information flow has**.

---

# 7. Let's actually inspect the attention
Right now our class only returns the output.

For learning, I want us to expose the attention weights.

Change:

```
return output
```

to:

```
return output, attention_weights
```

Then in `main.py`:

```
attention_output, attention_weights = attention(
    combined
)

print("\nAttention weights:")
print(attention_weights)
```

You should see something shaped approximately:

```
torch.Size([1, 5, 5])
```

Conceptually, it should look like:

```
[
 [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],

 [0.xxxx, 0.xxxx, 0.0000, 0.0000, 0.0000],

 [0.xxxx, 0.xxxx, 0.xxxx, 0.0000, 0.0000],

 [0.xxxx, 0.xxxx, 0.xxxx, 0.xxxx, 0.0000],

 [0.xxxx, 0.xxxx, 0.xxxx, 0.xxxx, 0.xxxx]
]
```

The exact values will differ because the weights are randomly initialized.

But the **zeros above the diagonal must remain**.

That's our first genuinely important architectural invariant.

---

# 8. Test the invariant
We can make the test explicit.

Add:

```
future_attention = torch.triu(
    attention_weights,
    diagonal=1
)

print("\nFuture attention:")
print(future_attention)

print(
    "\nCausal constraint satisfied:",
    torch.all(future_attention == 0).item()
)
```

You want:

```
Causal constraint satisfied: True
```

Now we're not merely looking at a tensor and saying "looks right."

We're testing an architectural property.

That's how we'll build OutreachLM from here onward:

> **Every important component should have an invariant that we can test.**

---

# 9. Important distinction
Causal masking does **not** mean the model can only understand previous words.

It means that **during prediction at position `t`**, information from positions `> t` is unavailable.

So:

```
position 1
↓
can use 1

position 2
↓
can use 1,2

position 3
↓
can use 1,2,3

position 4
↓
can use 1,2,3,4
```

During training, all positions can be processed **in parallel**.

That's one of the major advantages of Transformer training.

We don't have to literally run:

```
token 1
wait
token 2
wait
token 3
...
```

The mask makes the parallel computation behave as though each position only had access to its permitted history.

---

# 10. One thing I want you to notice
We now have:

```
X
↓
Q K V
↓
QKᵀ
↓
scale
↓
CAUSAL MASK
↓
softmax
↓
weighted V
↓
output
```

This is a complete **causal self-attention mechanism**.

But it is still only **one attention head**.

The next step is where the architecture becomes much more interesting:

# Multi-Head Attention
Instead of having one set of:

```
Q K V
```

we'll have several.

For example, with:

```
embedding_dim = 16
num_heads = 4
```

each head gets:

```
16 / 4 = 4
```

dimensions.

Conceptually:

```
                 X
                 │
      ┌──────────┼──────────┐
      ↓          ↓          ↓
    Head 1      Head 2     Head 3 ... Head 4
      │          │          │
      └──────────┼──────────┘
                 ↓
             concatenate
                 ↓
          output projection
```

Different heads can learn different relationships.

One may become useful for nearby syntax.

Another may capture longer-range dependencies.

Another may focus on structural patterns.

We don't hard-code what each head learns—the optimization process determines that.

---

# Notes — continue

```
==================================================
OUTREACHLM NOTES — CAUSAL SELF-ATTENTION
==================================================

WHY CAUSAL MASKING EXISTS
--------------------------------------------------
An autoregressive model predicts future tokens.

Therefore a token must not have access to tokens
that occur after it.

Without masking, training would allow information
from the future to leak into the prediction.

CAUSAL MASK
--------------------------------------------------
For sequence length 5:

        1   2   3   4   5

1       ✓   ✗   ✗   ✗   ✗
2       ✓   ✓   ✗   ✗   ✗
3       ✓   ✓   ✓   ✗   ✗
4       ✓   ✓   ✓   ✓   ✗
5       ✓   ✓   ✓   ✓   ✓

MASK IMPLEMENTATION
--------------------------------------------------
A lower-triangular matrix is created using:

torch.tril()

Allowed positions contain:

1

Forbidden positions contain:

0

Before softmax:

allowed score → unchanged

forbidden score → -infinity

After softmax:

forbidden positions → 0

REGISTER_BUFFER
--------------------------------------------------
The causal mask is not trainable.

It is stored using:

register_buffer()

Parameters:
learned by optimization.

Buffers:
stored as model state but not optimized.

ATTENTION WEIGHTS
--------------------------------------------------
For sequence length N:

attention weights have shape:

(batch, N, N)

Example:

(1, 5, 5)

The upper triangular region must contain zero
attention weights.

ARCHITECTURAL INVARIANT
--------------------------------------------------
For causal attention:

attention[i,j] = 0

whenever:

j > i

This should be tested explicitly.

CURRENT ATTENTION PIPELINE
--------------------------------------------------

Input X
   ↓
Q = XW_Q
K = XW_K
V = XW_V
   ↓
QK^T
   ↓
divide by sqrt(d_k)
   ↓
causal mask
   ↓
softmax
   ↓
weighted V
   ↓
attention output

CURRENT OUTREACHLM ARCHITECTURE
--------------------------------------------------

Raw Text
   ↓
Corpus Loader
   ↓
BPE Tokenizer
   ↓
Token IDs
   ↓
Dataset
   ↓
Token Embedding
   ↓
Positional Embedding
   ↓
Combined Representation
   ↓
Causal Self-Attention
   ↓
[NEXT] Multi-Head Attention
   ↓
Feed-Forward Network
   ↓
Residual Connections
   ↓
Layer Normalization
   ↓
Transformer Block
   ↓
Output Projection
   ↓
Logits
   ↓
Loss
   ↓
Backpropagation
```

**Next: Multi-Head Attention.**
