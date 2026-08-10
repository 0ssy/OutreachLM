# NEXT — Self-Attention
We're at the point where OutreachLM stops merely **representing tokens** and starts **connecting tokens to one another**.

So far:

```
Text
 ↓
BPE
 ↓
Token IDs
 ↓
Token Embeddings
 ↓
Positional Embeddings
 ↓
Combined representations
```

Now:

```
Combined representations
        ↓
   SELF-ATTENTION
```

## 1. The problem attention solves
Suppose the model sees:

```
"The animal didn't cross the street because it was tired."
```

What does **"it"** refer to?

The model needs to examine relationships between tokens.

Self-attention lets each token ask:

> "Which other tokens in this context are relevant to me?"
> For example:

```
animal ───────────────┐
                      ↓
"The animal didn't cross the street because it was tired."
                                      ↑
                                     it
```

The representation of `it` can place more attention on `animal`.

That's the fundamental idea.

---

# 2. Query, Key, Value
Self-attention creates three representations from every token:

```
Q = Query
K = Key
V = Value
```

Think of them conceptually as:

### Query

> What information am I looking for?

### Key

> What kind of information do I contain?

### Value

> What information should I actually provide?
For every token:

```
embedding
    │
    ├──→ Query
    ├──→ Key
    └──→ Value
```

These are produced using learned linear transformations.

genui{"physics_motion_forces":{"type_id":"NEWTON_SECOND_LAW","locale_override":"en-US"}}

Ignore the physics widget above if it renders; it isn't part of our LLM lesson.

The important equations are:

```
Q = XW_Q
K = XW_K
V = XW_V
```

where:

```
X   = combined token + position representations
W_Q = learned query weights
W_K = learned key weights
W_V = learned value weights
```

---

# 3. How does one token decide what matters?
We compare a query against keys.

The standard Transformer calculates:

```
QKᵀ
```

This produces a **score matrix**.

If we have 5 tokens:

```
        token 1 token 2 token 3 token 4 token 5

token 1    ?       ?       ?       ?       ?
token 2    ?       ?       ?       ?       ?
token 3    ?       ?       ?       ?       ?
token 4    ?       ?       ?       ?       ?
token 5    ?       ?       ?       ?       ?
```

Each row represents:

> How much should this token pay attention to every other token?

---

# 4. Scaling
The raw scores can become large.

So we divide by:

```
√d_k
```

where `d_k` is the dimension of the key vectors.

The score becomes:

```
QKᵀ / √d_k
```

Then we apply **softmax**.

Softmax converts the scores into weights that sum to 1.

For example:

```
[2.1, 0.4, -0.2, 1.3]
```

might become something approximately like:

```
[0.59, 0.11, 0.06, 0.24]
```

Now the model has an attention distribution.

---

# 5. Then we use the Values
The final operation is:

```
Attention(Q,K,V)
=
softmax(QKᵀ / √d_k)V
```

So the complete flow is:

```
                X
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
       Q        K        V
       │        │        │
       └────┬───┘        │
            ↓            │
          QKᵀ            │
            ↓            │
         scale            │
            ↓            │
         softmax          │
            │             │
            └──────┬──────┘
                   ↓
            weighted values
                   ↓
              attention
```

This is the core mechanism of the Transformer.

---

# 6. But OutreachLM is a language model
There's another critical problem.

When predicting the next token, the model **cannot look into the future**.

Suppose:

```
Input:

"The cat sat on"
```

The model is trying to predict:

```
"the"
```

It cannot cheat by looking at tokens after that position.

So we use a **causal mask**.

Example:

```
        1   2   3   4   5

1       ✓   ✗   ✗   ✗   ✗
2       ✓   ✓   ✗   ✗   ✗
3       ✓   ✓   ✓   ✗   ✗
4       ✓   ✓   ✓   ✓   ✗
5       ✓   ✓   ✓   ✓   ✓
```

Token 1 can see itself.

Token 2 can see tokens 1–2.

Token 3 can see tokens 1–3.

And so on.

This is what makes it **autoregressive**.

---

# 7. Let's implement the first attention layer
Now create a new class in your `model.py`.

Keep your existing `TokenEmbedding` and `PositionalEmbedding`.

Add:

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

Notice that we're not using a prebuilt Transformer.

We're explicitly building:

```
Q
K
V
scores
scaling
softmax
weighted values
```

ourselves.

---

# 8. Test it
In `main.py`, import:

```
from outreachlm.model import (
    TokenEmbedding,
    PositionalEmbedding,
    SelfAttention
)
```

Then:

```
attention = SelfAttention(
    embedding_dim=16
)

attention_output = attention(
    combined
)

print("\n==================================================")
print("SELF-ATTENTION TEST")
print("==================================================")

print("Input shape:")
print(combined.shape)

print("\nAttention output shape:")
print(attention_output.shape)
```

You should get:

```
==================================================
SELF-ATTENTION TEST
==================================================

Input shape:
torch.Size([1, 5, 16])

Attention output shape:
torch.Size([1, 5, 16])
```

That's important.

Attention has taken:

```
[1, 5, 16]
```

and returned:

```
[1, 5, 16]
```

The sequence length hasn't changed.

The embedding dimension hasn't changed.

But the **information contained in each token representation has changed**, because every token has now incorporated information from the other tokens it was allowed to attend to.

---

# 9. We aren't done with attention yet
Don't jump ahead to the Transformer block.

Our first implementation deliberately leaves out two things we'll implement next:

```
1. Causal masking
2. Multi-head attention
```

We need to understand both properly before assembling the Transformer.

The progression is:

```
Single-head self-attention
        ↓
Causal self-attention
        ↓
Multi-head attention
        ↓
Causal multi-head attention
        ↓
Feed-forward network
        ↓
Residual connections
        ↓
Layer normalization
        ↓
Transformer block
```

That is where we're going.

---

# Notes — continue your BPE/OutreachLM notes

```
==================================================
OUTREACHLM NOTES — SELF-ATTENTION
==================================================

PURPOSE
--------------------------------------------------
Self-attention allows each token representation to
incorporate information from other tokens in the
same context.

This allows the model to represent relationships
between tokens.

QUERY, KEY, VALUE
--------------------------------------------------
Every input representation X is transformed into:

Q = XW_Q
K = XW_K
V = XW_V

Q = Query
K = Key
V = Value

ATTENTION SCORES
--------------------------------------------------
Queries are compared against keys:

QK^T

This produces a score describing how strongly each
token should attend to other tokens.

SCALING
--------------------------------------------------
The scores are divided by the square root of the
key dimension:

QK^T / sqrt(d_k)

This prevents excessively large values before
softmax.

SOFTMAX
--------------------------------------------------
Softmax converts attention scores into weights.

The weights across each attention row sum to 1.

SCALED DOT-PRODUCT ATTENTION
--------------------------------------------------
Attention(Q,K,V)
=
softmax(QK^T / sqrt(d_k))V

CAUSAL ATTENTION
--------------------------------------------------
For autoregressive language modeling, a token cannot
attend to future tokens.

Example:

        1   2   3   4   5

1       ✓   ✗   ✗   ✗   ✗
2       ✓   ✓   ✗   ✗   ✗
3       ✓   ✓   ✓   ✗   ✗
4       ✓   ✓   ✓   ✓   ✗
5       ✓   ✓   ✓   ✓   ✓

SHAPES
--------------------------------------------------
Input:

(batch, sequence_length, embedding_dim)

Example:

(1, 5, 16)

Attention output:

(1, 5, 16)

CURRENT ARCHITECTURE
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
Self-Attention ← CURRENT
   ↓
Causal Mask
   ↓
Multi-Head Attention
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
Next Token
   ↓
Loss
   ↓
Backpropagation
```

**Next: we add the causal mask and make the attention genuinely autoregressiv**
